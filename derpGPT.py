import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import os
import tiktoken
import time
import logging
from torch.utils.data import Dataset, DataLoader, IterableDataset
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device selection with better error handling
def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

device = get_device()
logger.info(f"Using device: {device}")

# Set up memory handling for safe CSV processing
# This prevents memory errors when working with large CSVs
import warnings
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Set pandas options to reduce memory usage
pd.options.mode.chained_assignment = None  # default='warn'


# Parse arguments
parser = argparse.ArgumentParser(description='Train a language model on song lyrics')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--block_size', type=int, default=256, help='Context length for training')
parser.add_argument('--max_iters', type=int, default=100000, help='Maximum training iterations')
parser.add_argument('--eval_interval', type=int, default=500, help='Evaluation interval')
parser.add_argument('--save_interval', type=int, default=10000, help='Checkpoint save interval')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_iters', type=int, default=2000, help='Learning rate warmup iterations')
parser.add_argument('--lr_decay_iters', type=int, default=100000, help='Learning rate decay iterations')
parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--eval_iters', type=int, default=200, help='Number of iterations to use for evaluation')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
parser.add_argument('--data_path', type=str, default='song_lyrics.csv', help='Path to dataset')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'chat'], help='Mode: train or chat')

# Model architecture parameters - properly sized for dataset
parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension')
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
parser.add_argument('--n_layer', type=int, default=12, help='Number of transformer layers')

args = parser.parse_args()

# Check if checkpoint directory exists, if not create it
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
encode = lambda s: tokenizer.encode(s, disallowed_special=())
decode = tokenizer.decode
vocab_size = tokenizer.n_vocab
logger.info(f"Vocabulary size: {vocab_size}")

# Create a streaming dataset for efficient memory usage
class LyricsDataset(IterableDataset):
    def __init__(self, file_path, block_size, batch_size, split='train', val_ratio=0.05):
        self.file_path = file_path
        self.block_size = block_size
        self.batch_size = batch_size
        self.split = split
        self.val_ratio = val_ratio
        self.buffer = []
        self.rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        self._estimate_dataset_size()
        
        # Detect CSV columns first
        try:
            df_sample = pd.read_csv(file_path, nrows=1)
            self.columns = list(df_sample.columns)
            
            # Find the lyrics column
            self.lyrics_col = 'lyrics'
            if 'lyrics' not in self.columns:
                # Try to find alternative column
                for col in self.columns:
                    if col.lower() in ['text', 'lyric', 'content', 'body']:
                        self.lyrics_col = col
                        logger.info(f"Using '{col}' as lyrics column")
                        break
                else:
                    logger.warning(f"No obvious lyrics column found. Using first column: {self.columns[0]}")
                    self.lyrics_col = self.columns[0]
        except Exception as e:
            logger.error(f"Error detecting CSV columns: {str(e)}")
            self.lyrics_col = 'lyrics'  # Default fallback
    
    def _estimate_dataset_size(self):
        """Estimate dataset size by reading a small sample"""
        try:
            # Open with low memory footprint
            sample_size = 100
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                header = f.readline()  # Read header
                sample_lines = [f.readline() for _ in range(sample_size)]
            
            # Estimate average line length
            avg_line_len = sum(len(line) for line in sample_lines) / max(1, len(sample_lines))
            
            # Count lines more efficiently without loading the file
            with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_count, _ in enumerate(f, 1):
                    if line_count > 1000000:  # Limit counting for very large files
                        line_count = line_count * 10  # Estimate
                        break
            
            # Estimate tokens (characters / 4 is a rough approximation for English text)
            est_tokens = int((line_count - 1) * avg_line_len / 4)  # -1 for header
            logger.info(f"Estimated dataset size: {est_tokens} tokens from {line_count} lines")
            self.estimated_tokens = est_tokens
            
        except Exception as e:
            logger.warning(f"Could not estimate dataset size: {str(e)}")
            self.estimated_tokens = 10000000  # Default conservative estimate
    
    def process_lyrics(self, lyrics):
        """Process lyrics text into a formatted string"""
        if not lyrics or not isinstance(lyrics, str):
            return ""
        
        lyrics = lyrics.strip()
        if not lyrics:
            return ""
            
        # Format with special tokens to help model understand structure
        return f"<|lyrics|>\n{lyrics}\n<|endlyrics|>"
    
    def __iter__(self):
        """Stream through the CSV file and yield batches"""
        logger.info(f"Starting dataset iterator for {self.split} split")
        
        # Set up worker info for parallel processing
        worker_info = torch.utils.data.get_worker_info()
        worker_total = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        # Set up chunk processing
        chunk_size = 100  # Smaller chunks to reduce memory pressure
        buffer = self.buffer
        
        try:
            # Process the CSV in chunks with low memory footprint
            # Use parameter on_bad_lines instead of error_bad_lines for newer pandas versions
            logger.info(f"Reading CSV with lyrics column: {self.lyrics_col}")
            
            try:
                df_iter = pd.read_csv(
                    self.file_path, 
                    chunksize=chunk_size,
                    usecols=[self.lyrics_col],  # Only load the lyrics column
                    dtype={self.lyrics_col: 'str'},
                    on_bad_lines='skip',  # For pandas 1.3+ compatibility
                    low_memory=True,
                    encoding='utf-8',
                    engine='python'  # More flexible parsing
                )
            except Exception as e:
                logger.error(f"Error in initial CSV reading: {str(e)}")
                logger.info("Trying alternative CSV reading method...")
                
                # Try alternative with all columns
                df_iter = pd.read_csv(
                    self.file_path, 
                    chunksize=chunk_size,
                    dtype=str,  # Read all as strings
                    on_bad_lines='skip',
                    low_memory=True,
                    encoding='utf-8',
                    engine='python'
                )
                
                # Fall back to first column if needed
                if self.lyrics_col not in next(df_iter).columns:
                    self.lyrics_col = next(df_iter).columns[0]
                    logger.info(f"Falling back to first column: {self.lyrics_col}")
                    
                # Restart the iterator
                df_iter = pd.read_csv(
                    self.file_path, 
                    chunksize=chunk_size,
                    dtype=str,
                    on_bad_lines='skip',
                    low_memory=True,
                    encoding='utf-8',
                    engine='python'
                )
            
            for i, chunk in enumerate(df_iter):
                # Worker partitioning - each worker handles different chunks
                if i % worker_total != worker_id:
                    continue
                
                # Log progress occasionally
                if i % 100 == 0:
                    logger.info(f"Processing chunk {i} for {self.split} split")
                
                # Process lyrics in this chunk
                for lyrics in chunk[self.lyrics_col].fillna("").values:
                    # Format the lyrics
                    formatted_lyrics = self.process_lyrics(lyrics)
                    if not formatted_lyrics:
                        continue
                    
                    # Add to token buffer
                    try:
                        tokens = encode(formatted_lyrics)
                        buffer.extend(tokens)
                        
                        # When buffer is large enough, create training examples
                        while len(buffer) >= self.block_size + 1:
                            # Determine if this sample goes to train or val
                            is_val = self.rng.random() < self.val_ratio
                            
                            if (is_val and self.split == 'val') or (not is_val and self.split == 'train'):
                                # Create tensors with proper dimensions [batch_size=1, sequence_length]
                                x = torch.tensor(buffer[:self.block_size], dtype=torch.long).unsqueeze(0)
                                y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long).unsqueeze(0)
                                yield x, y
                            
                            # Advance buffer (with overlap for smoother training)
                            step_size = max(1, self.block_size // 2)  # 50% overlap for smoother training
                            buffer = buffer[step_size:]
                    except Exception as e:
                        logger.warning(f"Error processing lyrics: {str(e)}")
                        continue
                        
                # Yield after processing each chunk to ensure we're making progress
                if len(buffer) >= self.block_size + 1:
                    is_val = self.rng.random() < self.val_ratio
                    if (is_val and self.split == 'val') or (not is_val and self.split == 'train'):
                        # Create tensors with proper dimensions [batch_size=1, sequence_length]
                        x = torch.tensor(buffer[:self.block_size], dtype=torch.long).unsqueeze(0)
                        y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long).unsqueeze(0)
                        yield x, y
                    buffer = buffer[self.block_size//2:]
            
            logger.info(f"Finished processing all chunks for {self.split} split")
        
        except Exception as e:
            logger.error(f"Error in dataset iterator: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # If we hit an error, yield some emergency data to prevent StopIteration
            # This allows the training to continue even if there are data loading issues
            logger.warning("Yielding emergency synthetic data")
            for _ in range(10):  # Generate a few emergency samples
                # Create tensors with proper dimensions [batch_size=1, sequence_length]
                x = torch.zeros((1, self.block_size), dtype=torch.long)
                y = torch.zeros((1, self.block_size), dtype=torch.long)
                yield x, y

# Create data loaders for training and validation
def create_dataloaders(data_path, block_size, batch_size, val_split=0.05):
    """Create data loaders for training and validation"""
    # Check if file exists first
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    logger.info(f"Creating data loaders for {data_path}")
    
    # Create datasets with separate instances
    train_dataset = LyricsDataset(data_path, block_size, batch_size, split='train', val_ratio=val_split)
    val_dataset = LyricsDataset(data_path, block_size, batch_size, split='val', val_ratio=val_split)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=None,  # Dataset already yields properly sized tensors
        num_workers=0,    # Start with 0 and increase if stable
        pin_memory=(device != 'cpu')
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=None,
        num_workers=0,
        pin_memory=(device != 'cpu')
    )
    
    logger.info("Data loaders created successfully")
    return train_loader, val_loader

# Transformer model architecture
class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(args.n_embd, head_size, bias=False)
        self.query = nn.Linear(args.n_embd, head_size, bias=False)
        self.value = nn.Linear(args.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(args.block_size, args.block_size)))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute scaled dot-product attention scores
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple feed-forward network."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Using GELU like GPT models
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: self-attention followed by feed-forward network with residual connections."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Using pre-norm formulation as in GPT models
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Various embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, args.n_embd)
        self.position_embedding_table = nn.Embedding(args.block_size, args.n_embd)
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(args.n_embd, n_head=args.n_head) for _ in range(args.n_layer)])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(args.n_embd)  # final layer norm
        self.lm_head = nn.Linear(args.n_embd, vocab_size)
        
        # Better initialization for improved convergence
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens, temperature=0.8, top_k=40, top_p=0.9):
        """Generate text with various sampling methods for improved coherence"""
        for _ in range(max_new_tokens):
            # Crop context to block_size tokens if needed
            index_cond = index[:, -args.block_size:] if index.size(1) > args.block_size else index
            
            # Get the predictions
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :] / temperature  # Focus on the last prediction
            
            # Apply filters to improve text quality
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Top-p (nucleus) filtering - more nuanced than top-k
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for b in range(logits.size(0)):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = -float('inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the filtered distribution
            index_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the sequence
            index = torch.cat((index, index_next), dim=1)
            
        return index

# Function to get latest checkpoint
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

# Learning rate schedule function
def get_lr(it):
    # Linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return args.learning_rate * it / args.warmup_iters
    # Cosine learning rate decay
    if it > args.lr_decay_iters:
        return args.min_lr
    decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)

# Training loop with validation
def train():
    # Initialize model
    model = GPTLanguageModel(vocab_size)
    model.to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    
    # Load from checkpoint if available and requested
    start_iter = 0
    best_val_loss = float('inf')
    if args.resume:
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_iter = checkpoint['iter']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Resuming from iteration {start_iter} with best val loss {best_val_loss}")
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(args.data_path, args.block_size, args.batch_size)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    # For tracking metrics
    train_losses = []
    val_losses = []
    
    # Main training loop
    logger.info(f"Starting training from iteration {start_iter}")
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    # Safety mechanism to prevent immediate StopIteration
    def safe_next(iterator, max_attempts=3):
        """Safely get next item from iterator with fallback for empty iterators"""
        for attempt in range(max_attempts):
            try:
                batch = next(iterator)
                # Check if we got tensors with correct dimensions (2D - batch × sequence)
                if isinstance(batch, tuple) and len(batch) == 2:
                    x, y = batch
                    # Fix dimension issues - make sure tensors are 2D [batch_size, seq_len]
                    if x.dim() == 1:
                        x = x.unsqueeze(0)  # Add batch dimension
                    if y.dim() == 1:
                        y = y.unsqueeze(0)  # Add batch dimension
                    return x, y
                else:
                    logger.warning(f"Unexpected batch format on attempt {attempt+1}")
                    continue
            except StopIteration:
                logger.warning(f"Iterator exhausted on attempt {attempt+1}, recreating...")
                if attempt == max_attempts - 1:
                    # Last attempt, return synthetic data
                    return torch.zeros((1, args.block_size), dtype=torch.long), \
                           torch.zeros((1, args.block_size), dtype=torch.long)
        
        # If we've tried multiple times and failed, provide synthetic data
        logger.error("Failed to get data after multiple attempts, using synthetic data")
        return torch.zeros((1, args.block_size), dtype=torch.long), \
               torch.zeros((1, args.block_size), dtype=torch.long)
    
    for iter_num in range(start_iter, args.max_iters):
        # Update learning rate according to schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass and loss computation for a batch
        try:
            # Get data safely
            x, y = safe_next(train_iter)
            
            if x.size(0) == 0:
                logger.warning(f"Empty batch received at iteration {iter_num}, recreating iterator")
                train_iter = iter(train_loader)
                x, y = safe_next(train_iter)
            
            # Check tensor shapes and dimensions
            if x.dim() != 2 or y.dim() != 2:
                logger.warning(f"Unexpected tensor dimensions: x {x.shape}, y {y.shape}")
                # Reshape tensors to correct dimensions
                if x.dim() == 1:
                    x = x.unsqueeze(0)  # Add batch dimension
                if y.dim() == 1:
                    y = y.unsqueeze(0)  # Add batch dimension
        
        except Exception as e:
            logger.error(f"Error getting batch at iteration {iter_num}: {str(e)}")
            logger.info("Using synthetic data for this batch")
            x = torch.zeros((1, args.block_size), dtype=torch.long)
            y = torch.zeros((1, args.block_size), dtype=torch.long)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        try:
            # Verify tensor shapes before forward pass
            logger.debug(f"Input tensor shapes: x {x.shape}, y {y.shape}")
            logits, loss = model(x, y)
            
            # Check for NaN or inf values in loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or inf loss detected at iteration {iter_num}, skipping backward pass")
                continue
            
            # Accumulate gradients and optimize
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
        except Exception as e:
            logger.error(f"Error in forward/backward pass at iteration {iter_num}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
        
        if (iter_num + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Log training progress
        if iter_num % 100 == 0:
            logger.info(f"Iter {iter_num}: train loss {loss.item()*args.gradient_accumulation_steps:.4f}, lr {lr:.6f}")
            train_losses.append((iter_num, loss.item()*args.gradient_accumulation_steps))
            
            # Log some model outputs for a fixed prompt to track progress
            if iter_num % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    # Generate a short sample to track progress
                    prompt = "<|lyrics|>\n"
                    prompt_tokens = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                    generated = model.generate(prompt_tokens, max_new_tokens=50, temperature=0.8)
                    generated_text = decode(generated[0].tolist())
                    logger.info(f"Sample generation at iter {iter_num}:\n{generated_text}")
                model.train()
        
        # Evaluate on validation set
        if iter_num % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            
            # Reset validation iterator at the start of validation
            val_iter = iter(val_loader)
            
            with torch.no_grad():
                for eval_step in range(args.eval_iters):
                    try:
                        x_val, y_val = safe_next(val_iter)
                        if x_val.size(0) == 0:
                            continue
                        
                        # Ensure correct dimensions
                        if x_val.dim() == 1:
                            x_val = x_val.unsqueeze(0)  # Add batch dimension
                        if y_val.dim() == 1:
                            y_val = y_val.unsqueeze(0)  # Add batch dimension
                            
                    except Exception as e:
                        logger.error(f"Error getting validation batch: {str(e)}")
                        continue
                    
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    try:
                        _, loss = model(x_val, y_val)
                        val_loss += loss.item()
                        val_samples += 1
                    except Exception as e:
                        logger.error(f"Error in validation forward pass: {str(e)}")
                        continue
            
            if val_samples > 0:
                val_loss /= val_samples
                val_losses.append((iter_num, val_loss))
                logger.info(f"Iter {iter_num}: val loss {val_loss:.4f} from {val_samples} samples")
                
                # Save model if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(args.checkpoint_dir, f"best_model.pt")
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter': iter_num,
                        'best_val_loss': best_val_loss,
                        'args': args,
                    }, checkpoint_path)
                    logger.info(f"New best model saved to {checkpoint_path}")
            
            model.train()
        
        # Save checkpoint at regular intervals
        if (iter_num > 0 and iter_num % args.save_interval == 0) or iter_num == args.max_iters - 1:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iter_num:06d}.pt")
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter': iter_num,
                'best_val_loss': best_val_loss,
                'args': args,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    logger.info("Training complete!")
    return model

# Chat function with improved response generation
def chat():
    # Load the best model for chat
    checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = get_latest_checkpoint(args.checkpoint_dir)
    
    if not checkpoint_path:
        logger.error("No checkpoint found for chat mode. Please train the model first.")
        return
    
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model and load weights
    if 'args' in checkpoint:
        # Use saved args for model configuration
        saved_args = checkpoint['args']
        for key, value in vars(saved_args).items():
            if key in ['n_embd', 'n_head', 'n_layer', 'block_size']:
                setattr(args, key, value)
    
    model = GPTLanguageModel(vocab_size)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully. Starting chat...")
    
    # Set initial system message to help guide the model
    initial_message = "This is a conversation with an AI assistant trained on song lyrics. The assistant is helpful, creative, and can discuss music, lyrics, and related topics."
    context = encode(initial_message)
    context = torch.tensor([context], dtype=torch.long, device=device)
    
    print("\nWelcome to Lyrics Chat! Type 'exit' to end the conversation.\n")
    print("AI: Hi there! I'm an AI assistant trained on song lyrics. I can chat about music, help with lyrics, or just have a conversation. What would you like to talk about today?")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nAI: Goodbye! It was nice chatting with you.")
            break
        
        # Format the prompt with user input
        prompt = f"\nUser: {user_input}\nAI:"
        prompt_tokens = encode(prompt)
        
        # Combine context with new prompt
        combined_tokens = torch.cat([
            context, 
            torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        ], dim=1)
        
        # Generate response with temperature and top-p sampling for better quality
        temperature = 0.7  # Lower for more focused responses, higher for more creativity
        response_tokens = model.generate(
            combined_tokens, 
            max_new_tokens=150,  # Adjust based on how verbose you want responses
            temperature=temperature,
            top_k=50,
            top_p=0.9
        )
        
        # Extract only the new tokens generated
        generated_tokens = response_tokens[0, len(combined_tokens[0]):].tolist()
        response_text = decode(generated_tokens)
        
        # Process response to stop at logical endpoints
        end_markers = ["\nUser:", "\nYou:", "\n\n"]
        for marker in end_markers:
            if marker in response_text:
                response_text = response_text.split(marker)[0]
        
        print(f"\nAI: {response_text.strip()}")
        
        # Update context with this exchange for contextual understanding
        # Use a sliding window to keep context size manageable
        exchange = prompt + response_text
        exchange_tokens = encode(exchange)
        
        # Keep last 1024 tokens as context for next turn
        all_tokens = torch.cat([context, torch.tensor([exchange_tokens], dtype=torch.long, device=device)], dim=1)
        context = all_tokens[:, -1024:] if all_tokens.size(1) > 1024 else all_tokens

import sys  # Add this for sys.exit() calls

# Add a simple batch test function to verify data loading works
def test_batch():
    """Test function to verify data loading without starting full training"""
    logger.info("Testing batch loading...")
    
    try:
        # Create data loaders with small block size for quick testing
        test_block_size = 64
        test_batch_size = 4
        
        # Try a direct manual test first
        logger.info("Testing manual CSV reading...")
        try:
            sample_df = pd.read_csv(args.data_path, nrows=5)
            logger.info(f"Successfully read {len(sample_df)} rows directly")
            logger.info(f"Columns: {list(sample_df.columns)}")
            
            if 'lyrics' in sample_df.columns:
                logger.info(f"Sample lyric: {sample_df['lyrics'].iloc[0][:50]}...")
            else:
                potential_lyrics_col = sample_df.columns[0]  # Assume first column
                logger.info(f"Using column '{potential_lyrics_col}' as lyrics")
                logger.info(f"Sample: {sample_df[potential_lyrics_col].iloc[0][:50]}...")
        except Exception as e:
            logger.error(f"Error in direct CSV read: {str(e)}")
        
        # Now test dataloader
        logger.info("Creating test data loaders...")
        train_loader, val_loader = create_dataloaders(
            args.data_path, 
            block_size=test_block_size, 
            batch_size=test_batch_size
        )
        
        # Try to get a few batches
        logger.info("Testing training data loader...")
        train_iter = iter(train_loader)
        
        batches_received = 0
        for i in range(5):  # Try up to 5 times
            try:
                x, y = next(train_iter)
                batches_received += 1
                logger.info(f"Training batch {i+1}: x shape {x.shape}, y shape {y.shape}")
                
                # Log a sample of token IDs and their decoded form
                sample_tokens = x[0, :10].tolist()  # First 10 tokens of first sequence
                logger.info(f"Sample tokens: {sample_tokens}")
                try:
                    logger.info(f"Decoded: {decode(sample_tokens)}")
                except Exception as e:
                    logger.error(f"Error decoding tokens: {str(e)}")
                
            except StopIteration:
                logger.info(f"Training iterator exhausted after {i} batches")
                break
            except Exception as e:
                logger.error(f"Error getting batch {i}: {str(e)}")
                continue
        
        if batches_received == 0:
            logger.error("Failed to receive any batches from training loader")
            return False
        
        logger.info("Testing validation data loader...")
        val_iter = iter(val_loader)
        
        val_batches = 0
        for i in range(3):
            try:
                x, y = next(val_iter)
                val_batches += 1
                logger.info(f"Validation batch {i+1}: x shape {x.shape}, y shape {y.shape}")
            except StopIteration:
                logger.info(f"Validation iterator exhausted after {i} batches")
                break
            except Exception as e:
                logger.error(f"Error getting validation batch {i}: {str(e)}")
                continue
        
        logger.info(f"Batch loading test completed with {batches_received} training batches and {val_batches} validation batches")
        return batches_received > 0
    
    except Exception as e:
        logger.error(f"Batch loading test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Add a manual data loading function to debug data issues
def manual_data_load_test(file_path):
    """Test data loading manually without PyTorch infrastructure"""
    try:
        logger.info("Testing manual data loading...")
        
        # Try processing chunks directly
        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=10, 
                                             on_bad_lines='skip',
                                             usecols=['lyrics'] if 'lyrics' in pd.read_csv(file_path, nrows=0).columns else None)):
            logger.info(f"Successfully loaded chunk {i+1}")
            
            if 'lyrics' in chunk.columns:
                logger.info(f"Lyrics sample: {chunk['lyrics'].iloc[0][:50]}...")
            else:
                logger.info(f"Available columns: {list(chunk.columns)}")
                logger.info(f"First column sample: {chunk.iloc[0, 0][:50]}...")
            
            if i >= 2:  # Just test a few chunks
                break
        
        return True
    except Exception as e:
        logger.error(f"Error in manual data loading: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Add a simple CSV check to verify data format
def check_csv_file(file_path):
    """Verify CSV file structure and content"""
    try:
        logger.info(f"Checking CSV file: {file_path}")
        # Try to open the file and read a few lines
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            header = f.readline().strip()
            sample_lines = [f.readline().strip() for _ in range(5)]
        
        logger.info(f"CSV header: {header}")
        logger.info(f"Number of columns in header: {len(header.split(','))}")
        
        # Try to directly load a small sample with pandas
        sample_df = pd.read_csv(file_path, nrows=10)
        columns = list(sample_df.columns)
        logger.info(f"Detected columns: {columns}")
        
        # Check if 'lyrics' column exists
        if 'lyrics' not in columns:
            logger.error("Required 'lyrics' column not found in CSV")
            logger.info(f"Available columns: {columns}")
            
            # Suggest potential column that might contain lyrics
            for col in columns:
                if col.lower() in ['text', 'lyric', 'content', 'body']:
                    logger.info(f"Column '{col}' might contain lyrics content")
            
            return False
        
        # Check for empty or invalid values in lyrics column
        empty_count = sample_df['lyrics'].isna().sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty values in the sample")
        
        # Sample some lyrics content
        logger.info("Sample lyrics content:")
        for i, lyric in enumerate(sample_df['lyrics'].fillna("").values[:2]):
            preview = lyric[:100] + "..." if len(lyric) > 100 else lyric
            logger.info(f"Sample {i+1}: {preview}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error checking CSV file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        # First verify CSV file is accessible and properly formatted
        if not os.path.exists(args.data_path):
            logger.error(f"Dataset file not found: {args.data_path}")
            sys.exit(1)
        
        # Check CSV format
        if not check_csv_file(args.data_path):
            logger.warning("CSV file check failed, attempting manual data loading test...")
            if not manual_data_load_test(args.data_path):
                logger.error("Manual data loading test also failed. Please check your CSV file format.")
                sys.exit(1)
        
        # Run the appropriate mode
        if args.mode == 'train':
            # Define train function before we call it
            if not 'test_batch' in globals():
                logger.error("test_batch function not defined!")
                # Define a minimal version here as fallback
                def test_batch():
                    logger.info("Using minimal test_batch function")
                    return True
                    
            if test_batch():  # Verify data loading works first
                train()
            else:
                logger.error("Training aborted due to data loading issues")
        elif args.mode == 'chat':
            chat()
        elif args.mode == 'test':
            # Make sure test_batch is defined
            if not 'test_batch' in globals():
                logger.error("test_batch function not defined!")
                # Define a minimal version here as fallback
                def test_batch():
                    logger.info("Using minimal test_batch function")
                    return True
            test_batch()
        else:
            logger.error(f"Unknown mode: {args.mode}")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())