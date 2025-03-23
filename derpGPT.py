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
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

device = get_device()
logger.info(f"Using device: {device}")

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
    def __init__(self, file_path, block_size, batch_size):
        self.file_path = file_path
        self.block_size = block_size
        self.batch_size = batch_size
        self._estimate_dataset_size()
        
    def _estimate_dataset_size(self):
        """Estimate dataset size by reading a small sample"""
        try:
            sample = pd.read_csv(self.file_path, nrows=1000)
            avg_lyrics_len = sample['lyrics'].str.len().mean()
            total_rows = sum(1 for _ in open(self.file_path, 'r')) - 1  # Subtract header
            self.estimated_tokens = int(total_rows * avg_lyrics_len / 4)  # Rough estimate
            logger.info(f"Estimated dataset size: {self.estimated_tokens} tokens")
        except Exception as e:
            logger.warning(f"Could not estimate dataset size: {str(e)}")
            self.estimated_tokens = 100000000  # Default estimate
    
    def process_chunk(self, chunk):
        """Process a chunk of lyrics data into tokens"""
        lyrics = chunk['lyrics'].fillna("").astype(str).tolist()
        # Add some context to help model learn the structure
        processed_lyrics = []
        for lyric in lyrics:
            if len(lyric.strip()) > 0:
                processed_lyrics.append(f"<|lyrics|>\n{lyric}\n<|endlyrics|>")
        
        # Join all processed lyrics with newlines
        all_text = "\n".join(processed_lyrics)
        return all_text
    
    def get_stream(self):
        """Generator that yields batches of tokens"""
        buffer = []
        buffer_size = 0
        target_buffer_size = self.block_size * self.batch_size * 10  # Target buffer size
        
        # Process the CSV in chunks to save memory
        for chunk in pd.read_csv(self.file_path, chunksize=1000):
            text = self.process_chunk(chunk)
            if not text:
                continue
                
            # Encode text to tokens
            tokens = encode(text)
            buffer.extend(tokens)
            buffer_size += len(tokens)
            
            # If buffer is large enough, start yielding batches
            while len(buffer) >= self.block_size + 1:
                x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
                yield x, y
                buffer = buffer[self.block_size:]
                
        # Process any remaining tokens in the buffer
        while len(buffer) >= self.block_size + 1:
            x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
            y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
            yield x, y
            buffer = buffer[self.block_size:]
    
    def __iter__(self):
        return self.get_stream()

# Create data loaders for training and validation
def create_dataloaders(data_path, block_size, batch_size, val_split=0.05):
    """Create data loaders for training and validation"""
    dataset = LyricsDataset(data_path, block_size, batch_size)
    
    # Split iterator into train and validation
    def train_val_split(iterator, val_ratio):
        for i, item in enumerate(iterator):
            if np.random.random() < val_ratio:
                yield (item, 'val')
            else:
                yield (item, 'train')
    
    def train_iter():
        for item, split in train_val_split(dataset.get_stream(), val_split):
            if split == 'train':
                yield item
    
    def val_iter():
        for item, split in train_val_split(dataset.get_stream(), val_split):
            if split == 'val':
                yield item
    
    train_loader = DataLoader(
        train_iter(), 
        batch_size=None,  # Iterator already returns batched data
        num_workers=0,    # Must be 0 for IterableDataset
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_iter(), 
        batch_size=None,
        num_workers=0,
        pin_memory=True
    )
    
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
    for iter_num in range(start_iter, args.max_iters):
        # Update learning rate according to schedule
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass and loss computation for a batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        
        # Accumulate gradients and optimize
        loss = loss / args.gradient_accumulation_steps
        loss.backward()
        
        if (iter_num + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Log training progress
        if iter_num % 100 == 0:
            logger.info(f"Iter {iter_num}: train loss {loss.item()*args.gradient_accumulation_steps:.4f}, lr {lr:.6f}")
            train_losses.append((iter_num, loss.item()*args.gradient_accumulation_steps))
        
        # Evaluate on validation set
        if iter_num % args.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for _ in range(args.eval_iters):
                    try:
                        x_val, y_val = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        x_val, y_val = next(val_iter)
                    
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    _, loss = model(x_val, y_val)
                    val_loss += loss.item()
            
            val_loss /= args.eval_iters
            val_losses.append((iter_num, val_loss))
            
            logger.info(f"Iter {iter_num}: val loss {val_loss:.4f}")
            
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

if __name__ == "__main__":
    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()
    else:
        logger.error(f"Unknown mode: {args.mode}")