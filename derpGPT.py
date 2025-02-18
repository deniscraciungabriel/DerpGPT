import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import pickle
import argparse
import pandas as pd

# Device selection
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Hyperparameters for a larger (~50M param) model with word-level tokenization
batch_size = 32
block_size = 128  # now refers to a block of words
max_iters = 10000
learning_rate = 3e-5
eval_iters = 100
n_embd = 512
n_head = 8
n_layer = 12
dropout = 0.2

# ---------------- Data Loading from CSV ---------------- #
# Load CSV files for training and validation data
df_train = pd.read_csv("Synthetic-Persona-Chat_train.csv")
df_val   = pd.read_csv("Synthetic-Persona-Chat_valid.csv")

# Extract conversation text from the correct column
train_text = " ".join(df_train["Best Generated Conversation"].astype(str).tolist())
val_text   = " ".join(df_val["Best Generated Conversation"].astype(str).tolist())

# Combine both texts to build a unified vocabulary
full_text = train_text + " " + val_text
words = full_text.split()
vocab = sorted(set(words))
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Create word-to-index and index-to-word mappings
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

def encode(s):
    """Encodes a string into a list of integers (word-level)."""
    return [word_to_idx[w] for w in s.split() if w in word_to_idx]

def decode(vector):
    """Decodes a list of integers back into a string."""
    return " ".join(idx_to_word[i] for i in vector)

# Create data tensors from the CSV texts
training_data = torch.tensor(encode(train_text), dtype=torch.long)
validation_data = torch.tensor(encode(val_text), dtype=torch.long)

def get_batch(split):
    """Generates a batch of inputs and targets for training."""
    data_split = training_data if split == 'train' else validation_data
    ix = torch.randint(0, len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- Transformer Model Definition ---------------- #

class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

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
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

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
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: self-attention followed by feed-forward network."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffwd(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index, max_new_tokens):
        # Sequentially generate new tokens
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.forward(index_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=1)
        return index

# Instantiate and load model parameters (if available)
model = GPTLanguageModel(vocab_size)
print('Loading model parameters...')
try:
    with open('model-01.pkl', 'rb') as f:
        model = pickle.load(f)
    print('Loaded model successfully!')
except FileNotFoundError:
    print('No saved model found, starting from scratch.')

model = model.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Ask the user to choose between training or chatting
mode = input("Choose mode: 'train' or 'chat': ").strip().lower()

if mode == 'train':
    # Training loop
    for iter in range(max_iters):
        if iter % eval_iters == 0:
            losses = estimate_loss()
            print(f"Step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Final loss:", loss.item())

    with open('model-01.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Model saved.')

elif mode == 'chat':
    # Chatbot interaction loop using your conversation tags
    while True:
        # Get the user's input and format the prompt so the model generates a reply as User 2
        user_input = input("User 1: ")
        prompt = f"User 1: {user_input}\nUser 2:"  # model will complete as User 2
        
        # Encode the prompt and generate tokens
        context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        generated_tokens = model.generate(context, max_new_tokens=100)[0].tolist()
        generated_text = decode(generated_tokens)
        
        # Remove the prompt from the generated text and extract only User 2's response.
        # We assume the model may generate a new turn with "User 1:".
        response_section = generated_text[len(prompt):].strip()
        if "User 1:" in response_section:
            user2_response = response_section.split("User 1:")[0].strip()
        else:
            user2_response = response_section
        
        print(f"User 2: {user2_response}")

else:
    print("Invalid mode selected. Please choose 'train' or 'chat'.")

