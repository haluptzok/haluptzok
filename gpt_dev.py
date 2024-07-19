import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import math
import argparse
import os
import sys

# run with python gpt_dev.py | tee tmp.log
# This is from the collab notebook for lecture 5 on gpt from:
# https://github.com/karpathy/ng-video-lecture
# This is derived from the code in:
# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py

# idea: mask off gradient from all partial words that start a sequence
# The gradient of the first character only applies if it's the first character of the word
# The gradient of parital words should be masked off
# Will it help log-loss on train and test?
# Track the accuracy and loss on the trainset - running averages

# hyperparameters
batch_size = 512 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 1000
eval_iters = 4
eval_gen_final = 20000
if True: # debug/test for quick runs
    max_iters = 1001
    eval_interval = 1000
    eval_iters = 4
    eval_gen_final = 200

learning_rate = 1e-3
device = 'cuda' # 'cpu' is 3X slower, on path8 219s on cpu vs 70s on cuda
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

if False: # 0.2M parameters to 10.78M parameters, 16X slower even with cuda
    batch_size = 512 # how many independent sequences will we process in parallel?
    block_size = 256 # what is the maximum context length for predictions?
    # learning_rate = 3e-4
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------

print(f'using device: {device}')
# torch.set_default_device(device)  # This makes it slower too on both cuda and cpu - a lot slower on both (?)
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text_words = text.split()
biggest_word = ""
for words in text_words:
    if len(words) > len(biggest_word):
        biggest_word = words

print("biggest word:,", biggest_word, len(biggest_word))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(f"{itos=}")
print(f"{vocab_size=}")

time_start = time.time()
# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
print("len(data)", len(data), max_iters * batch_size)
n = int(0.9*len(data)) + 3 # first 90% will be train, rest val, add 3 to end at sentence boundary
trn_text = text[:n]
trn_data = data[:n]
val_text = text[n:]
val_data = data[n:]
# print("end of train:", text[n-100:n])
# print("begin of val:", text[n:n+100])

trn_index = 0
trn_len = len(trn_data) - block_size
print(f"{len(trn_data)=}")
print(f"{trn_len=}")
trn_indexes = torch.randperm(trn_len)

val_index = 0
val_len = len(val_data) - block_size
print(f"{len(val_data)=}")
print(f"{val_len=}")
val_indexes = torch.randperm(val_len)

epoch = 0

# data loading - this is simple - but maybe suboptimal for accuracy performance
# because it randomly samples, it skips some of the data, and doubles or triples up on other samples.
# so it probably converges more slowly that get_batch1 for the same number of iterations
# once nice thing is batches can start at all the indexes into the data.  But maybe it should only
# allow starting at beginnings of words or sentences, the model doesn't have to model partial word starts
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = trn_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# this is a better version of get_batch - it uses all the data, and doesn't skip any with the random sampling
# every index will be a start index of a batch before repeating
def get_batch1(split):
    ix = torch.zeros(batch_size, dtype=torch.long)
    i = 0
    data = None
    if split == 'train':
        data = trn_data
        global trn_index, trn_len, trn_indexes, epoch
        while i < batch_size:
            if trn_index >= trn_len:
                epoch += 1
                print("full_epoch", epoch)
                trn_index = 0
                trn_indexes = torch.randperm(trn_len)
            ix[i] = trn_indexes[trn_index]
            trn_index += 1
            i += 1
    else:
        global val_index, val_len, val_indexes
        data = val_data
        while i < batch_size:
            if val_index >= val_len:
                val_index = 0
                val_indexes = torch.randperm(val_len)
            ix[i] = val_indexes[val_index]
            val_index += 1
            i += 1
            
    # ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Dataset to predict the next character in the sequence
class GPTCharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        # trick to mask off the loss at some positions - when start is partial word
        # y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

trn_dataset = GPTCharDataset(trn_data, block_size)
val_dataset = GPTCharDataset(val_data, block_size)
trn_dataloader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

@torch.inference_mode()
def estimate_loss(model):
    # computes loss at each position - so some chars have 1 char of context
    # and some chars have block_size chars of context.
    # Save the current state of the data enumeration for training and validation
    # set the data enumeration to the beginning of the data, not randomized
    # so it's the exact same data every time
    global trn_index, trn_len, trn_indexes
    global val_index, val_len, val_indexes
    tmp_trn_index, tmp_trn_len, tmp_trn_indexes = trn_index, trn_len, trn_indexes
    tmp_val_index, tmp_val_len, tmp_val_indexes = val_index, val_len, val_indexes

    trn_index = 0
    trn_indexes = torch.arange(trn_len)

    val_index = 0
    val_indexes = torch.arange(val_len)

    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch1(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    trn_index, trn_len, trn_indexes = tmp_trn_index, tmp_trn_len, tmp_trn_indexes
    val_index, val_len, val_indexes = tmp_val_index, tmp_val_len, tmp_val_indexes
    return out

@torch.inference_mode()
def estimate_generate_loss(model, max_new_tokens=2000):
    # Compute the log-loss to generate the trainset and valset
    # by the model.  This is a better estimate of the log-loss than
    # estimate_loss because this uses the full context at each step of 
    # the generation, after generating the first block_size tokens.

    out = {}
    model.eval()

    for split in ['train', 'val']:
        split_data = trn_data if split == 'train' else val_data
        split_text = trn_text if split == 'train' else val_text

        score = 0.0 # sum of log-loss
        cCorrect = 0
        # generate from the model
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        # print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

        # def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for i_char in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = model(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) # sample from the distribution
            prob_next, idx_next = torch.max(probs, dim=1, keepdim=True) # (B, 1)  # take the best one
            # Did we predict the next character correctly?
            if idx_next[0,0] == split_data[i_char]:
                cCorrect += 1
            # Set the correct index to the next character
            idx_next[0,0] = split_data[i_char]
            # print(f"{idx_next[0]=}, {idx_next[0].tolist()=}, {decode(idx_next[0].tolist())=}")
            # print(f"{i_char=}, {split_text[i_char]=}, {split_data[i_char]=}, {decode(idx_next[0].tolist())=}, {prob_next=}")
            # Get the probability for the correct next character
            score += -torch.log(probs[0, split_data[i_char]]).item()

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        out[split] = (score, max_new_tokens, cCorrect)
        avg_log_prob = score / max_new_tokens
        print(f"GLos/{split[:3]} {(score/max_new_tokens):.4f}, prob={math.exp(-avg_log_prob):.4f}, {cCorrect=}/{max_new_tokens}")
    model.train()

    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

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
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False) # language model head

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

@torch.inference_mode()
def evaluate(model, dataset, batch_size, max_batches):
    model.eval()
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train() # reset model back to training mode
    return mean_loss

model = GPTLanguageModel()
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
print(sum(p.numel() for p in model.parameters()), 'M parameters')
if False:
    for p in model.parameters():
        print(p.shape, p.numel(), p.name)
model = model.to(device)
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

size = len(trn_dataloader.dataset)
print(f"trn_dataloader.dataset {size=}")
iter_train = iter(trn_dataloader)
epoch_trn = 0
t0 = time.time()
for step in range(max_iters):
    # sample a batch of data
    if False:
        xb, yb = get_batch1('train')
    else:
        try:
            xb, yb = next(iter_train)
        except StopIteration: # this will happen every epoch
            epoch_trn += 1
            print("epoch_trn step", epoch_trn, step)
            iter_train = iter(trn_dataloader)
            xb, yb = next(iter_train)
        # print(f"{xb.shape=}, {yb.shape=}")
        xb, yb = xb.to(device), yb.to(device)

    # evaluate the loss
    logits, loss = model(xb, yb)
    model.zero_grad(set_to_none=True)
    # optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        t1 = time.time()
        print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms | total time {(t1-time_start)*1000:.2f}ms")
        t0 = t1

    # every once in a while evaluate the loss on train and val sets
    if (step > 0 and step % eval_interval == 0) or step == max_iters - 1:
        t0 = time.time()
        train_loss = evaluate(model, trn_dataloader.dataset, batch_size=batch_size, max_batches=4)
        test_loss  = evaluate(model, val_dataset, batch_size=batch_size, max_batches=4)
        print(f"Loss/trn {train_loss:.4f}", step)
        print(f"Loss/tst {test_loss:.4f}", step)

        with torch.inference_mode():
            losses = estimate_loss(model)
            print(f"{step}: tra {losses['train']:.4f}, val {losses['val']:.4f}")
            estimate_generate_loss(model, eval_iters * batch_size)
        t1 = time.time()
        print(f"eval {step} | eval time {(t1-t0)*1000:.2f}ms | total time {(t1-time_start)*1000:.2f}ms")
        t0 = time.time()

time_end = time.time()
time_diff = time_end - time_start
print(f"Took {time_diff:.3f} seconds {(time_diff/60):.3f} minutes {(time_diff/3600):.3f} hours.\n")

# compute log-loss on the data set.
print("\nEstimate the log-loss on the train and val sets to generate them")
estimate_generate_loss(model, eval_gen_final)

# generate from the model
with torch.inference_mode():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=(eval_gen_final//10))[0].tolist()))

time_end = time.time()
time_diff = time_end - time_start
print(f"Took {time_diff:.3f} seconds {(time_diff/60):.3f} minutes {(time_diff/3600):.3f} hours.\n")
exit()

if False:
    # This was just me trying to figure out how much data is expected to be skipped in
    # 1 epoch with the simple random sampling with replacement
    # that Karpathy uses in his code.  Turns out it's a lot, 36.8%, 1/e.  So I wrote a
    # better version of get_batch that doesn't skip over any data - it uses all the data
    # every time.  Karpathy's version is simple and short - but in an epoch 1 / e of the
    # data won't be used.  So it's not a good way to train a model ususally, but for
    # keeping the code simple Karpathy's version is fine for educational purposes.

    torch.manual_seed(42)
    num_samples = 200000
    print("e 1/e", math.e, 1/math.e)

    for exp_samples in range(5):
        num_samples = 10 ** exp_samples
        ix = torch.randint(num_samples, (num_samples,))
        # iy is a histogram of the number of times each index is sampled
        iy = torch.zeros(num_samples)
        for i in ix:
            iy[i] += 1
        analytic_zeros = num_samples * math.pow(((num_samples - 1) / num_samples), num_samples)
        print("num_samples", num_samples, "analytic_zeros", analytic_zeros, "Actual_Zero", (iy == 0).sum().item())
        print("iy", iy.sum(), iy.max(), iy.min(), iy.mean(), iy.std(), iy.var(), iy.median(), "\n")
    exit()
