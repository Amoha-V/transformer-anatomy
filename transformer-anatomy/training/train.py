"""
Train a tiny GPT-style transformer (char-level) on tinyshakespeare.
Architecture deliberately mirrors a minimal nanoGPT: token+pos embeddings,
N decoder blocks (LN -> causal MHA -> residual -> LN -> MLP(GELU) -> residual),
final LN, output head tied to token embedding.

We export:
  - weights.json   (every parameter, plain nested lists)
  - vocab.json      (char <-> id mappings)
  - training_log.json (loss curve: step, train_loss, val_loss)
  - sample.txt      (a generated sample so we can sanity check quality)
"""
import math, json, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
random.seed(1337)

# ---------------- config ----------------
D_MODEL   = 64
N_HEAD    = 4
N_LAYER   = 4
D_FF      = 256
BLOCK     = 64
BATCH     = 32
MAX_ITERS = 3000
LR        = 3e-3
EVAL_EVERY = 250
DEVICE = 'cpu'
torch.set_num_threads(1)

# ---------------- data ----------------
text = open('tinyshakespeare.txt').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for i,c in enumerate(chars)}
print('vocab_size', vocab_size)

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split=='train' else val_data
    ix = torch.randint(len(d)-BLOCK-1, (BATCH,))
    x = torch.stack([d[i:i+BLOCK] for i in ix])
    y = torch.stack([d[i+1:i+BLOCK+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# ---------------- model ----------------
HEAD_DIM = D_MODEL // N_HEAD

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.Wq = nn.Linear(D_MODEL, D_MODEL)
        self.Wk = nn.Linear(D_MODEL, D_MODEL)
        self.Wv = nn.Linear(D_MODEL, D_MODEL)
        self.Wo = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF)
        self.fc2 = nn.Linear(D_FF, D_MODEL)
        mask = torch.tril(torch.ones(BLOCK, BLOCK))
        self.register_buffer('mask', mask)

    def forward(self, x):
        B,T,C = x.shape
        h = self.ln1(x)
        q = self.Wq(h).view(B,T,N_HEAD,HEAD_DIM).transpose(1,2)
        k = self.Wk(h).view(B,T,N_HEAD,HEAD_DIM).transpose(1,2)
        v = self.Wv(h).view(B,T,N_HEAD,HEAD_DIM).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(HEAD_DIM)
        att = att.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        out = self.Wo(out)
        x = x + out
        h2 = self.ln2(x)
        h2 = self.fc2(F.gelu(self.fc1(h2)))
        x = x + h2
        return x

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(D_MODEL)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None,:,:]
        for blk in self.blocks:
            x = blk(x)
        x = self.lnf(x)
        logits = x @ self.tok_emb.weight.T   # weight tying
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

model = TinyGPT().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print('n_params', n_params)

opt = torch.optim.AdamW(model.parameters(), lr=LR)

import os
CKPT = 'ckpt.pt'
start_it = 0
log = []
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    start_it = ck['it'] + 1
    log = ck['log']
    print(f'resumed from checkpoint at step {ck["it"]}')

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train','val']:
        losses = []
        for _ in range(10):
            x,y = get_batch(split)
            _, loss = model(x,y)
            losses.append(loss.item())
        out[split] = sum(losses)/len(losses)
    model.train()
    return out

TIME_BUDGET = 240  # seconds per invocation; save+exit before the sandbox call timeout
t0 = time.time()
it = start_it
for it in range(start_it, MAX_ITERS+1):
    if it % EVAL_EVERY == 0:
        losses = estimate_loss()
        log.append({'step': it, 'train_loss': losses['train'], 'val_loss': losses['val']})
        print(f"step {it}: train {losses['train']:.4f} val {losses['val']:.4f}  ({time.time()-t0:.1f}s)")
    x,y = get_batch('train')
    logits, loss = model(x,y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if time.time() - t0 > TIME_BUDGET:
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'it': it, 'log': log}, CKPT)
        print(f'time budget hit at step {it}, checkpoint saved, rerun script to continue')
        raise SystemExit(0)

print('training complete at step', it)
torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'it': it, 'log': log}, CKPT)

# ---------------- sample ----------------
context = torch.zeros((1,1), dtype=torch.long)  # start token id 0
out_ids = model.generate(context, 400, temperature=0.8, top_k=10)[0].tolist()
sample_text = ''.join(itos[i] for i in out_ids)
open('sample.txt','w').write(sample_text)
print('---- sample ----')
print(sample_text)

# ---------------- export ----------------
def to_list(t):
    return t.detach().cpu().tolist()

weights = {
    'config': {
        'd_model': D_MODEL, 'n_head': N_HEAD, 'n_layer': N_LAYER,
        'd_ff': D_FF, 'block_size': BLOCK, 'vocab_size': vocab_size
    },
    'tok_emb': to_list(model.tok_emb.weight),       # [vocab, d_model]
    'pos_emb': to_list(model.pos_emb.weight),        # [block, d_model]
    'lnf_g': to_list(model.lnf.weight), 'lnf_b': to_list(model.lnf.bias),
    'blocks': []
}
for blk in model.blocks:
    weights['blocks'].append({
        'ln1_g': to_list(blk.ln1.weight), 'ln1_b': to_list(blk.ln1.bias),
        'Wq': to_list(blk.Wq.weight.T), 'bq': to_list(blk.Wq.bias),
        'Wk': to_list(blk.Wk.weight.T), 'bk': to_list(blk.Wk.bias),
        'Wv': to_list(blk.Wv.weight.T), 'bv': to_list(blk.Wv.bias),
        'Wo': to_list(blk.Wo.weight.T), 'bo': to_list(blk.Wo.bias),
        'ln2_g': to_list(blk.ln2.weight), 'ln2_b': to_list(blk.ln2.bias),
        'W1': to_list(blk.fc1.weight.T), 'b1': to_list(blk.fc1.bias),
        'W2': to_list(blk.fc2.weight.T), 'b2': to_list(blk.fc2.bias),
    })

with open('weights.json','w') as f:
    json.dump(weights, f)
with open('vocab.json','w') as f:
    json.dump({'stoi': stoi, 'itos': itos, 'chars': chars}, f)
with open('training_log.json','w') as f:
    json.dump({'log': log, 'n_params': n_params, 'final_train_loss': log[-1]['train_loss'], 'final_val_loss': log[-1]['val_loss']}, f)

print('exported weights.json, vocab.json, training_log.json')
import os
print('weights.json size (bytes):', os.path.getsize('weights.json'))
