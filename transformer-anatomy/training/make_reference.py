import json, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# rebuild the exact same architecture/classes as train.py so we can load weights cleanly
D_MODEL, N_HEAD, N_LAYER, D_FF, BLOCK = 64, 4, 4, 256, 64
HEAD_DIM = D_MODEL // N_HEAD
vocab = json.load(open('vocab.json'))
stoi, itos, chars = vocab['stoi'], vocab['itos'], vocab['chars']
vocab_size = len(chars)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.Wq = nn.Linear(D_MODEL, D_MODEL); self.Wk = nn.Linear(D_MODEL, D_MODEL)
        self.Wv = nn.Linear(D_MODEL, D_MODEL); self.Wo = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.fc1 = nn.Linear(D_MODEL, D_FF); self.fc2 = nn.Linear(D_FF, D_MODEL)
        self.register_buffer('mask', torch.tril(torch.ones(BLOCK, BLOCK)))
    def forward(self, x, capture=None):
        B,T,C = x.shape
        h = self.ln1(x)
        q = self.Wq(h).view(B,T,N_HEAD,HEAD_DIM).transpose(1,2)
        k = self.Wk(h).view(B,T,N_HEAD,HEAD_DIM).transpose(1,2)
        v = self.Wv(h).view(B,T,N_HEAD,HEAD_DIM).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(HEAD_DIM)
        att = att.masked_fill(self.mask[:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1,2).contiguous().view(B,T,C)
        out = self.Wo(out)
        if capture is not None:
            capture['Q'] = self.Wq(h); capture['K'] = self.Wk(h); capture['V'] = self.Wv(h)
            capture['attn'] = att  # [B, n_head, T, T]
        x = x + out
        h2 = self.ln2(x)
        h2 = self.fc2(F.gelu(self.fc1(h2)))
        return x + h2

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(BLOCK, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(D_MODEL)
    def forward(self, idx, capture_layer=0):
        B,T = idx.shape
        pos = torch.arange(T)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None,:,:]
        capture = {}
        for i, blk in enumerate(self.blocks):
            x = blk(x, capture if i==capture_layer else None)
        x = self.lnf(x)
        logits = x @ self.tok_emb.weight.T
        return logits, capture

model = TinyGPT()
ck = torch.load('ckpt.pt', map_location='cpu')
model.load_state_dict(ck['model'])
model.eval()

prompt = "ROMEO: "
ids = [stoi[c] for c in prompt]
idx = torch.tensor([ids], dtype=torch.long)
with torch.no_grad():
    logits, capture = model(idx, capture_layer=0)

result = {
    'prompt': prompt,
    'ids': ids,
    'last_logits': logits[0, -1, :].tolist(),
    'capture_layer0_Q': capture['Q'][0].tolist(),
    'capture_layer0_K': capture['K'][0].tolist(),
    'capture_layer0_V': capture['V'][0].tolist(),
    'capture_layer0_attn_head0': capture['attn'][0,0].tolist(),
    'capture_layer0_attn_head1': capture['attn'][0,1].tolist(),
}
json.dump(result, open('test_reference.json','w'))
print('saved test_reference.json')
print('top-5 next char probs:')
probs = F.softmax(logits[0,-1,:], dim=-1)
top = torch.topk(probs, 5)
for p, i in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {itos[str(i)]!r}: {p:.4f}")
