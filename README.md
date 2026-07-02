# Transformer Anatomy

A small GPT-style transformer, **trained from scratch** on Shakespeare, **reimplemented from scratch in TypeScript**, and verified numerically against the original PyTorch model — wrapped in an interactive React app that visualizes every internal stage of inference live, in the browser.

This is not a simulation and not an animation. The weights are real, the training run actually happened, and every heatmap, attention weight, and probability bar you see is computed in your browser at the moment you interact with the page.

## What's actually in here

| Layer | What it is |
|---|---|
| **Model** | 4-layer, 4-head, 64-dim GPT-style decoder, 208,320 parameters, char-level, trained with PyTorch/AdamW on [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) |
| **Inference engine** | A from-scratch TypeScript reimplementation of the forward pass (`src/model/`) — tokenizer, embeddings, causal multi-head attention, MLP, weight-tied output head, temperature/top-k/top-p sampling |
| **Correctness** | `scripts/verify-against-pytorch.cjs` checks the JS forward pass against logits and internal activations captured directly from the trained PyTorch model. Max abs diff: **~2×10⁻⁶** (float32 rounding noise) |
| **Visualization** | React + TypeScript app using **D3** (heatmaps: embeddings, Q/K/V, attention weights, MLP activations) and **Recharts** (training loss curve, output probability distribution, PCA-projected embedding space) |
| **Dimensionality reduction** | 2D PCA via power iteration with deflation, implemented from scratch (`src/model/pca.ts`) — no `numpy`, no linear-algebra library |
| **Deploy** | GitHub Actions workflow builds and publishes to GitHub Pages on every push to `main` |

## Why build it this way

The original inspiration, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Cho et al., Georgia Tech), runs a real GPT-2 via ONNX Runtime Web. This project takes a different, smaller-scope approach that's easier to fully own and explain end to end: train a tiny model myself, port its forward pass to TypeScript by hand, and *prove* the port is correct with a numerical cross-check rather than assuming it. That cross-check (`npm run verify`) is the part of this repo I'd point to first in an interview — it's the difference between "I built a UI that looks like a transformer" and "I built and verified an inference engine."

## Architecture

```
Input text
   │  char-level tokenize (65-char vocab: letters, digits, punctuation, space, newline)
   ▼
Token embedding (65 × 64) + Position embedding (64 × 64)
   ▼
┌─────────────────────────────────────────┐
│  Block × 4                               │
│  ┌─────────────────────────────────┐    │
│  │ LayerNorm → Q,K,V (64×64 each)  │    │
│  │ → split into 4 heads (16 dims)  │    │
│  │ → causal scaled dot-product     │    │
│  │   attention → concat → project  │    │
│  └──────────────┬──────────────────┘    │
│                 │ + residual             │
│  ┌──────────────▼──────────────────┐    │
│  │ LayerNorm → Linear(64→256)      │    │
│  │ → GELU → Linear(256→64)         │    │
│  └──────────────┬──────────────────┘    │
│                 │ + residual             │
└─────────────────┼─────────────────────────┘
                   ▼
         Final LayerNorm
                   ▼
   Linear, weight-tied to token embedding (64 → 65)
                   ▼
              Logits → softmax → sample
```

Every box in that diagram has a matching figure in the app, built from the actual matrices flowing through the model on whatever prompt you type.

## Running it locally

```bash
npm install
npm run dev       # http://localhost:5173
```

```bash
npm run build     # production build to dist/
npm run preview   # serve the production build locally
```

```bash
npm run verify    # cross-check the TS inference engine against the PyTorch reference
```

## Deploying

The included workflow (`.github/workflows/deploy.yml`) builds and deploys to **GitHub Pages** automatically:

1. Push this repo to GitHub.
2. In **Settings → Pages**, set the source to **GitHub Actions**.
3. Push to `main` — the site builds and deploys automatically.

To deploy on **Vercel** instead: import the repo, framework preset "Vite", no other config needed.

## Retraining or changing the model

Training lives outside the app, in `training/`, since it only needs to run once and the output (`public/weights.json`, `public/vocab.json`, `public/training_log.json`) is what the app actually loads.

```bash
cd training
pip install -r requirements.txt
python train.py
```

`train.py` checkpoints itself (`ckpt.pt`) every run, so it's safe to interrupt and resume — useful on CPU-only or time-limited machines. Current hyperparameters (`D_MODEL=64, N_HEAD=4, N_LAYER=4, D_FF=256, BLOCK=64`) train to a final loss of **1.58 (train) / 1.77 (val)** in ~3000 steps, a few minutes on a single CPU core.

To change the architecture: edit the constants at the top of `train.py`, retrain, then copy the three exported JSON files into `public/`. **`src/model/transformer.ts` reads `config` out of `weights.json` directly**, so most architecture changes (dims, heads, layers, FF width) need no code changes on the frontend — only block size and number of heads need to divide evenly, same constraint PyTorch enforces.

Sample output after training (temperature 0.8, top-k 10, unconditioned):

```
Which you have sent the faret the to all arest;
And sorrow you shalt make the brother the lawly:--

GLOUCESTER:
I shall not! Away hear, his that hath! I was the fate.
```

It's not coherent English — 208K parameters and a few minutes of CPU training won't get you there — but it has learned real structure: character names in caps followed by a colon, line breaks between speakers, plausible English letter sequences. That's the honest, visible signature of real (if undertrained) learning, which is the whole point of the project.

## Project structure

```
├── public/
│   ├── weights.json          trained model weights (~4.3MB)
│   ├── vocab.json            char ↔ id mappings
│   └── training_log.json     loss curve + param count
├── src/
│   ├── model/
│   │   ├── types.ts          shared types
│   │   ├── math.ts           matmul, layernorm, softmax, gelu (erf-exact, matches PyTorch)
│   │   ├── tokenizer.ts      char-level tokenizer
│   │   ├── transformer.ts    the verified forward pass
│   │   ├── sampling.ts       temperature / top-k / top-p
│   │   └── pca.ts            from-scratch 2D PCA (power iteration + deflation)
│   ├── components/           one component per figure, plus a shared D3 Heatmap
│   ├── App.tsx
│   └── main.tsx
├── scripts/
│   ├── verify-against-pytorch.cjs
│   └── test_reference.json   captured PyTorch activations for the cross-check
├── training/
│   ├── train.py               PyTorch training script (resumable)
│   ├── make_reference.py      generates scripts/test_reference.json
│   └── requirements.txt
└── .github/workflows/deploy.yml
```

## Known limitations / honest caveats

- **The model is small and undertrained on purpose.** This is a portfolio piece about the *mechanism*, not a state-of-the-art language model. Don't expect coherent generations.
- **Bundle size.** The production JS bundle is ~590KB minified (mostly D3 + Recharts). Code-splitting with dynamic `import()` per figure would be the next optimization if this were a production app rather than a demo.
- **64-character context window.** Long prompts are truncated to the most recent 64 characters before inference, matching the block size used during training.

## Credits

Loosely inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Aeree Cho, Grace C. Kim, Alexander Karpekov, Alec Helbling, Zijie J. Wang, Seongmin Lee, Benjamin Hoover, Duen Horng Chau — Georgia Tech), which runs a real, trained GPT-2 in the browser via ONNX Runtime Web. Trained on [tiny-shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), the same public-domain dataset Andrej Karpathy's char-rnn and nanoGPT tutorials use.
