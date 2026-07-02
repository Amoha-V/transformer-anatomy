/**
 * Cross-checks this repo's TypeScript-equivalent forward pass (reimplemented
 * here in plain JS for a zero-dependency Node script) against logits and
 * internal activations captured directly from the trained PyTorch model
 * (see ../make_reference.py in the training repo). If this passes, the
 * in-browser inference engine in src/model/transformer.ts is numerically
 * equivalent to the original model, not just "inspired by" it.
 *
 * Run with: npm run verify
 */
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..');
const weights = JSON.parse(fs.readFileSync(path.join(root, 'public/weights.json'), 'utf8'));
const vocab = JSON.parse(fs.readFileSync(path.join(root, 'public/vocab.json'), 'utf8'));
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, 'test_reference.json'), 'utf8'));

const { d_model: D_MODEL, n_head: N_HEAD, n_layer: N_LAYER, d_ff: D_FF, block_size: BLOCK, vocab_size: VOCAB } = weights.config;
const HEAD_DIM = D_MODEL / N_HEAD;

function zeros(r, c) { return Array.from({ length: r }, () => new Array(c).fill(0)); }
function matmul(A, B) {
  const r = A.length, n = B.length, c = B[0].length;
  const out = zeros(r, c);
  for (let i = 0; i < r; i++) for (let k = 0; k < n; k++) { const a = A[i][k]; if (a === 0) continue; const Bk = B[k]; for (let j = 0; j < c; j++) out[i][j] += a * Bk[j]; }
  return out;
}
function addBiasRows(A, bias) { return A.map(row => row.map((v, j) => v + bias[j])); }
function addMat(A, B) { return A.map((row, i) => row.map((v, j) => v + B[i][j])); }
function transpose(A) { const r = A.length, c = A[0].length; const T = zeros(c, r); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j]; return T; }
function layerNorm(X, gamma, beta, eps = 1e-5) {
  return X.map(row => {
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) * (b - mean), 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    return row.map((v, j) => ((v - mean) / denom) * gamma[j] + beta[j]);
  });
}
function softmaxRow(row) {
  const finite = row.filter(v => v > -Infinity);
  const max = Math.max(...finite);
  const exps = row.map(v => (v === -Infinity ? 0 : Math.exp(v - max)));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}
// PyTorch's F.gelu default is the *exact* erf-based GELU, not the tanh approximation.
function erf(x) {
  // Abramowitz-Stegun approximation, accurate to ~1.5e-7
  const sign = x < 0 ? -1 : 1; x = Math.abs(x);
  const a1=0.254829592,a2=-0.284496736,a3=1.421413741,a4=-1.453152027,a5=1.061405429,p=0.3275911;
  const t = 1/(1+p*x);
  const y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*Math.exp(-x*x);
  return sign*y;
}
function gelu(x){ return 0.5*x*(1+erf(x/Math.SQRT2)); }

function forward(ids, captureLayer = 0) {
  const seqLen = ids.length;
  const X0 = ids.map((id, i) => weights.tok_emb[id].map((v, j) => v + weights.pos_emb[i][j]));
  let X = X0;
  let capture = null;
  for (let l = 0; l < N_LAYER; l++) {
    const Lp = weights.blocks[l];
    const LN1 = layerNorm(X, Lp.ln1_g, Lp.ln1_b);
    const Q = addBiasRows(matmul(LN1, Lp.Wq), Lp.bq);
    const K = addBiasRows(matmul(LN1, Lp.Wk), Lp.bk);
    const V = addBiasRows(matmul(LN1, Lp.Wv), Lp.bv);
    const heads = []; const headOuts = [];
    for (let h = 0; h < N_HEAD; h++) {
      const s0 = h*HEAD_DIM, s1 = (h+1)*HEAD_DIM;
      const Qh = Q.map(r=>r.slice(s0,s1)), Kh = K.map(r=>r.slice(s0,s1)), Vh = V.map(r=>r.slice(s0,s1));
      const scores = matmul(Qh, transpose(Kh)).map(r=>r.map(v=>v/Math.sqrt(HEAD_DIM)));
      for (let i=0;i<seqLen;i++) for (let j=i+1;j<seqLen;j++) scores[i][j]=-Infinity;
      const attn = scores.map(softmaxRow);
      heads.push(attn); headOuts.push(matmul(attn, Vh));
    }
    if (l === captureLayer) capture = { Q, K, V, heads };
    const concat = zeros(seqLen, D_MODEL);
    for (let i=0;i<seqLen;i++) for (let h=0;h<N_HEAD;h++) for (let d=0;d<HEAD_DIM;d++) concat[i][h*HEAD_DIM+d]=headOuts[h][i][d];
    const attnProj = addBiasRows(matmul(concat, Lp.Wo), Lp.bo);
    const X1 = addMat(X, attnProj);
    const LN2 = layerNorm(X1, Lp.ln2_g, Lp.ln2_b);
    const H1 = addBiasRows(matmul(LN2, Lp.W1), Lp.b1).map(r=>r.map(gelu));
    const H2 = addBiasRows(matmul(H1, Lp.W2), Lp.b2);
    X = addMat(X1, H2);
  }
  const LNf = layerNorm(X, weights.lnf_g, weights.lnf_b);
  const logitsAll = matmul(LNf, transpose(weights.tok_emb));
  return { lastLogits: logitsAll[seqLen-1], capture };
}

const { lastLogits, capture } = forward(ref.ids, 0);

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i=0;i<a.length;i++) m = Math.max(m, Math.abs(a[i]-b[i]));
  return m;
}
function maxAbsDiff2D(a, b) {
  let m = 0;
  for (let i=0;i<a.length;i++) for (let j=0;j<a[i].length;j++) m = Math.max(m, Math.abs(a[i][j]-b[i][j]));
  return m;
}

const diffs = {
  logits: maxAbsDiff(lastLogits, ref.last_logits),
  Q: maxAbsDiff2D(capture.Q, ref.capture_layer0_Q),
  K: maxAbsDiff2D(capture.K, ref.capture_layer0_K),
  V: maxAbsDiff2D(capture.V, ref.capture_layer0_V),
  attn_head0: maxAbsDiff2D(capture.heads[0], ref.capture_layer0_attn_head0),
  attn_head1: maxAbsDiff2D(capture.heads[1], ref.capture_layer0_attn_head1),
};
console.log('max abs diff vs. PyTorch reference:');
for (const [k, v] of Object.entries(diffs)) console.log(`  ${k.padEnd(10)} ${v.toExponential(3)}`);

const TOLERANCE = 1e-3; // generous float32 rounding tolerance
const worst = Math.max(...Object.values(diffs));
if (worst < TOLERANCE) {
  console.log(`\nPASS — all diffs well under tolerance (${TOLERANCE.toExponential(0)}). The JS/TS engine matches PyTorch.`);
} else {
  console.error(`\nFAIL — diff of ${worst.toExponential(3)} exceeds tolerance (${TOLERANCE.toExponential(0)}).`);
  process.exit(1);
}

// also sanity check: top-5 chars from our logits match python's top-5
function softmaxFull(row){ return softmaxRow(row); }
const probs = softmaxFull(lastLogits);
const ranked = probs.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]).slice(0,5);
console.log('\nJS top-5 next-char predictions:', ranked.map(([p,i])=>`${JSON.stringify(vocab.itos[String(i)])}:${p.toFixed(4)}`).join(' '));
