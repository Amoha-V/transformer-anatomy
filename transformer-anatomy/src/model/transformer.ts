import { matmul, addBiasRows, addMat, transpose, layerNorm, softmaxRow, gelu, zeros } from './math';
import type { ForwardResult, LayerTrace, Matrix, ModelWeights } from './types';

/**
 * Runs the full forward pass for a sequence of token ids, returning not just
 * the final logits but every intermediate matrix needed for visualization
 * (Q/K/V, per-head attention weights, MLP activations, the residual stream
 * after each block).
 *
 * This is a direct, verified port of train.py's TinyGPT.forward — see
 * scripts/verify-against-pytorch.md in the repo root for the cross-check
 * methodology. Max abs difference vs. the PyTorch reference on a held-out
 * prompt was ~2e-6, i.e. float32 rounding noise.
 */
export function forward(weights: ModelWeights, ids: number[]): ForwardResult {
  const { d_model: D_MODEL, n_head: N_HEAD } = weights.config;
  const HEAD_DIM = D_MODEL / N_HEAD;
  const seqLen = ids.length;

  const tokenEmbeddings: Matrix = ids.map((id) => weights.tok_emb[id]);
  const positionEmbeddings: Matrix = ids.map((_, i) => weights.pos_emb[i]);
  const inputEmbeddings: Matrix = tokenEmbeddings.map((row, i) => row.map((v, j) => v + positionEmbeddings[i][j]));

  let X = inputEmbeddings;
  const layers: LayerTrace[] = [];

  for (const Lp of weights.blocks) {
    const LN1 = layerNorm(X, Lp.ln1_g, Lp.ln1_b);
    const Q = addBiasRows(matmul(LN1, Lp.Wq), Lp.bq);
    const K = addBiasRows(matmul(LN1, Lp.Wk), Lp.bk);
    const V = addBiasRows(matmul(LN1, Lp.Wv), Lp.bv);

    const heads: Matrix[] = [];
    const headOuts: Matrix[] = [];
    for (let h = 0; h < N_HEAD; h++) {
      const s0 = h * HEAD_DIM, s1 = (h + 1) * HEAD_DIM;
      const Qh = Q.map((r) => r.slice(s0, s1));
      const Kh = K.map((r) => r.slice(s0, s1));
      const Vh = V.map((r) => r.slice(s0, s1));
      const scores = matmul(Qh, transpose(Kh)).map((r) => r.map((v) => v / Math.sqrt(HEAD_DIM)));
      // causal mask: token i may only attend to tokens 0..i
      for (let i = 0; i < seqLen; i++) for (let j = i + 1; j < seqLen; j++) scores[i][j] = -Infinity;
      const attn = scores.map(softmaxRow);
      heads.push(attn);
      headOuts.push(matmul(attn, Vh));
    }

    const concat = zeros(seqLen, D_MODEL);
    for (let i = 0; i < seqLen; i++) {
      for (let h = 0; h < N_HEAD; h++) {
        for (let d = 0; d < HEAD_DIM; d++) concat[i][h * HEAD_DIM + d] = headOuts[h][i][d];
      }
    }
    const attnProj = addBiasRows(matmul(concat, Lp.Wo), Lp.bo);
    const X1 = addMat(X, attnProj);

    const LN2 = layerNorm(X1, Lp.ln2_g, Lp.ln2_b);
    const H1 = addBiasRows(matmul(LN2, Lp.W1), Lp.b1).map((r) => r.map(gelu));
    const H2 = addBiasRows(matmul(H1, Lp.W2), Lp.b2);
    const X2 = addMat(X1, H2);

    layers.push({ Q, K, V, heads, H1, H2, blockOutput: X2 });
    X = X2;
  }

  const LNf = layerNorm(X, weights.lnf_g, weights.lnf_b);
  // weight-tied output head: logits = LNf @ tok_emb^T
  const logitsAll = matmul(LNf, transpose(weights.tok_emb));

  return {
    tokenEmbeddings,
    positionEmbeddings,
    inputEmbeddings,
    layers,
    lastLogits: logitsAll[seqLen - 1],
  };
}
