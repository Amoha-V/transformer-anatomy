import type { Matrix } from './types';

export function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

/** Standard matrix multiply: A [r,n] x B [n,c] -> [r,c]. */
export function matmul(A: Matrix, B: Matrix): Matrix {
  const r = A.length, n = B.length, c = B[0].length;
  const out = zeros(r, c);
  for (let i = 0; i < r; i++) {
    const Ai = A[i];
    for (let k = 0; k < n; k++) {
      const a = Ai[k];
      if (a === 0) continue;
      const Bk = B[k];
      const Oi = out[i];
      for (let j = 0; j < c; j++) Oi[j] += a * Bk[j];
    }
  }
  return out;
}

export function addBiasRows(A: Matrix, bias: number[]): Matrix {
  return A.map((row) => row.map((v, j) => v + bias[j]));
}

export function addMat(A: Matrix, B: Matrix): Matrix {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

export function transpose(A: Matrix): Matrix {
  const r = A.length, c = A[0].length;
  const T = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
  return T;
}

/** Row-wise layer normalization: zero mean, unit variance, then scale/shift. */
export function layerNorm(X: Matrix, gamma: number[], beta: number[], eps = 1e-5): Matrix {
  return X.map((row) => {
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) * (b - mean), 0) / row.length;
    const denom = Math.sqrt(variance + eps);
    return row.map((v, j) => ((v - mean) / denom) * gamma[j] + beta[j]);
  });
}

/** Numerically-stable softmax over a single row. -Infinity entries (masked) contribute zero. */
export function softmaxRow(row: number[]): number[] {
  const finite = row.filter((v) => v > -Infinity);
  const max = Math.max(...finite);
  const exps = row.map((v) => (v === -Infinity ? 0 : Math.exp(v - max)));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/**
 * Error function approximation (Abramowitz & Stegun 7.1.26), accurate to ~1.5e-7.
 * PyTorch's default `F.gelu` uses the *exact* erf form (not the tanh approximation),
 * so this implementation matches that exactly rather than the commonly-seen tanh approx.
 */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

export function gelu(x: number): number {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}
