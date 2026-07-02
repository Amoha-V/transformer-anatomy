import type { Matrix } from './types';

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}
function scale(a: number[], s: number): number[] {
  return a.map((v) => v * s);
}
function sub(a: number[], b: number[]): number[] {
  return a.map((v, i) => v - b[i]);
}
function matVec(M: Matrix, v: number[]): number[] {
  return M.map((row) => dot(row, v));
}

/**
 * Top-k eigenvectors of a symmetric matrix via power iteration + deflation.
 * Good enough for the d_model x d_model covariance matrices we use here
 * (d_model is at most a few hundred), no need for a full SVD library.
 */
function topEigenvectors(symmetric: Matrix, k: number, iters = 200): { vectors: number[][]; values: number[] } {
  const n = symmetric.length;
  let M = symmetric.map((row) => row.slice());
  const vectors: number[][] = [];
  const values: number[] = [];

  for (let c = 0; c < k; c++) {
    let v = Array.from({ length: n }, () => Math.random() - 0.5);
    let vn = norm(v);
    v = scale(v, 1 / vn);
    let eigenvalue = 0;
    for (let it = 0; it < iters; it++) {
      const Mv = matVec(M, v);
      const newNorm = norm(Mv);
      if (newNorm < 1e-12) break;
      v = scale(Mv, 1 / newNorm);
      eigenvalue = newNorm;
    }
    // sign/magnitude refine: eigenvalue = v^T M v
    eigenvalue = dot(v, matVec(M, v));
    vectors.push(v);
    values.push(eigenvalue);
    // deflate: M -= eigenvalue * v v^T
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        M[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }
  return { vectors, values };
}

export interface PcaResult {
  coords: [number, number][]; // one per input row
  explainedVariance: [number, number]; // top-2 eigenvalues (proportional to variance along each axis)
}

/** Project rows of `X` (e.g. the token embedding table) onto their top-2 principal components. */
export function pca2D(X: Matrix): PcaResult {
  const n = X.length, d = X[0].length;
  const mean = new Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j] / n;
  const centered = X.map((row) => sub(row, mean));

  // covariance = centered^T @ centered / (n - 1)   -- a d x d symmetric matrix
  const cov: Matrix = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const row of centered) {
    for (let i = 0; i < d; i++) {
      const ri = row[i];
      if (ri === 0) continue;
      for (let j = 0; j < d; j++) cov[i][j] += ri * row[j];
    }
  }
  for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) cov[i][j] /= n - 1;

  const { vectors, values } = topEigenvectors(cov, 2);
  const coords: [number, number][] = centered.map((row) => [dot(row, vectors[0]), dot(row, vectors[1])]);
  return { coords, explainedVariance: [values[0], values[1]] };
}
