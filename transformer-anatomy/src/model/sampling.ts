import { softmaxRow } from './math';

export interface SamplingResult {
  probs: number[];           // full softmax distribution (post-temperature)
  includedIds: Set<number>;  // which ids survive the top-k/top-p filter
}

export function applyTemperature(logits: number[], temperature: number): number[] {
  return logits.map((v) => v / Math.max(temperature, 1e-6));
}

export function topKFilter(probs: number[], k: number): Set<number> {
  const ranked = probs.map((p, i): [number, number] => [p, i]).sort((a, b) => b[0] - a[0]);
  return new Set(ranked.slice(0, Math.max(1, Math.round(k))).map(([, i]) => i));
}

export function topPFilter(probs: number[], p: number): Set<number> {
  const ranked = probs.map((pr, i): [number, number] => [pr, i]).sort((a, b) => b[0] - a[0]);
  const included = new Set<number>();
  let cum = 0;
  for (const [pr, i] of ranked) {
    cum += pr;
    included.add(i);
    if (cum >= p) break;
  }
  return included;
}

export function computeDistribution(
  lastLogits: number[],
  temperature: number,
  mode: 'topk' | 'topp',
  filterValue: number
): SamplingResult {
  const probs = softmaxRow(applyTemperature(lastLogits, temperature));
  const includedIds = mode === 'topk' ? topKFilter(probs, filterValue) : topPFilter(probs, filterValue);
  return { probs, includedIds };
}

/** Sample one id from `probs`, restricted to and renormalized over `includedIds`. */
export function sample(probs: number[], includedIds: Set<number>): number {
  let total = 0;
  const pool: number[] = [];
  includedIds.forEach((i) => { total += probs[i]; pool.push(i); });
  let r = Math.random() * total;
  for (const i of pool) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return pool[pool.length - 1];
}
