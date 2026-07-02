export type Matrix = number[][];

export interface BlockWeights {
  ln1_g: number[]; ln1_b: number[];
  Wq: Matrix; bq: number[];
  Wk: Matrix; bk: number[];
  Wv: Matrix; bv: number[];
  Wo: Matrix; bo: number[];
  ln2_g: number[]; ln2_b: number[];
  W1: Matrix; b1: number[];
  W2: Matrix; b2: number[];
}

export interface ModelConfig {
  d_model: number;
  n_head: number;
  n_layer: number;
  d_ff: number;
  block_size: number;
  vocab_size: number;
}

export interface ModelWeights {
  config: ModelConfig;
  tok_emb: Matrix;   // [vocab_size, d_model]
  pos_emb: Matrix;   // [block_size, d_model]
  lnf_g: number[];
  lnf_b: number[];
  blocks: BlockWeights[];
}

export interface VocabData {
  stoi: Record<string, number>;
  itos: Record<string, string>;
  chars: string[];
}

/** Everything computed for one transformer block, kept around for visualization. */
export interface LayerTrace {
  Q: Matrix; K: Matrix; V: Matrix;   // [seqLen, d_model], pre-head-split
  heads: Matrix[];                    // n_head matrices, each [seqLen, seqLen] (post-softmax attention weights)
  H1: Matrix;                         // [seqLen, d_ff]  (post-GELU expanded activation)
  H2: Matrix;                         // [seqLen, d_model] (this block's output delta before the residual add)
  blockOutput: Matrix;                // [seqLen, d_model] (X after this block, i.e. the residual stream)
}

export interface ForwardResult {
  tokenEmbeddings: Matrix;   // [seqLen, d_model] raw token embedding lookup, no position
  positionEmbeddings: Matrix; // [seqLen, d_model]
  inputEmbeddings: Matrix;   // [seqLen, d_model] tokenEmbeddings + positionEmbeddings
  layers: LayerTrace[];
  lastLogits: number[];     // [vocab_size] logits for the position right after the last input token
}

export interface Token {
  id: number;
  text: string;
}
