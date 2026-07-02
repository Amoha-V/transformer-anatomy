import type { Token, VocabData } from './types';

/**
 * Char-level tokenizer — every character the model was trained on (including
 * punctuation, newline, and space) is its own token. This matches train.py
 * exactly: `chars = sorted(set(text))`.
 */
export class Tokenizer {
  private stoi: Record<string, number>;
  private itos: Record<string, string>;
  readonly vocabSize: number;

  constructor(vocab: VocabData) {
    this.stoi = vocab.stoi;
    this.itos = vocab.itos;
    this.vocabSize = vocab.chars.length;
  }

  /** Encode text into tokens, dropping any character the model never saw during training. */
  encode(text: string, maxLen?: number): Token[] {
    const tokens: Token[] = [];
    for (const ch of text) {
      if (ch in this.stoi) tokens.push({ id: this.stoi[ch], text: ch });
    }
    if (maxLen && tokens.length > maxLen) return tokens.slice(tokens.length - maxLen);
    return tokens;
  }

  idToChar(id: number): string {
    return this.itos[String(id)] ?? '\uFFFD';
  }

  allChars(): string[] {
    return Object.keys(this.stoi).sort((a, b) => this.stoi[a] - this.stoi[b]);
  }
}
