import { useEffect, useState, useCallback, useMemo } from 'react';
import { Tokenizer } from './model/tokenizer';
import { forward } from './model/transformer';
import { sample, computeDistribution } from './model/sampling';
import type { ForwardResult, ModelWeights, Token, VocabData } from './model/types';
import { PromptBar } from './components/PromptBar';
import { EmbeddingFigure } from './components/EmbeddingFigure';
import { AttentionFigure } from './components/AttentionFigure';
import { MlpFigure } from './components/MlpFigure';
import { OutputFigure } from './components/OutputFigure';
import { LossCurveFigure } from './components/LossCurveFigure';
import { EmbeddingSpaceFigure } from './components/EmbeddingSpaceFigure';

const DEFAULT_PROMPT = 'ROMEO:\nBut soft, what light';

interface TrainingLog {
  log: { step: number; train_loss: number; val_loss: number }[];
  n_params: number;
}

export default function App() {
  const [weights, setWeights] = useState<ModelWeights | null>(null);
  const [vocab, setVocab] = useState<VocabData | null>(null);
  const [trainingLog, setTrainingLog] = useState<TrainingLog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [tokens, setTokens] = useState<Token[]>([]);
  const [result, setResult] = useState<ForwardResult | null>(null);
  const [layerIdx, setLayerIdx] = useState(0);

  const [temperature, setTemperature] = useState(0.8);
  const [mode, setMode] = useState<'topk' | 'topp'>('topk');
  const [filterValue, setFilterValue] = useState(8);

  useEffect(() => {
    Promise.all([
      fetch('./weights.json').then((r) => r.json()),
      fetch('./vocab.json').then((r) => r.json()),
      fetch('./training_log.json').then((r) => r.json()),
    ])
      .then(([w, v, t]) => { setWeights(w); setVocab(v); setTrainingLog(t); })
      .catch((e) => setError(String(e)));
  }, []);

  const tokenizer = useMemo(() => (vocab ? new Tokenizer(vocab) : null), [vocab]);

  const run = useCallback((text: string) => {
    if (!weights || !tokenizer) return;
    const toks = tokenizer.encode(text, weights.config.block_size);
    if (toks.length === 0) return;
    setTokens(toks);
    setResult(forward(weights, toks.map((t) => t.id)));
  }, [weights, tokenizer]);

  useEffect(() => { if (weights && tokenizer) run(prompt); }, [weights, tokenizer]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleGenerate = () => {
    if (!result || !tokenizer || !weights) return;
    const { probs, includedIds } = computeDistribution(result.lastLogits, temperature, mode, filterValue);
    const nextId = sample(probs, includedIds);
    const nextChar = tokenizer.idToChar(nextId);
    const nextPrompt = prompt + nextChar;
    setPrompt(nextPrompt);
    run(nextPrompt);
  };

  const handleReset = () => { setPrompt(DEFAULT_PROMPT); run(DEFAULT_PROMPT); };

  if (error) {
    return <div className="wrap"><div className="hero"><h1>Failed to load model</h1><p className="lede mono">{error}</p></div></div>;
  }
  if (!weights || !vocab || !trainingLog || !tokenizer || !result) {
    return <div className="wrap"><div className="hero"><h1>Loading model…</h1><p className="lede">Fetching trained weights (~4.3MB) and running the first forward pass.</p></div></div>;
  }

  const { d_model, n_head, n_layer, d_ff } = weights.config;

  return (
    <div className="wrap">
      <div className="hero">
        <div className="eyebrow">Real model · real training · verified inference</div>
        <h1>A transformer, actually trained, inspected live.</h1>
        <p className="lede">
          This is a {d_model}-dimensional, {n_layer}-layer, {n_head}-head GPT-style transformer, trained from
          scratch on tiny-shakespeare — then reimplemented in TypeScript and checked against the original PyTorch
          model until the two agreed to within float32 rounding error. Every number on this page is a live
          computation in your browser, not an animation.
        </p>
        <div className="stat-row">
          <div className="stat"><span className="v">{trainingLog.n_params.toLocaleString()}</span><span className="l">parameters</span></div>
          <div className="stat"><span className="v">{trainingLog.log[trainingLog.log.length - 1].train_loss.toFixed(2)}</span><span className="l">final train loss</span></div>
          <div className="stat"><span className="v">~2e-6</span><span className="l">max diff vs. pytorch</span></div>
          <div className="stat"><span className="v">{vocab.chars.length}</span><span className="l">vocabulary size</span></div>
        </div>
        <PromptBar value={prompt} onChange={(v) => { setPrompt(v); run(v); }} onRun={() => run(prompt)} maxLen={weights.config.block_size} />
      </div>

      <EmbeddingFigure tokens={tokens} result={result} />
      <AttentionFigure tokens={tokens} result={result} nLayer={n_layer} nHead={n_head} layerIdx={layerIdx} onLayerChange={setLayerIdx} />
      <MlpFigure tokens={tokens} result={result} dModel={d_model} dFf={d_ff} layerIdx={layerIdx} nLayer={n_layer} onLayerChange={setLayerIdx} />
      <OutputFigure
        lastLogits={result.lastLogits}
        tokenizer={tokenizer}
        temperature={temperature}
        onTemperatureChange={setTemperature}
        mode={mode}
        onModeChange={(m) => { setMode(m); setFilterValue(m === 'topk' ? 8 : 0.9); }}
        filterValue={filterValue}
        onFilterValueChange={setFilterValue}
        onGenerate={handleGenerate}
        onReset={handleReset}
      />
      <LossCurveFigure log={trainingLog.log} nParams={trainingLog.n_params} />
      <EmbeddingSpaceFigure tokEmb={weights.tok_emb} chars={vocab.chars} />

      <footer>
        Built end-to-end: a {trainingLog.n_params.toLocaleString()}-parameter transformer trained in PyTorch on{' '}
        <a href="https://www.gutenberg.org/" target="_blank" rel="noopener">public-domain</a> Shakespeare text,
        exported as raw weights, and reimplemented from scratch in TypeScript — tokenizer, attention, MLP, and
        sampling — verified numerically against the original model. Visualizations built with React, D3, and
        Recharts. Source on{' '}
        <a href="https://github.com/" target="_blank" rel="noopener">GitHub</a>. Loosely inspired by{' '}
        <a href="https://poloclub.github.io/transformer-explainer/" target="_blank" rel="noopener">Transformer Explainer</a>{' '}
        (Cho, Kim, Karpekov, Helbling, Wang, Lee, Hoover &amp; Chau — Georgia Tech).
      </footer>
    </div>
  );
}
