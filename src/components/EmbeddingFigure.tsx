import { FigurePanel } from './FigurePanel';
import { Heatmap } from './Heatmap';
import { TokenStrip } from './TokenStrip';
import type { ForwardResult, Token } from '../model/types';

interface EmbeddingFigureProps {
  tokens: Token[];
  result: ForwardResult;
}

export function EmbeddingFigure({ tokens, result }: EmbeddingFigureProps) {
  const labels = tokens.map((t) => (t.text === '\n' ? '\u23ce' : t.text === ' ' ? '\u2423' : t.text));
  return (
    <FigurePanel
      tag="FIG. 01 — TOKENIZE & EMBED"
      title="From characters to vectors"
      description={
        <>Each character is looked up in a 64-dimensional embedding table, then a separate position vector is
        added so the model knows <em>where</em> in the sequence it sits. The sum is what actually enters the model.</>
      }
      footer="This model uses 65 vocabulary entries (one per character it saw during training) and 64-dimensional vectors. GPT-2 (small) uses 50,257 byte-pair tokens and 768 dimensions — same idea, far bigger table."
    >
      <TokenStrip tokens={tokens} activeIndex={null} />
      <div className="heatrow">
        <div className="heatcol">
          <span className="label">Token embedding</span>
          <Heatmap matrix={result.tokenEmbeddings} rowLabels={labels} mode="diverging" cellWidth={9} />
        </div>
        <div className="op">+</div>
        <div className="heatcol">
          <span className="label">Positional encoding</span>
          <Heatmap matrix={result.positionEmbeddings} rowLabels={labels} mode="diverging" cellWidth={9} />
        </div>
        <div className="op">=</div>
        <div className="heatcol">
          <span className="label">Final embedding</span>
          <Heatmap matrix={result.inputEmbeddings} rowLabels={labels} mode="diverging" cellWidth={9} />
        </div>
      </div>
      <div className="legend">
        <span className="swatch"><span className="bar div" /> negative → zero → positive (vector value)</span>
      </div>
    </FigurePanel>
  );
}
