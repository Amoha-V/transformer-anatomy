import { FigurePanel } from './FigurePanel';
import { Heatmap } from './Heatmap';
import type { ForwardResult, Token } from '../model/types';

interface MlpFigureProps {
  tokens: Token[];
  result: ForwardResult;
  dModel: number;
  dFf: number;
  layerIdx: number;
  nLayer: number;
  onLayerChange: (l: number) => void;
}

function displayChar(ch: string): string {
  if (ch === '\n') return '\u23ce';
  if (ch === ' ') return '\u2423';
  return ch;
}

export function MlpFigure({ tokens, result, dModel, dFf, layerIdx, nLayer, onLayerChange }: MlpFigureProps) {
  const labels = tokens.map((t) => displayChar(t.text));
  const trace = result.layers[layerIdx];

  return (
    <FigurePanel
      tag="FIG. 03 — MLP"
      title="Refining each token on its own"
      description={
        <>Where attention moves information <em>between</em> tokens, the MLP refines each token's vector
        independently: expand to a wider space, apply a nonlinearity (GELU), then compress back down.</>
      }
      controls={
        <div className="tabs">
          {Array.from({ length: nLayer }, (_, l) => (
            <button key={l} className={`tab${l === layerIdx ? ' active' : ''}`} onClick={() => onLayerChange(l)}>
              Layer {l + 1} of {nLayer}
            </button>
          ))}
        </div>
      }
      footer={`GPT-2 (small) expands 768 → 3072 dims here. Same ×4 ratio — this model just runs it at ${dModel}→${dFf} scale.`}
    >
      <div className="mlp-flow">
        <span className="mlp-box">{dModel} dims</span><span className="arrow">→</span>
        <span className="mlp-box">{dFf} dims (×4 expand)</span><span className="arrow">→</span>
        <span className="mlp-box">GELU</span><span className="arrow">→</span>
        <span className="mlp-box">{dModel} dims (compress)</span>
      </div>
      <div className="heatrow">
        <div className="heatcol">
          <span className="label">Expanded activation ({dFf} dims)</span>
          <Heatmap matrix={trace.H1} rowLabels={labels} mode="diverging" cellWidth={4} />
        </div>
        <div className="op">→</div>
        <div className="heatcol">
          <span className="label">Block output ({dModel} dims)</span>
          <Heatmap matrix={trace.blockOutput} rowLabels={labels} mode="diverging" cellWidth={9} />
        </div>
      </div>
    </FigurePanel>
  );
}
