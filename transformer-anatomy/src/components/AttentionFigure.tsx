import { useEffect, useState } from 'react';
import { FigurePanel } from './FigurePanel';
import { Heatmap } from './Heatmap';
import type { ForwardResult, Token } from '../model/types';

interface AttentionFigureProps {
  tokens: Token[];
  result: ForwardResult;
  nLayer: number;
  nHead: number;
  layerIdx: number;
  onLayerChange: (l: number) => void;
}

function displayChar(ch: string): string {
  if (ch === '\n') return '\u23ce';
  if (ch === ' ') return '\u2423';
  return ch;
}

export function AttentionFigure({ tokens, result, nLayer, nHead, layerIdx, onLayerChange }: AttentionFigureProps) {
  const [headIdx, setHeadIdx] = useState(0);
  const [hoverRow, setHoverRow] = useState<number | null>(tokens.length - 1);

  useEffect(() => { setHoverRow(tokens.length - 1); }, [tokens.length, result]);

  const labels = tokens.map((t) => displayChar(t.text));
  const trace = result.layers[layerIdx];
  const attn = trace.heads[headIdx];
  const row = hoverRow ?? tokens.length - 1;
  const weights = attn[row];
  const ranked = weights
    .map((w, i) => [w, i] as [number, number])
    .filter(([, i]) => i <= row)
    .sort((a, b) => b[0] - a[0]);

  return (
    <FigurePanel
      tag="FIG. 02 — MULTI-HEAD SELF-ATTENTION"
      title="Tokens looking at tokens"
      description={
        <>Every token's vector becomes a Query, Key, and Value. Each token's Query is matched against every{' '}
        <em>earlier</em> token's Key (future tokens are masked off — causal attention) to get attention weights,
        which pull in a weighted blend of Values.</>
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
      footer="GPT-2 (small) stacks 12 of these blocks with 12 heads each. This model runs 4 blocks of 4 heads — small enough to read, big enough to be the real mechanism."
    >
      <div className="heatrow">
        <div className="heatcol"><span className="label">Query (Q)</span><Heatmap matrix={trace.Q} rowLabels={labels} mode="diverging" cellWidth={9} /></div>
        <div className="heatcol"><span className="label">Key (K)</span><Heatmap matrix={trace.K} rowLabels={labels} mode="diverging" cellWidth={9} /></div>
        <div className="heatcol"><span className="label">Value (V)</span><Heatmap matrix={trace.V} rowLabels={labels} mode="diverging" cellWidth={9} /></div>
      </div>

      <div className="tabs" style={{ marginTop: 22 }}>
        {Array.from({ length: nHead }, (_, h) => (
          <button key={h} className={`tab${h === headIdx ? ' active' : ''}`} onClick={() => setHeadIdx(h)}>
            Head {h + 1}
          </button>
        ))}
      </div>

      <div className="attn-grid">
        <div className="heatcol">
          <span className="label">Attention weights (hover a row)</span>
          <Heatmap
            matrix={attn}
            rowLabels={labels}
            colLabels={labels}
            mode="sequential"
            causalMask
            highlightRow={hoverRow}
            onHoverRow={setHoverRow}
            cellWidth={18}
            cellHeight={18}
          />
        </div>
        <div className="attn-side">
          <h4>"{displayChar(tokens[row]?.text ?? '')}" attends to —</h4>
          <div>
            {ranked.map(([w, i]) => (
              <div className="bar-row" key={i}>
                <span className="tok mono">{displayChar(tokens[i].text)}</span>
                <span className="track"><span className="fill" style={{ width: `${w * 100}%` }} /></span>
                <span className="pct">{(w * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="legend">
        <span className="swatch"><span className="bar seq" /> low → high attention weight</span>
        <span className="swatch"><span className="bar mask" /> masked (future token, never attended to)</span>
      </div>
    </FigurePanel>
  );
}
