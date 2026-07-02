import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from 'recharts';
import { FigurePanel } from './FigurePanel';
import { computeDistribution } from '../model/sampling';
import { Tokenizer } from '../model/tokenizer';

interface OutputFigureProps {
  lastLogits: number[];
  tokenizer: Tokenizer;
  temperature: number;
  onTemperatureChange: (v: number) => void;
  mode: 'topk' | 'topp';
  onModeChange: (m: 'topk' | 'topp') => void;
  filterValue: number;
  onFilterValueChange: (v: number) => void;
  onGenerate: () => void;
  onReset: () => void;
}

function displayChar(ch: string): string {
  if (ch === '\n') return '\u23ce';
  if (ch === ' ') return '\u2423';
  return ch;
}

export function OutputFigure({
  lastLogits, tokenizer, temperature, onTemperatureChange, mode, onModeChange,
  filterValue, onFilterValueChange, onGenerate, onReset,
}: OutputFigureProps) {
  const { probs, includedIds } = computeDistribution(lastLogits, temperature, mode, filterValue);
  const ranked = probs
    .map((p, i) => ({ id: i, p, char: displayChar(tokenizer.idToChar(i)), included: includedIds.has(i) }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 12);

  return (
    <FigurePanel
      tag="FIG. 04 — OUTPUT PROBABILITIES"
      title="Picking the next character"
      description={
        <>The final vector is projected back across the whole vocabulary and turned into a probability for every
        character. Temperature reshapes the distribution; top-k and top-p decide which characters are even
        allowed to be sampled.</>
      }
      footer="Generate appends one real sampled character and re-runs the entire forward pass — every figure above updates to match. Click it a few times to watch generation happen, one true inference step at a time."
    >
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer>
          <BarChart data={ranked} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" vertical={false} />
            <XAxis dataKey="char" tick={{ fontFamily: 'JetBrains Mono', fontSize: 12, fill: 'var(--ink)' }} />
            <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--ink-soft)' }} width={44} />
            <Tooltip
              formatter={(v: number, _n, item) => [`${(v * 100).toFixed(2)}%`, item.payload.included ? 'included' : 'excluded by filter']}
              labelFormatter={(l) => `character: "${l}"`}
              contentStyle={{ fontFamily: 'JetBrains Mono', fontSize: 12, border: '1.5px solid var(--ink)', borderRadius: 2 }}
            />
            <Bar dataKey="p" radius={[2, 2, 0, 0]}>
              {ranked.map((d) => (
                <Cell key={d.id} fill={d.included ? 'var(--amber)' : 'transparent'} stroke={d.included ? 'var(--amber-deep)' : 'var(--line-strong)'} strokeDasharray={d.included ? undefined : '3 3'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="sliders">
        <div className="slider-group">
          <div className="row"><span>Temperature</span><b>{temperature.toFixed(2)}</b></div>
          <input type="range" min={0.1} max={2} step={0.05} value={temperature} onChange={(e) => onTemperatureChange(parseFloat(e.target.value))} />
        </div>
        <div className="slider-group">
          <div className="mode-toggle">
            <button className={`tab${mode === 'topk' ? ' active' : ''}`} onClick={() => onModeChange('topk')} type="button">Top-k</button>
            <button className={`tab${mode === 'topp' ? ' active' : ''}`} onClick={() => onModeChange('topp')} type="button">Top-p</button>
          </div>
          <div className="row"><span>{mode === 'topk' ? 'k' : 'p'}</span><b>{mode === 'topk' ? filterValue.toFixed(0) : filterValue.toFixed(2)}</b></div>
          <input
            type="range"
            min={mode === 'topk' ? 1 : 0.1}
            max={mode === 'topk' ? 20 : 1}
            step={mode === 'topk' ? 1 : 0.05}
            value={filterValue}
            onChange={(e) => onFilterValueChange(parseFloat(e.target.value))}
          />
        </div>
      </div>

      <div className="controls" style={{ marginTop: 22 }}>
        <button onClick={onGenerate}>Generate next character →</button>
        <button className="ghost" onClick={onReset} type="button">Reset prompt</button>
      </div>
    </FigurePanel>
  );
}
