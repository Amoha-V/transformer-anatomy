import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FigurePanel } from './FigurePanel';
import { pca2D } from '../model/pca';
import type { Matrix } from '../model/types';

interface EmbeddingSpaceFigureProps {
  tokEmb: Matrix;
  chars: string[];
}

type Category = 'vowel' | 'consonant' | 'digit' | 'punctuation' | 'whitespace';

function categorize(ch: string): Category {
  if (ch === ' ' || ch === '\n' || ch === '\t') return 'whitespace';
  if (/[aeiouAEIOU]/.test(ch)) return 'vowel';
  if (/[a-zA-Z]/.test(ch)) return 'consonant';
  if (/[0-9]/.test(ch)) return 'digit';
  return 'punctuation';
}

function displayChar(ch: string): string {
  if (ch === '\n') return '\u23ce';
  if (ch === ' ') return '\u2423';
  if (ch === '\t') return '\u2192';
  return ch;
}

const CATEGORY_COLOR: Record<Category, string> = {
  vowel: 'var(--amber-deep)',
  consonant: 'var(--amber)',
  digit: 'var(--rust)',
  punctuation: 'var(--teal)',
  whitespace: 'var(--ink-soft)',
};

export function EmbeddingSpaceFigure({ tokEmb, chars }: EmbeddingSpaceFigureProps) {
  const { coords, explainedVariance } = useMemo(() => pca2D(tokEmb), [tokEmb]);

  const byCategory = useMemo(() => {
    const groups: Record<Category, { x: number; y: number; char: string }[]> = {
      vowel: [], consonant: [], digit: [], punctuation: [], whitespace: [],
    };
    chars.forEach((ch, i) => {
      const cat = categorize(ch);
      groups[cat].push({ x: coords[i][0], y: coords[i][1], char: displayChar(ch) });
    });
    return groups;
  }, [chars, coords]);

  const totalVar = explainedVariance[0] + explainedVariance[1];

  return (
    <FigurePanel
      tag="FIG. 06 — EMBEDDING SPACE"
      title="What the model learned about characters"
      description={
        <>Every character has a learned 64-dimensional vector. To see it, we run a from-scratch 2D PCA (power
        iteration with deflation, implemented in <code>src/model/pca.ts</code> — no library) on the real, trained
        embedding table and plot the top two components.</>
      }
      footer={`These two components explain a combined ${(totalVar > 0 ? (explainedVariance[0] / totalVar) * 100 : 0).toFixed(0)}/${(totalVar > 0 ? (explainedVariance[1] / totalVar) * 100 : 0).toFixed(0)} split of the variance captured. With only ~8M training characters and a 4-block model, don't expect textbook-clean vowel/consonant clusters — but related characters (e.g. punctuation, or letters that occur in similar contexts) often do drift closer together than chance would predict.`}
    >
      <div style={{ width: '100%', height: 360 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 12, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
            <XAxis type="number" dataKey="x" name="PC1" tick={{ fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--ink-soft)' }} />
            <YAxis type="number" dataKey="y" name="PC2" tick={{ fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--ink-soft)' }} />
            <ZAxis range={[80, 80]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(_v, _n, item: any) => [item.payload.char, 'character']}
              contentStyle={{ fontFamily: 'JetBrains Mono', fontSize: 12, border: '1.5px solid var(--ink)', borderRadius: 2 }}
            />
            <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 12 }} />
            {(Object.keys(byCategory) as Category[]).map((cat) => (
              <Scatter key={cat} name={cat} data={byCategory[cat]} fill={CATEGORY_COLOR[cat]} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </FigurePanel>
  );
}
