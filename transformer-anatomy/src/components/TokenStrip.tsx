import type { Token } from '../model/types';

interface TokenStripProps {
  tokens: Token[];
  activeIndex: number | null;
}

function displayChar(ch: string): string {
  if (ch === '\n') return '\u23ce'; // ⏎
  if (ch === ' ') return '\u2423';  // ␣
  return ch;
}

export function TokenStrip({ tokens, activeIndex }: TokenStripProps) {
  return (
    <div className="chips">
      {tokens.map((t, i) => (
        <div key={i} className={`chip${i === activeIndex ? ' active' : ''}`}>
          <span>{displayChar(t.text)}</span>
          <span className="id">#{t.id}</span>
        </div>
      ))}
    </div>
  );
}
