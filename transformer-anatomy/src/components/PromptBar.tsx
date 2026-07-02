interface PromptBarProps {
  value: string;
  onChange: (v: string) => void;
  onRun: () => void;
  maxLen: number;
}

export function PromptBar({ value, onChange, onRun, maxLen }: PromptBarProps) {
  return (
    <>
      <div className="controls">
        <input
          type="text"
          className="mono"
          value={value}
          spellCheck={false}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') onRun(); }}
        />
        <button onClick={onRun}>Run ▶</button>
      </div>
      <p className="hint">
        Only characters the model saw during training (English letters, basic punctuation, newline, space) are
        kept — anything else is silently dropped. Context window is {maxLen} characters; longer prompts keep
        only the most recent {maxLen}.
      </p>
    </>
  );
}
