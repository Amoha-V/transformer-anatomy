import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { FigurePanel } from './FigurePanel';

interface TrainingLogEntry {
  step: number;
  train_loss: number;
  val_loss: number;
}

interface LossCurveFigureProps {
  log: TrainingLogEntry[];
  nParams: number;
}

export function LossCurveFigure({ log, nParams }: LossCurveFigureProps) {
  const final = log[log.length - 1];
  return (
    <FigurePanel
      tag="FIG. 05 — TRAINING HISTORY"
      title="How this model actually learned"
      description={
        <>This isn't simulated. The chart below is the real loss curve recorded while training this{' '}
        {nParams.toLocaleString()}-parameter model on tiny-shakespeare with AdamW — cross-entropy loss on
        next-character prediction, evaluated on a held-out validation split every 250 steps.</>
      }
      footer={`Final: train loss ${final.train_loss.toFixed(3)}, val loss ${final.val_loss.toFixed(3)}. The gap between them is normal overfitting at this scale — a model this small memorizes some of tiny-shakespeare rather than generalizing far beyond it.`}
    >
      <div style={{ width: '100%', height: 260 }}>
        <ResponsiveContainer>
          <LineChart data={log} margin={{ top: 12, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" />
            <XAxis dataKey="step" tick={{ fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--ink-soft)' }} />
            <YAxis tick={{ fontFamily: 'JetBrains Mono', fontSize: 11, fill: 'var(--ink-soft)' }} width={36} />
            <Tooltip contentStyle={{ fontFamily: 'JetBrains Mono', fontSize: 12, border: '1.5px solid var(--ink)', borderRadius: 2 }} />
            <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 12 }} />
            <Line type="monotone" dataKey="train_loss" name="train loss" stroke="var(--amber-deep)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="val_loss" name="val loss" stroke="var(--teal)" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </FigurePanel>
  );
}
