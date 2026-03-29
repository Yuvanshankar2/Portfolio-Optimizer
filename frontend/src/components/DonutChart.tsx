import styles from "./DonutChart.module.css";

const CHART_COLORS = [
  "#00d4ff",
  "#22c55e",
  "#f59e0b",
  "#a855f7",
  "#ec4899",
  "#f97316",
  "#6366f1",
  "#14b8a6",
];

interface DonutChartProps {
  allocations: Record<string, number>;
  size?: number;
  strokeWidth?: number;
}

export default function DonutChart({ allocations, size = 220, strokeWidth = 32 }: DonutChartProps) {
  const entries = Object.entries(allocations);
  const r = size / 2 - strokeWidth / 2;
  const cx = size / 2;
  const circumference = 2 * Math.PI * r;

  let cumulative = 0;
  const slices = entries.map(([ticker, weight], i) => {
    const dashArray = `${(weight * circumference).toFixed(2)} ${circumference.toFixed(2)}`;
    const dashOffset = -(cumulative * circumference);
    cumulative += weight;
    return {
      ticker,
      weight,
      dashArray,
      dashOffset: dashOffset.toFixed(2),
      color: CHART_COLORS[i % CHART_COLORS.length],
    };
  });

  return (
    <div className={styles.wrapper}>
      <svg width={size} height={size} className={styles.svg}>
        {/* Background ring */}
        <circle
          cx={cx}
          cy={cx}
          r={r}
          fill="none"
          stroke="#1a2a4a"
          strokeWidth={strokeWidth}
        />
        {/* Slice rings — rotated so first slice starts at 12 o'clock */}
        <g transform={`rotate(-90 ${cx} ${cx})`}>
          {slices.map((s) => (
            <circle
              key={s.ticker}
              cx={cx}
              cy={cx}
              r={r}
              fill="none"
              stroke={s.color}
              strokeWidth={strokeWidth}
              strokeDasharray={s.dashArray}
              strokeDashoffset={s.dashOffset}
              strokeLinecap="butt"
            />
          ))}
        </g>
        {/* Center label */}
        <text x={cx} y={cx - 10} className={styles.centerLabel}>
          Assets
        </text>
        <text x={cx} y={cx + 14} className={styles.centerValue}>
          {entries.length}
        </text>
      </svg>

      <div className={styles.legend}>
        {slices.map((s) => (
          <div key={s.ticker} className={styles.legendItem}>
            <div className={styles.legendDot} style={{ background: s.color }} />
            <span className={styles.legendLabel}>{s.ticker}</span>
            <span className={styles.legendValue}>{(s.weight * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
