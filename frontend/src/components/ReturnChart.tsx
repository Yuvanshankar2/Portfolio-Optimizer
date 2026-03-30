"use client";

interface SeriesData {
  label: string;
  data: number[];
  color: string;
}

interface ReturnChartProps {
  series: SeriesData[];
  height?: number;
}

const PAD = { top: 12, right: 8, bottom: 4, left: 8 };
const W = 500;

export default function ReturnChart({ series, height = 160 }: ReturnChartProps) {
  const activeSeries = series.filter((s) => s.data.length > 0);
  if (activeSeries.length === 0) return null;

  const allValues = activeSeries.flatMap((s) => s.data);
  const yMin = Math.min(...allValues);
  const yMax = Math.max(...allValues);
  const yRange = yMax - yMin || 1;

  const maxLen = Math.max(...activeSeries.map((s) => s.data.length));
  const innerW = W - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;

  function toPoints(data: number[]): string {
    return data
      .map((v, i) => {
        const x = PAD.left + (i / (maxLen - 1 || 1)) * innerW;
        const y = PAD.top + (1 - (v - yMin) / yRange) * innerH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }

  // Zero line
  const zeroY = PAD.top + (1 - (0 - yMin) / yRange) * innerH;

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${height}`}
        style={{ width: "100%", display: "block" }}
        aria-label="Cumulative returns chart"
      >
        {/* Zero baseline */}
        <line
          x1={PAD.left}
          y1={zeroY.toFixed(1)}
          x2={W - PAD.right}
          y2={zeroY.toFixed(1)}
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="1"
        />

        {activeSeries.map((s) => (
          <polyline
            key={s.label}
            points={toPoints(s.data)}
            fill="none"
            stroke={s.color}
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
      </svg>

      {/* Legend */}
      <div
        style={{
          display: "flex",
          gap: "1rem",
          marginTop: "0.5rem",
          fontSize: "0.72rem",
          fontFamily: "var(--font-ui)",
          flexWrap: "wrap",
        }}
      >
        {activeSeries.map((s) => (
          <span
            key={s.label}
            style={{ display: "flex", alignItems: "center", gap: "0.3rem", color: "var(--color-text-secondary)" }}
          >
            <span
              style={{
                display: "inline-block",
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: s.color,
                flexShrink: 0,
              }}
            />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
