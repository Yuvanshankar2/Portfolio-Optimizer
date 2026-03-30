"use client";

interface CorrelationHeatmapProps {
  tickers: string[];
  matrix: number[][];
}

const CELL = 52;
const LEFT_MARGIN = 56;
const TOP_MARGIN = 44;

function corrToColor(v: number): string {
  const c = Math.max(-1, Math.min(1, v));
  let r: number, g: number, b: number;

  if (c < 0) {
    // -1 → red(239,68,68)  to  0 → surface(15,22,41)
    const t = c + 1; // 0..1
    r = Math.round(239 + (15 - 239) * (1 - t));
    g = Math.round(68 + (22 - 68) * (1 - t));
    b = Math.round(68 + (41 - 68) * (1 - t));
  } else {
    // 0 → surface(15,22,41)  to  +1 → cyan(0,212,255)
    const t = c; // 0..1
    r = Math.round(15 + (0 - 15) * t);
    g = Math.round(22 + (212 - 22) * t);
    b = Math.round(41 + (255 - 41) * t);
  }
  return `rgb(${r},${g},${b})`;
}

export default function CorrelationHeatmap({ tickers, matrix }: CorrelationHeatmapProps) {
  const N = tickers.length;
  const svgW = LEFT_MARGIN + N * CELL;
  const svgH = TOP_MARGIN + N * CELL;

  return (
    <div style={{ overflowX: "auto", marginTop: "1.25rem" }}>
      <svg
        viewBox={`0 0 ${svgW} ${svgH}`}
        width={svgW}
        height={svgH}
        aria-label="Asset correlation heatmap"
        style={{ display: "block", fontFamily: "var(--font-mono, monospace)" }}
      >
        {/* Column header labels */}
        {tickers.map((ticker, j) => (
          <text
            key={`col-${j}`}
            x={LEFT_MARGIN + j * CELL + CELL / 2}
            y={TOP_MARGIN - 6}
            textAnchor="middle"
            fontSize={10}
            fill="var(--color-text-secondary, #9ca3af)"
          >
            {ticker}
          </text>
        ))}

        {/* Row header labels */}
        {tickers.map((ticker, i) => (
          <text
            key={`row-${i}`}
            x={LEFT_MARGIN - 6}
            y={TOP_MARGIN + i * CELL + CELL / 2}
            textAnchor="end"
            dominantBaseline="middle"
            fontSize={10}
            fill="var(--color-text-secondary, #9ca3af)"
          >
            {ticker}
          </text>
        ))}

        {/* Heatmap cells */}
        {matrix.map((row, i) =>
          row.map((val, j) => {
            const x = LEFT_MARGIN + j * CELL;
            const y = TOP_MARGIN + i * CELL;
            const bg = corrToColor(val);
            const textFill = val > 0.4 ? "#000" : "var(--color-text-primary, #f9fafb)";

            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={x + 1}
                  y={y + 1}
                  width={CELL - 2}
                  height={CELL - 2}
                  rx={3}
                  fill={bg}
                />
                <text
                  x={x + CELL / 2}
                  y={y + CELL / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fontSize={10}
                  fill={textFill}
                >
                  {val.toFixed(2)}
                </text>
              </g>
            );
          })
        )}
      </svg>

      {/* Color scale legend */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          marginTop: "0.75rem",
          fontSize: "0.72rem",
          fontFamily: "var(--font-mono, monospace)",
          color: "var(--color-text-secondary, #9ca3af)",
        }}
      >
        <span style={{ color: "rgb(239,68,68)" }}>-1.0</span>
        <svg width={120} height={12} style={{ display: "block" }}>
          <defs>
            <linearGradient id="corrGradient" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="rgb(239,68,68)" />
              <stop offset="50%" stopColor="rgb(15,22,41)" />
              <stop offset="100%" stopColor="rgb(0,212,255)" />
            </linearGradient>
          </defs>
          <rect width={120} height={12} rx={3} fill="url(#corrGradient)" />
        </svg>
        <span style={{ color: "rgb(0,212,255)" }}>+1.0</span>
      </div>
    </div>
  );
}
