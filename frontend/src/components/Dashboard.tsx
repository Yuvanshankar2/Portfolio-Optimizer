"use client";

import { useEffect, useState } from "react";
import styles from "./Dashboard.module.css";
import TabNav, { TabId } from "./TabNav";
import AgentStatus from "./AgentStatus";
import DonutChart from "./DonutChart";
import ReturnChart from "./ReturnChart";
import CorrelationHeatmap from "./CorrelationHeatmap";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BackendConfig {
  tickers: string[];
  start_date: string;
  end_date: string;
  use_synthetic: boolean;
}

type RiskProfile = "conservative" | "moderate" | "aggressive";

interface AllocateResponse {
  allocations: Record<string, number>;
  dollar_allocations?: Record<string, number>;
  model_version: string;
  risk_profile: RiskProfile;
  timestamp: string;
  disclaimer: string;
}

interface BacktestResponse {
  sharpe_ratio: number;
  max_drawdown: number;
  total_return: number;
  annualized_return: number;
  annualized_volatility: number;
  calmar_ratio: number;
  sortino_ratio: number;
  var_95: number;
  cumulative_returns: number[];
  benchmark_cumulative_returns: number[];
  spy_benchmark_returns: number[];
  num_rebalances: number;
  disclaimer: string;
}

interface CorrelationResponse {
  tickers: string[];
  matrix: number[][];
  start_date: string;
  end_date: string;
  disclaimer: string;
}

// ---------------------------------------------------------------------------
// Error helper — FastAPI returns JSON {detail: string | [{msg,loc,type}]}
// ---------------------------------------------------------------------------

async function parseApiError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((e: { msg: string }) => e.msg).join("; ");
    }
    return JSON.stringify(body);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

// ---------------------------------------------------------------------------
// Internal sub-component — full backtest metrics table
// ---------------------------------------------------------------------------

function BacktestMetricsCard({ result }: { result: BacktestResponse }) {
  const rows: [string, string, "positive" | "negative" | "neutral"][] = [
    ["Total Return", `${(result.total_return * 100).toFixed(2)}%`, result.total_return >= 0 ? "positive" : "negative"],
    ["Annualised Return", `${(result.annualized_return * 100).toFixed(2)}%`, result.annualized_return >= 0 ? "positive" : "negative"],
    ["Annualised Volatility", `${(result.annualized_volatility * 100).toFixed(2)}%`, "neutral"],
    ["Sharpe Ratio", result.sharpe_ratio.toFixed(3), result.sharpe_ratio >= 0 ? "positive" : "negative"],
    ["Sortino Ratio", result.sortino_ratio.toFixed(3), result.sortino_ratio >= 1 ? "positive" : result.sortino_ratio >= 0 ? "neutral" : "negative"],
    ["VaR 95%", `${(result.var_95 * 100).toFixed(2)}%`, "negative"],
    ["Max Drawdown", `${(result.max_drawdown * 100).toFixed(2)}%`, "negative"],
    ["Calmar Ratio", result.calmar_ratio.toFixed(3), result.calmar_ratio >= 0 ? "positive" : "negative"],
    ["Rebalance Steps", String(result.num_rebalances), "neutral"],
  ];

  return (
    <section className={styles.card}>
      <h2 className={styles.cardTitle}>Backtest Metrics</h2>
      <table className={styles.table}>
        <thead>
          <tr>
            <th className={styles.th}>Metric</th>
            <th className={styles.th}>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, value, sentiment]) => (
            <tr key={label} className={styles.tr}>
              <td className={styles.td}>{label}</td>
              <td className={`${styles.td} ${styles[sentiment]}`}>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <ReturnChart
        series={[
          { label: "Strategy", data: result.cumulative_returns, color: "var(--color-accent)" },
          { label: "Equal Weight", data: result.benchmark_cumulative_returns, color: "#6b7280" },
          { label: "SPY", data: result.spy_benchmark_returns, color: "#f59e0b" },
        ]}
      />
    </section>
  );
}

// ---------------------------------------------------------------------------
// Dashboard component
// ---------------------------------------------------------------------------

export default function Dashboard() {
  const [config, setConfig] = useState<BackendConfig | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [allocations, setAllocations] = useState<Record<string, number> | null>(null);
  const [dollarAllocations, setDollarAllocations] = useState<Record<string, number> | null>(null);
  const [investmentAmount, setInvestmentAmount] = useState<string>("10000");
  const [backtestResult, setBacktestResult] = useState<BacktestResponse | null>(null);
  const [allocLoading, setAllocLoading] = useState(false);
  const [backtestLoading, setBacktestLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("portfolio");
  const [allocModelVersion, setAllocModelVersion] = useState<string>("latest");
  const [allocTimestamp, setAllocTimestamp] = useState<string | null>(null);
  const [riskProfile, setRiskProfile] = useState<RiskProfile>("moderate");
  const [correlationData, setCorrelationData] = useState<CorrelationResponse | null>(null);
  const [corrLoading, setCorrLoading] = useState(false);

  // ── Fetch backend config on mount ────────────────────────────────────────

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/config");
        if (!res.ok) {
          const detail = await parseApiError(res);
          throw new Error(`Config fetch failed — ${res.status}: ${detail}`);
        }
        setConfig(await res.json());
      } catch (err: unknown) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setConfigLoading(false);
      }
    })();
  }, []);

  // ── Auto-fetch correlation when tab is first visited ────────────────────

  useEffect(() => {
    if (activeTab === "correlation" && config && !correlationData && !corrLoading) {
      fetchCorrelation();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, config]);

  // ── Fetch portfolio allocation ──────────────────────────────────────────

  async function fetchAllocations() {
    if (!config) return;
    setAllocLoading(true);
    setError(null);
    setDollarAllocations(null);

    try {
      const parsedAmount = parseFloat(investmentAmount);
      const res = await fetch("/api/portfolio/allocate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: config.tickers,
          model_version: "latest",
          use_synthetic: config.use_synthetic,
          investment_amount: parsedAmount > 0 ? parsedAmount : null,
          risk_profile: riskProfile,
        }),
      });

      if (!res.ok) {
        const detail = await parseApiError(res);
        throw new Error(`${res.status}: ${detail}`);
      }

      const data: AllocateResponse = await res.json();
      setAllocations(data.allocations);
      setDollarAllocations(data.dollar_allocations ?? null);
      setAllocModelVersion(data.model_version);
      setAllocTimestamp(data.timestamp);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setAllocLoading(false);
    }
  }

  // ── Run backtest ────────────────────────────────────────────────────────

  async function runBacktest() {
    if (!config) return;
    setBacktestLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/backtest/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tickers: config.tickers,
          start_date: config.start_date,
          end_date: config.end_date,
          model_version: "latest",
          rebalance_freq: "daily",
          transaction_cost: 0.001,
          initial_capital: 10000,
        }),
      });

      if (!res.ok) {
        const detail = await parseApiError(res);
        throw new Error(`${res.status}: ${detail}`);
      }

      const data: BacktestResponse = await res.json();
      setBacktestResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBacktestLoading(false);
    }
  }

  // ── Fetch correlation matrix ─────────────────────────────────────────────

  async function fetchCorrelation() {
    if (!config) return;
    setCorrLoading(true);
    setError(null);
    try {
      const syntheticParam = config.use_synthetic ? "?use_synthetic=true" : "";
      const res = await fetch(`/api/portfolio/correlation${syntheticParam}`);
      if (!res.ok) {
        const detail = await parseApiError(res);
        throw new Error(`${res.status}: ${detail}`);
      }
      setCorrelationData(await res.json());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setCorrLoading(false);
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────

  if (configLoading) {
    return <p className={styles.loading}>Loading configuration…</p>;
  }

  return (
    <div className={styles.dashboard}>

      {/* ── RL Agent status bar ── */}
      <AgentStatus
        modelVersion={allocModelVersion}
        configLoaded={config !== null}
        useSynthetic={config?.use_synthetic ?? true}
        lastUpdated={allocTimestamp}
      />

      {/* ── Tab navigation ── */}
      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      {/* ── Tab: Portfolio Performance ── */}
      {activeTab === "portfolio" && (
        <div className={styles.tabContent}>
          <div className={styles.twoCol}>

            {/* Allocation table */}
            <section className={styles.card}>
              <h2 className={styles.cardTitle}>Current Allocation</h2>
              <p className={styles.cardMeta}>
                {config
                  ? `Tickers: ${config.tickers.join(", ")} · Model: ${allocModelVersion}`
                  : "Configuration unavailable"}
              </p>
              <div className={styles.riskSelector}>
                {(["conservative", "moderate", "aggressive"] as const).map((p) => (
                  <button
                    key={p}
                    className={`${styles.riskBtn} ${riskProfile === p ? styles.riskBtnActive : ""}`}
                    onClick={() => setRiskProfile(p)}
                  >
                    {p.charAt(0).toUpperCase() + p.slice(1)}
                  </button>
                ))}
              </div>
              <div className={styles.inputRow}>
                <label className={styles.inputLabel}>Capital ($)</label>
                <input
                  type="number"
                  min="1"
                  value={investmentAmount}
                  onChange={(e) => setInvestmentAmount(e.target.value)}
                  className={styles.inputField}
                />
              </div>
              <button
                onClick={fetchAllocations}
                disabled={!config || allocLoading}
                className={styles.btn}
              >
                {allocLoading ? "Loading…" : "Get Allocation"}
              </button>
              {allocations && (
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th className={styles.th}>Ticker</th>
                      <th className={styles.th}>Weight</th>
                      {dollarAllocations && <th className={styles.th}>Amount ($)</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(allocations).map(([ticker, weight]) => (
                      <tr key={ticker} className={styles.tr}>
                        <td className={styles.td}>{ticker}</td>
                        <td className={`${styles.td} ${styles.mono}`}>
                          {(weight * 100).toFixed(2)}%
                        </td>
                        {dollarAllocations && (
                          <td className={`${styles.td} ${styles.mono}`}>
                            ${dollarAllocations[ticker].toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>

            {/* Donut chart */}
            {allocations && (
              <section className={styles.card}>
                <h2 className={styles.cardTitle}>Weight Distribution</h2>
                <DonutChart allocations={allocations} />
              </section>
            )}
          </div>

          {/* Full backtest metrics (shown when available) */}
          {backtestResult && <BacktestMetricsCard result={backtestResult} />}
        </div>
      )}

      {/* ── Tab: AI Trades ── */}
      {activeTab === "ai-trades" && (
        <div className={styles.tabContent}>
          <section className={styles.card}>
            <h2 className={styles.cardTitle}>AI Allocation Signal</h2>
            <p className={styles.cardMeta}>
              {config
                ? `Tickers: ${config.tickers.join(", ")} · Model: ${allocModelVersion}`
                : "Configuration unavailable"}
            </p>
            <div className={styles.riskSelector}>
              {(["conservative", "moderate", "aggressive"] as const).map((p) => (
                <button
                  key={p}
                  className={`${styles.riskBtn} ${riskProfile === p ? styles.riskBtnActive : ""}`}
                  onClick={() => setRiskProfile(p)}
                >
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>
            <div className={styles.inputRow}>
              <label className={styles.inputLabel}>Capital ($)</label>
              <input
                type="number"
                min="1"
                value={investmentAmount}
                onChange={(e) => setInvestmentAmount(e.target.value)}
                className={styles.inputField}
              />
            </div>
            <button
              onClick={fetchAllocations}
              disabled={!config || allocLoading}
              className={styles.btn}
            >
              {allocLoading ? "Loading…" : "Get Allocation"}
            </button>
            {allocations && (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th className={styles.th}>Ticker</th>
                    <th className={styles.th}>Weight</th>
                    {dollarAllocations && <th className={styles.th}>Amount ($)</th>}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(allocations).map(([ticker, weight]) => (
                    <tr key={ticker} className={styles.tr}>
                      <td className={styles.td}>{ticker}</td>
                      <td className={`${styles.td} ${styles.mono}`}>
                        {(weight * 100).toFixed(2)}%
                      </td>
                      {dollarAllocations && (
                        <td className={`${styles.td} ${styles.mono}`}>
                          ${dollarAllocations[ticker].toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>

          <section className={`${styles.card} ${styles.rebalanceSuggested}`}>
            <h2 className={styles.cardTitle}>Rebalance Suggested</h2>
            <p className={`${styles.rebalanceHint} ${allocations ? styles.rebalanceHintActive : ""}`}>
              {allocations
                ? "RL Agent recommends rebalancing based on latest market features. Run allocation to refresh signals."
                : "Run allocation to receive rebalance suggestions from the RL agent."}
            </p>
          </section>
        </div>
      )}

      {/* ── Tab: Drawdowns ── */}
      {activeTab === "drawdowns" && (
        <div className={styles.tabContent}>
          <section className={styles.card}>
            <h2 className={styles.cardTitle}>Drawdown Analysis</h2>
            <p className={styles.cardMeta}>
              {config
                ? `Period: ${config.start_date} → ${config.end_date} · Freq: daily · TC: 0.1%`
                : "Configuration unavailable"}
            </p>
            <button
              onClick={runBacktest}
              disabled={!config || backtestLoading}
              className={styles.btn}
            >
              {backtestLoading ? "Running…" : "Run Backtest"}
            </button>
            {backtestResult && (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th className={styles.th}>Metric</th>
                    <th className={styles.th}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ["Max Drawdown", `${(backtestResult.max_drawdown * 100).toFixed(2)}%`, "negative"],
                      ["Calmar Ratio", backtestResult.calmar_ratio.toFixed(3), backtestResult.calmar_ratio >= 0 ? "positive" : "negative"],
                      ["Total Return", `${(backtestResult.total_return * 100).toFixed(2)}%`, backtestResult.total_return >= 0 ? "positive" : "negative"],
                      ["Sharpe Ratio", backtestResult.sharpe_ratio.toFixed(3), backtestResult.sharpe_ratio >= 0 ? "positive" : "negative"],
                      ["Sortino Ratio", backtestResult.sortino_ratio.toFixed(3), backtestResult.sortino_ratio >= 1 ? "positive" : backtestResult.sortino_ratio >= 0 ? "neutral" : "negative"],
                      ["VaR 95%", `${(backtestResult.var_95 * 100).toFixed(2)}%`, "negative"],
                    ] as [string, string, "positive" | "negative" | "neutral"][]
                  ).map(([label, value, sentiment]) => (
                    <tr key={label} className={styles.tr}>
                      <td className={styles.td}>{label}</td>
                      <td className={`${styles.td} ${styles[sentiment]}`}>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </div>
      )}

      {/* ── Tab: Correlation ── */}
      {activeTab === "correlation" && (
        <div className={styles.tabContent}>
          <section className={styles.card}>
            <h2 className={styles.cardTitle}>Asset Correlation</h2>
            <p className={styles.cardMeta}>
              {config
                ? `Log-return correlations · ${config.start_date} to ${config.end_date} · ${config.tickers.length} assets`
                : "Configuration unavailable"}
            </p>
            <button
              onClick={fetchCorrelation}
              disabled={!config || corrLoading}
              className={styles.btn}
            >
              {corrLoading ? "Loading…" : "Refresh Correlations"}
            </button>
            {correlationData && (
              <CorrelationHeatmap
                tickers={correlationData.tickers}
                matrix={correlationData.matrix}
              />
            )}
          </section>
        </div>
      )}

      {/* ── Error display ── */}
      {error && (
        <div className={styles.errorBox}>
          <strong>Error:</strong> {error}
        </div>
      )}
    </div>
  );
}
