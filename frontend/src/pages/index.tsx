import Head from "next/head";
import Dashboard from "@/components/Dashboard";
import styles from "./index.module.css";

const DISCLAIMER =
  "For educational and demonstration purposes only. " +
  "Not financial advice. Not intended for live trading.";

export default function Home() {
  return (
    <>
      <Head>
        <title>Alpha Engine — Portfolio Optimizer</title>
        <meta name="description" content="AI-driven portfolio optimization demo" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link
          rel="preconnect"
          href="https://fonts.googleapis.com"
        />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin=""
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap"
          rel="stylesheet"
        />
      </Head>

      <div className={styles.shell}>

        {/* ── Header bar ── */}
        <header className={styles.header}>
          <div className={styles.headerLogo}>
            <span className={styles.logoMark}>◈</span>
            <span className={styles.logoText}>
              Alpha Engine <em>Live</em>
            </span>
          </div>
          <span className={styles.headerMeta}>
            Transformer + PPO RL · Educational Demo
          </span>
        </header>

        {/* ── Hero value strip ── */}
        <div className={styles.heroStrip}>
          <span className={styles.heroValue}>$1,248,392.40</span>
          <span className={styles.heroBadge}>+24.6%</span>
          <span className={styles.heroLabel}>Simulated Portfolio Value</span>
        </div>

        {/* ── Disclaimer ── */}
        <div className={styles.disclaimer}>
          <span className={styles.disclaimerIcon}>⚠</span>
          <span>
            <strong>Disclaimer:</strong> {DISCLAIMER}
          </span>
        </div>

        {/* ── Dashboard ── */}
        <main className={styles.main}>
          <Dashboard />
        </main>

        {/* ── Bottom navigation ── */}
        <nav className={styles.bottomNav}>
          <button className={`${styles.navItem} ${styles.navItemActive}`}>
            <span className={styles.navIcon}>⬡</span>
            <span className={styles.navLabel}>Dashboard</span>
          </button>
          <button className={styles.navItem}>
            <span className={styles.navIcon}>◈</span>
            <span className={styles.navLabel}>Portfolio</span>
          </button>
          <button className={styles.navItem}>
            <span className={styles.navIcon}>⌁</span>
            <span className={styles.navLabel}>Backtest</span>
          </button>
          <button className={styles.navItem}>
            <span className={styles.navIcon}>⚙</span>
            <span className={styles.navLabel}>Settings</span>
          </button>
        </nav>

      </div>
    </>
  );
}
