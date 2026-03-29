"use client";

import { useRef } from "react";
import styles from "./AgentStatus.module.css";

interface AgentStatusProps {
  modelVersion: string;
  configLoaded: boolean;
  useSynthetic: boolean;
  lastUpdated: string | null;
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch {
    return ts;
  }
}

export default function AgentStatus({ modelVersion, configLoaded, useSynthetic, lastUpdated }: AgentStatusProps) {
  // Stable mock latency — seeded once on mount, never re-randomizes
  const latencyRef = useRef<number>(Math.round(Math.random() * 40 + 8));

  return (
    <div className={styles.bar}>
      <span className={styles.barLabel}>RL Agent Status</span>
      <div className={styles.divider} />
      <div className={styles.pills}>

        {/* STATUS */}
        <div className={styles.pill}>
          <div className={`${styles.dot} ${configLoaded ? styles.dotGreen : styles.dotRed}`} />
          <span className={styles.pillKey}>Status</span>
          <span className={styles.pillVal}>{configLoaded ? "Active" : "Offline"}</span>
        </div>

        {/* MODE */}
        <div className={styles.pill}>
          <div className={`${styles.dot} ${useSynthetic ? styles.dotCyan : styles.dotAmber}`} />
          <span className={styles.pillKey}>Mode</span>
          <span className={styles.pillVal}>{useSynthetic ? "Synthetic" : "Live"}</span>
        </div>

        {/* MODEL */}
        <div className={styles.pill}>
          <span className={styles.pillKey}>Model</span>
          <span className={styles.pillVal}>{modelVersion}</span>
        </div>

        {/* LATENCY */}
        <div className={styles.pill}>
          <div className={`${styles.dot} ${styles.dotCyan}`} />
          <span className={styles.pillKey}>Latency</span>
          <span className={styles.pillVal}>{latencyRef.current}ms</span>
        </div>

        {/* LAST UPDATE */}
        <div className={styles.pill}>
          <span className={styles.pillKey}>Updated</span>
          <span className={styles.pillVal}>{formatTimestamp(lastUpdated)}</span>
        </div>

      </div>
    </div>
  );
}
