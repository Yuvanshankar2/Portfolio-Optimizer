"use client";

import styles from "./TabNav.module.css";

export type TabId = "portfolio" | "ai-trades" | "drawdowns";

const TABS: { id: TabId; label: string }[] = [
  { id: "portfolio", label: "Portfolio Performance" },
  { id: "ai-trades", label: "AI Trades" },
  { id: "drawdowns", label: "Drawdowns" },
];

interface TabNavProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

export default function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <nav className={styles.nav}>
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`${styles.tab} ${activeTab === tab.id ? styles.tabActive : ""}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
