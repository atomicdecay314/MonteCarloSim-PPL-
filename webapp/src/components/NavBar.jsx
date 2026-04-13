const TABS = [
  { id: 'pricing',    label: 'Pricing',      icon: 'query_stats' },
  { id: 'greeks',     label: 'Greeks',       icon: 'functions' },
  { id: 'paths',      label: '3D Paths',     icon: 'view_in_ar' },
  { id: 'volatility', label: 'Volatility AI', icon: 'psychology' },
]

export default function NavBar({ activeTab, onTabChange }) {
  return (
    <nav className="flex items-center justify-between px-6 py-3 bg-surface border-b border-outline-variant/10 shrink-0 z-10">
      {/* Logo */}
      <div className="flex items-center gap-2">
        <span className="material-symbols-outlined text-primary">query_stats</span>
        <span className="font-headline font-bold text-lg text-on-surface tracking-tight">
          QuantSim <span className="text-primary">Pro</span>
        </span>
        <span className="ml-2 text-xs font-mono text-on-surface-variant/60 hidden sm:inline">
          v2.4.0-stable
        </span>
      </div>

      {/* Tab navigation */}
      <div className="seg-track">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`seg-pill flex items-center gap-1.5 ${activeTab === tab.id ? 'active' : ''}`}
          >
            <span className="material-symbols-outlined text-base leading-none">{tab.icon}</span>
            <span className="hidden md:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-2">
        {/* Live market status dot */}
        <div className="flex items-center gap-1.5 text-xs font-mono text-tertiary">
          <span className="live-dot" />
          <span className="hidden sm:inline">MARKET OPEN</span>
        </div>
      </div>
    </nav>
  )
}
