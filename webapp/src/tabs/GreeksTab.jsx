import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

const GREEK_CONFIG = [
  {
    key:    'delta',
    symbol: 'Δ',
    label:  'Delta',
    desc:   'Sensitivity to spot price. Range [-1, 1].',
    fmt:    v => v.toFixed(4),
  },
  {
    key:    'gamma',
    symbol: 'Γ',
    label:  'Gamma',
    desc:   'Rate of change of delta. Second derivative of price w.r.t. spot.',
    fmt:    v => v.toFixed(6),
  },
  {
    key:    'vega',
    symbol: 'ν',
    label:  'Vega',
    desc:   'Price change per +1% volatility (absolute).',
    fmt:    v => `$${v.toFixed(4)}`,
  },
  {
    key:    'theta',
    symbol: 'θ',
    label:  'Theta',
    desc:   'Price decay per calendar day ($).',
    fmt:    v => `$${v.toFixed(4)}`,
  },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card px-3 py-2 text-xs font-mono">
      <p className="text-on-surface-variant mb-1">S = ${Number(label).toFixed(2)}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(5)}
        </p>
      ))}
    </div>
  )
}

function GreekChart({ cfg, results }) {
  if (!results) {
    return (
      <div className="flex items-center justify-center h-44 text-on-surface-variant/40 text-sm">
        Run simulation
      </div>
    )
  }

  const { spot_range, bs, mc, S0, K } = results
  const chartData = spot_range.map((s, i) => ({
    spot: s.toFixed(1),
    BS: bs[cfg.key]?.[i],
    MC: mc[cfg.key]?.[i],
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={chartData} margin={{ top: 5, right: 15, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(67,70,85,0.3)" />
        <XAxis
          dataKey="spot"
          tick={{ fill: '#c3c6d7', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickFormatter={v => `$${Number(v).toFixed(0)}`}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: '#c3c6d7', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          width={60}
          tickFormatter={v => v.toFixed(3)}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 11 }} />
        {S0 && (
          <ReferenceLine
            x={S0.toFixed(1)}
            stroke="rgba(255,107,107,0.6)"
            strokeDasharray="4 3"
            label={{ value: 'S₀', fill: '#ff6b6b', fontSize: 10 }}
          />
        )}
        {K && (
          <ReferenceLine
            x={K.toFixed(1)}
            stroke="rgba(152,251,152,0.6)"
            strokeDasharray="4 3"
            label={{ value: 'K', fill: '#98fb98', fontSize: 10 }}
          />
        )}
        <Line
          type="monotone" dataKey="BS" name="Black-Scholes"
          stroke="#FFD700" strokeWidth={2} dot={false}
        />
        <Line
          type="monotone" dataKey="MC" name="MC (finite diff)"
          stroke="#00BFFF" strokeWidth={1.5} strokeDasharray="5 3" dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

function GreekCard({ cfg, value }) {
  return (
    <div className="glass-card p-4 flex items-center gap-4">
      <div
        className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0 font-headline font-bold text-xl"
        style={{ background: 'rgba(180,197,255,0.1)', color: '#b4c5ff' }}
      >
        {cfg.symbol}
      </div>
      <div>
        <p className="label-xs">{cfg.label}</p>
        <p className="font-mono text-lg font-medium text-on-surface mt-0.5">
          {value !== undefined && value !== null ? cfg.fmt(value) : '—'}
        </p>
      </div>
    </div>
  )
}

export default function GreeksTab({ results, loading }) {
  const current = results?.current

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div>
        <h1 className="font-headline font-bold text-2xl text-on-surface">Greeks Analysis</h1>
        <p className="text-sm text-on-surface-variant mt-1">
          Black-Scholes (gold) vs Monte Carlo finite differences (cyan) across spot price range.
        </p>
      </div>

      {/* Current Greeks summary */}
      {loading ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          {[0,1,2,3].map(i => (
            <div key={i} className="glass-card p-4 h-20 animate-pulse bg-surface-container/50" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          {GREEK_CONFIG.map(cfg => (
            <GreekCard key={cfg.key} cfg={cfg} value={current?.[cfg.key]} />
          ))}
        </div>
      )}

      {/* Charts grid */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        {GREEK_CONFIG.map(cfg => (
          <div key={cfg.key} className="glass-card p-5">
            <div className="flex items-center justify-between mb-1">
              <h3 className="font-headline font-semibold text-sm text-on-surface">
                {cfg.symbol} — {cfg.label}
              </h3>
              {current?.[cfg.key] !== undefined && (
                <span className="font-mono text-xs text-primary">
                  @ S₀: {cfg.fmt(current[cfg.key])}
                </span>
              )}
            </div>
            <p className="text-xs text-on-surface-variant mb-3">{cfg.desc}</p>
            <GreekChart cfg={cfg} results={results} />
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 text-xs text-on-surface-variant">
        <div className="flex items-center gap-2">
          <span className="inline-block w-5 h-0.5 bg-[#FFD700]" /> Black-Scholes (analytical)
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-5 h-0.5 bg-[#00BFFF] border-dashed" /> MC finite diff
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-5 h-0.5 bg-[#FF6B6B]" /> Current spot (S₀)
        </div>
        <div className="flex items-center gap-2">
          <span className="inline-block w-5 h-0.5 bg-[#98FB98]" /> Strike (K)
        </div>
      </div>
    </div>
  )
}
