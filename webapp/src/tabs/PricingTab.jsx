import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts'

// ── Sub-components ────────────────────────────────────────────────────────────

function PriceCard({ title, callPrice, putPrice, vsBs, badge, accentColor }) {
  const diff = vsBs !== null && vsBs !== undefined
  const sign = diff && vsBs >= 0 ? '+' : ''
  return (
    <div className="glass-card p-5 flex flex-col gap-3">
      <div className="flex items-start justify-between">
        <p className="font-headline font-semibold text-sm text-on-surface">{title}</p>
        {badge && (
          <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-tertiary/10 text-tertiary border border-tertiary/20">
            {badge}
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="label-xs mb-1">Call Price</p>
          <p className="font-mono text-2xl font-medium" style={{ color: accentColor || '#b4c5ff' }}>
            ${callPrice !== null ? callPrice.toFixed(4) : '—'}
          </p>
        </div>
        <div>
          <p className="label-xs mb-1">Put Price</p>
          <p className="font-mono text-2xl font-medium text-secondary">
            ${putPrice !== null ? putPrice.toFixed(4) : '—'}
          </p>
        </div>
      </div>

      {diff && (
        <p className={`text-xs font-mono ${vsBs >= 0 ? 'text-tertiary' : 'text-error'}`}>
          {sign}{vsBs.toFixed(4)} vs BS
        </p>
      )}
    </div>
  )
}

const CHART_COLORS = {
  gbm: '#b4c5ff',
  mjd: '#FF6B35',
  bs:  '#4edea3',
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card px-3 py-2 text-xs font-mono">
      <p className="text-on-surface-variant mb-1">{label} paths</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: ${p.value?.toFixed(4)}
        </p>
      ))}
    </div>
  )
}

function ConvergenceChart({ data }) {
  if (!data?.length) return (
    <div className="flex items-center justify-center h-48 text-on-surface-variant/40 text-sm">
      Run simulation to see convergence
    </div>
  )

  const chartData = data.map(d => ({
    paths: d.paths,
    GBM: d.gbm,
    MJD: d.mjd,
    BS: d.bs,
  }))

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(67,70,85,0.3)" />
        <XAxis
          dataKey="paths"
          tick={{ fill: '#c3c6d7', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          tickFormatter={v => v >= 1000 ? `${v / 1000}K` : v}
        />
        <YAxis
          tick={{ fill: '#c3c6d7', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          tickFormatter={v => `$${v.toFixed(2)}`}
          width={55}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 11, fontFamily: 'Inter' }}
          iconType="plainline"
        />
        <Line type="monotone" dataKey="GBM" stroke={CHART_COLORS.gbm} strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="MJD" stroke={CHART_COLORS.mjd} strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="BS" stroke={CHART_COLORS.bs} strokeWidth={1.5} strokeDasharray="5 4" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function SensitivityTable({ data }) {
  if (!data?.length) return (
    <div className="text-center py-8 text-on-surface-variant/40 text-sm">
      Run simulation to populate sensitivity table
    </div>
  )

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr>
            {['Parameter', 'Value', 'Call ($)', 'Put ($)', 'Δ Call', 'Δ Put'].map(h => (
              <th key={h} className="label-xs text-left pb-3 pr-4 first:pl-0">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr
              key={i}
              className="border-t border-outline-variant/10 hover:bg-surface-high/40 transition-colors"
            >
              <td className="py-2.5 pr-4 font-label text-on-surface">{row.param}</td>
              <td className="py-2.5 pr-4 font-mono text-on-surface-variant">{row.value}</td>
              <td className="py-2.5 pr-4 font-mono text-primary">${row.call?.toFixed(4)}</td>
              <td className="py-2.5 pr-4 font-mono text-secondary">${row.put?.toFixed(4)}</td>
              <td className={`py-2.5 pr-4 font-mono ${row.d_call >= 0 ? 'text-tertiary' : 'text-error'}`}>
                {row.d_call >= 0 ? '+' : ''}{row.d_call?.toFixed(4)}
              </td>
              <td className={`py-2.5 font-mono ${row.d_put >= 0 ? 'text-tertiary' : 'text-error'}`}>
                {row.d_put >= 0 ? '+' : ''}{row.d_put?.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Main tab ──────────────────────────────────────────────────────────────────

export default function PricingTab({ results, loading }) {
  const p = results?.pricing

  const bsCall = p?.bs?.call ?? null
  const bsPut  = p?.bs?.put  ?? null

  const cards = [
    {
      title: 'MC (GBM)',
      callPrice: p?.mc_gbm?.call ?? null,
      putPrice:  p?.mc_gbm?.put  ?? null,
      vsBs: p ? (p.mc_gbm.call - p.bs.call) : null,
      accentColor: '#b4c5ff',
    },
    {
      title: 'MC (MJD)',
      callPrice: p?.mc_mjd?.call ?? null,
      putPrice:  p?.mc_mjd?.put  ?? null,
      vsBs: p ? (p.mc_mjd.call - p.bs.call) : null,
      accentColor: '#FF6B35',
    },
    {
      title: 'Black-Scholes',
      callPrice: bsCall,
      putPrice:  bsPut,
      vsBs: null,
      badge: 'Benchmark',
      accentColor: '#4edea3',
    },
    {
      title: 'Merton Analytical',
      callPrice: p?.merton?.call ?? null,
      putPrice:  p?.merton?.put  ?? null,
      vsBs: p ? (p.merton.call - p.bs.call) : null,
      accentColor: '#ddb7ff',
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      {/* Page header */}
      <div>
        <h1 className="font-headline font-bold text-2xl text-on-surface">Pricing Dashboard</h1>
        <p className="text-sm text-on-surface-variant mt-1">
          Cross-model derivative valuation and convergence analytics.
        </p>
      </div>

      {/* Pricing cards */}
      {loading ? (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          {[0,1,2,3].map(i => (
            <div key={i} className="glass-card p-5 h-36 animate-pulse bg-surface-container/50" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          {cards.map(card => <PriceCard key={card.title} {...card} />)}
        </div>
      )}

      {/* Bottom row: convergence + sensitivity */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Convergence chart */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-headline font-semibold text-sm text-on-surface">
              Price Convergence
            </h3>
            <span className="label-xs">MC paths →</span>
          </div>
          <ConvergenceChart data={results?.convergence} />
        </div>

        {/* Sensitivity table */}
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-headline font-semibold text-sm text-on-surface">
              Sensitivity Table
            </h3>
            <span className="text-xs font-mono text-on-surface-variant/60">Δ per +1 unit bump</span>
          </div>
          <SensitivityTable data={results?.sensitivity} />
        </div>
      </div>
    </div>
  )
}
