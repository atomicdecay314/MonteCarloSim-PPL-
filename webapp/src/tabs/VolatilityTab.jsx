import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'
import { predictVol } from '../api.js'

// ── Feature importance chart ──────────────────────────────────────────────────

const FI_DESCRIPTIONS = {
  rv_har_d:  'HAR-RV daily component',
  rv_har_w:  'HAR-RV weekly (5d)',
  rv_har_m:  'HAR-RV monthly (22d)',
  rv5:       '5-day realized vol',
  rv21:      '21-day realized vol',
  rv63:      '63-day realized vol',
  rv21_lag1: 'RV21 lag-1 (autocorr)',
  rv21_lag5: 'RV21 lag-5',
  sq_ret:    'Squared log return',
  abs_ret:   'Absolute log return',
  jump_5d:   '5d max |return| (jump)',
  bb_width:  'Bollinger Band width',
  rsi:       'RSI-14 momentum',
}

function FeatureImportanceChart({ importances }) {
  if (!importances) return null

  const data = Object.entries(importances)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([k, v]) => ({
      name: k,
      value: Number((v * 100).toFixed(2)),
      desc: FI_DESCRIPTIONS[k] || k,
    }))

  const FI_COLORS = ['#b4c5ff', '#a0b4ff', '#8da4ff', '#7a93ff', '#6682ff',
                     '#5572ff', '#4462ff', '#3352ff', '#2242f5', '#1132e5']

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 5, right: 20, left: 80, bottom: 5 }}
      >
        <XAxis
          type="number"
          tick={{ fill: '#c3c6d7', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          tickFormatter={v => `${v.toFixed(1)}%`}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fill: '#c3c6d7', fontSize: 10, fontFamily: 'JetBrains Mono' }}
          width={78}
        />
        <Tooltip
          formatter={(v, _, props) => [`${v}%`, props.payload.desc]}
          contentStyle={{
            background: 'rgba(29,31,38,0.95)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 8,
            fontFamily: 'JetBrains Mono',
            fontSize: 11,
            color: '#e1e2eb',
          }}
          cursor={{ fill: 'rgba(180,197,255,0.05)' }}
        />
        <Bar dataKey="value" radius={[0, 3, 3, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={FI_COLORS[i % FI_COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Vol gauge ─────────────────────────────────────────────────────────────────

function VolGauge({ label, value, color, subtext }) {
  return (
    <div className="glass-card p-5 flex flex-col gap-2">
      <p className="label-xs">{label}</p>
      <p className="font-mono text-3xl font-medium" style={{ color }}>
        {value !== null && value !== undefined
          ? `${(value * 100).toFixed(2)}%`
          : '—'}
      </p>
      {subtext && <p className="text-xs text-on-surface-variant">{subtext}</p>}
    </div>
  )
}

// ── Main tab ──────────────────────────────────────────────────────────────────

export default function VolatilityTab({ params }) {
  const [ticker, setTicker] = useState(params?.ticker || 'SPY')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await predictVol({
        ticker: ticker.trim().toUpperCase(),
        S0: params?.S0,
        r: params?.r,
        T: params?.T,
        optionType: params?.optionType,
      })
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-headline font-bold text-2xl text-on-surface">Volatility AI</h1>
          <p className="text-sm text-on-surface-variant mt-1">
            Random Forest model trained on HAR-RV + ARCH features (5y history). Predicts 21-day forward realized vol.
          </p>
        </div>

        {/* Engine badge */}
        <div className="glass-card px-3 py-2 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-secondary animate-pulse-slow"
                style={{ boxShadow: '0 0 8px rgba(221,183,255,0.4)' }} />
          <span className="text-xs font-mono text-secondary">Neural Engine v3.0</span>
        </div>
      </div>

      {/* Ticker input + predict button */}
      <div className="glass-card p-5">
        <div className="flex flex-col sm:flex-row items-start sm:items-end gap-4">
          <div className="flex flex-col gap-1.5 flex-1">
            <label className="label-xs">Ticker Symbol</label>
            <input
              type="text"
              value={ticker}
              onChange={e => setTicker(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === 'Enter' && handlePredict()}
              placeholder="e.g. SPY, AAPL, TSLA"
              maxLength={8}
              className="num-input uppercase text-base max-w-xs"
            />
          </div>
          <button
            onClick={handlePredict}
            disabled={loading}
            className="btn-primary flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <span className="material-symbols-outlined text-base animate-spin-slow">sync</span>
                Training RF model…
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-base">psychology</span>
                Predict Volatility
              </>
            )}
          </button>
        </div>

        {error && (
          <p className="mt-3 text-sm font-mono text-error">
            <span className="material-symbols-outlined text-base mr-1">error</span>
            {error}
          </p>
        )}

        <p className="mt-3 text-xs text-on-surface-variant/60">
          Downloads 5 years of OHLCV data, engineers HAR-RV + ARCH features, trains a 500-tree Random Forest,
          and returns the forward 21-day vol forecast. Takes ~10–20s on first run.
        </p>
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Vol gauges */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <VolGauge
              label="RF Predicted Vol"
              value={result.predicted_vol}
              color="#b4c5ff"
              subtext="21-day forward forecast"
            />
            <VolGauge
              label="Historical Vol (21d)"
              value={result.hist_vol}
              color="#4edea3"
              subtext="Trailing realized vol"
            />
            <VolGauge
              label="Market Implied Vol"
              value={result.implied_vol}
              color="#ddb7ff"
              subtext={result.implied_vol ? 'From live options chain' : 'Unavailable'}
            />
          </div>

          {/* R² + metadata */}
          <div className="glass-card p-5 flex items-center justify-between flex-wrap gap-4">
            <div>
              <p className="label-xs mb-1">Model Performance</p>
              <p className="font-mono text-2xl font-medium text-on-surface">
                R² = <span className={result.r2 > 0.5 ? 'text-tertiary' : result.r2 > 0.3 ? 'text-primary' : 'text-error'}>
                  {result.r2.toFixed(4)}
                </span>
              </p>
              <p className="text-xs text-on-surface-variant mt-1">Out-of-sample (20% hold-out set)</p>
            </div>
            <div className="text-right">
              <p className="label-xs mb-1">Ticker</p>
              <p className="font-mono text-2xl font-bold text-primary">{result.ticker}</p>
            </div>
          </div>

          {/* Feature importances */}
          {result.feat_importances && Object.keys(result.feat_importances).length > 0 && (
            <div className="glass-card p-5">
              <h3 className="font-headline font-semibold text-sm text-on-surface mb-1">
                Feature Importances
              </h3>
              <p className="text-xs text-on-surface-variant mb-4">
                Top 10 features by RF split importance. HAR-RV components dominate, confirming volatility clustering.
              </p>
              <FeatureImportanceChart importances={result.feat_importances} />
            </div>
          )}

          {/* Interpretation */}
          <div className="glass-card p-5 flex gap-4">
            <span className="material-symbols-outlined text-secondary text-2xl shrink-0 mt-0.5">lightbulb</span>
            <div>
              <p className="font-headline font-semibold text-sm text-on-surface mb-1">Interpretation</p>
              <p className="text-sm text-on-surface-variant leading-relaxed">
                {result.predicted_vol > result.hist_vol * 1.1
                  ? `The model forecasts vol expansion: predicted ${(result.predicted_vol * 100).toFixed(1)}% vs trailing ${(result.hist_vol * 100).toFixed(1)}%. Consider long vega strategies.`
                  : result.predicted_vol < result.hist_vol * 0.9
                  ? `The model forecasts vol contraction: predicted ${(result.predicted_vol * 100).toFixed(1)}% vs trailing ${(result.hist_vol * 100).toFixed(1)}%. Current options may be overpriced.`
                  : `Stable vol regime expected: predicted ${(result.predicted_vol * 100).toFixed(1)}% ≈ trailing ${(result.hist_vol * 100).toFixed(1)}%. No significant regime shift signalled.`
                }
                {result.implied_vol && ` Market implied vol is ${(result.implied_vol * 100).toFixed(1)}% — ${result.implied_vol > result.predicted_vol ? 'options appear rich vs model.' : 'options appear cheap vs model.'}`}
              </p>
            </div>
          </div>
        </>
      )}

      {!result && !loading && (
        <div className="flex flex-col items-center justify-center py-16 text-on-surface-variant/30 gap-3">
          <span className="material-symbols-outlined text-5xl">psychology</span>
          <p className="text-sm">Enter a ticker and click Predict Volatility</p>
        </div>
      )}
    </div>
  )
}
