import { useState } from 'react'
import { fetchLiveMarket } from '../api.js'

function NumField({ label, id, value, step, min, max, onChange, unit }) {
  const dec = () => {
    const next = Math.round((Number(value) - Number(step)) * 1e9) / 1e9
    onChange(id, min !== undefined ? Math.max(min, next) : next)
  }
  const inc = () => {
    const next = Math.round((Number(value) + Number(step)) * 1e9) / 1e9
    onChange(id, max !== undefined ? Math.min(max, next) : next)
  }

  return (
    <div className="flex flex-col gap-1">
      <label className="label-xs">{label}{unit ? ` (${unit})` : ''}</label>
      <div className="flex items-center gap-1">
        <button
          onClick={dec}
          className="w-7 h-7 rounded-md bg-surface-high hover:bg-surface-bright text-on-surface-variant text-lg leading-none flex items-center justify-center transition-colors"
        >−</button>
        <input
          type="number"
          value={value}
          step={step}
          min={min}
          max={max}
          onChange={e => onChange(id, Number(e.target.value))}
          className="num-input text-center flex-1 min-w-0"
        />
        <button
          onClick={inc}
          className="w-7 h-7 rounded-md bg-surface-high hover:bg-surface-bright text-on-surface-variant text-lg leading-none flex items-center justify-center transition-colors"
        >+</button>
      </div>
    </div>
  )
}

export default function SimulationParams({ params, onChange, onRun, loading }) {
  const [ticker, setTicker] = useState('')
  const [fetchStatus, setFetchStatus] = useState(null)
  const [fetchingMarket, setFetchingMarket] = useState(false)

  const handleFetch = async () => {
    if (!ticker.trim()) return
    setFetchingMarket(true)
    setFetchStatus(null)
    try {
      const data = await fetchLiveMarket(ticker.trim())
      onChange('S0', Math.round(data.price * 100) / 100)
      onChange('K', Math.round(data.price * 100) / 100)
      onChange('sigma', Math.round(data.hist_vol * 10000) / 10000)
      setFetchStatus({ ok: true, msg: `${data.ticker}: $${data.price.toFixed(2)} | σ=${(data.hist_vol * 100).toFixed(1)}%` })
    } catch (e) {
      setFetchStatus({ ok: false, msg: e.message })
    } finally {
      setFetchingMarket(false)
    }
  }

  return (
    <aside className="w-64 shrink-0 bg-surface border-r border-outline-variant/10 overflow-y-auto flex flex-col p-4 gap-4">
      {/* Header */}
      <div>
        <p className="font-headline font-semibold text-sm text-on-surface">Simulation Parameters</p>
        <p className="text-xs text-on-surface-variant mt-0.5">Monte Carlo Engine v2.4</p>
      </div>

      {/* Ticker lookup */}
      <div className="flex flex-col gap-2">
        <label className="label-xs">Ticker Lookup</label>
        <div className="flex gap-1">
          <input
            type="text"
            value={ticker}
            onChange={e => setTicker(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleFetch()}
            placeholder="e.g. AAPL"
            maxLength={8}
            className="num-input flex-1 text-xs uppercase"
          />
          <button
            onClick={handleFetch}
            disabled={fetchingMarket}
            className="px-2 py-1 rounded-lg bg-surface-high hover:bg-surface-bright text-xs text-primary font-label transition-colors disabled:opacity-50"
          >
            {fetchingMarket ? '…' : 'Fetch'}
          </button>
        </div>
        {fetchStatus && (
          <p className={`text-xs font-mono ${fetchStatus.ok ? 'text-tertiary' : 'text-error'}`}>
            {fetchStatus.msg}
          </p>
        )}
      </div>

      {/* Market parameters */}
      <div className="flex flex-col gap-3">
        <p className="label-xs border-b border-outline-variant/20 pb-1">Market Parameters</p>
        <NumField label="Spot Price S₀" id="S0" value={params.S0} step={1} min={1} onChange={onChange} unit="$" />
        <NumField label="Strike K" id="K" value={params.K} step={5} min={1} onChange={onChange} unit="$" />
        <NumField label="Risk-Free r" id="r" value={params.r} step={0.01} min={0} onChange={onChange} unit="%" />
        <NumField label="Volatility σ" id="sigma" value={params.sigma} step={0.01} min={0.01} onChange={onChange} />
        <NumField label="Tenor T" id="T" value={params.T} step={0.25} min={0.1} onChange={onChange} unit="yr" />
        <NumField label="MC Paths" id="nPaths" value={params.nPaths} step={100} min={100} max={20000} onChange={onChange} />
      </div>

      {/* MJD parameters */}
      <div className="flex flex-col gap-3">
        <p className="label-xs border-b border-outline-variant/20 pb-1">Jump Diffusion (MJD)</p>
        <NumField label="Intensity λ" id="lam" value={params.lam} step={0.1} min={0} onChange={onChange} />
        <NumField label="Jump Mean μⱼ" id="muJ" value={params.muJ} step={0.01} onChange={onChange} />
        <NumField label="Jump Vol σⱼ" id="sigmaJ" value={params.sigmaJ} step={0.01} min={0.01} onChange={onChange} />
      </div>

      {/* Option type toggle */}
      <div className="flex flex-col gap-1.5">
        <label className="label-xs">Option Type</label>
        <div className="seg-track w-full">
          {['call', 'put'].map(t => (
            <button
              key={t}
              onClick={() => onChange('optionType', t)}
              className={`seg-pill flex-1 text-center capitalize ${params.optionType === t ? 'active' : ''}`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={onRun}
        disabled={loading}
        className="btn-primary w-full mt-auto flex items-center justify-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {loading ? (
          <>
            <span className="material-symbols-outlined text-base animate-spin-slow">sync</span>
            Running…
          </>
        ) : (
          <>
            <span className="material-symbols-outlined text-base">play_arrow</span>
            Run Simulation
          </>
        )}
      </button>

      {/* Legend */}
      <div className="flex flex-col gap-1.5 pt-2 border-t border-outline-variant/20">
        <div className="flex items-center gap-2 text-xs text-on-surface-variant">
          <span className="w-3 h-0.5 bg-[#b4c5ff] rounded-full" /> GBM paths
        </div>
        <div className="flex items-center gap-2 text-xs text-on-surface-variant">
          <span className="w-3 h-0.5 bg-[#FF6B35] rounded-full" /> MJD paths
        </div>
      </div>
    </aside>
  )
}
