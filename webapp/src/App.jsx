import { useState, useCallback } from 'react'
import NavBar from './components/NavBar.jsx'
import SimulationParams from './components/SimulationParams.jsx'
import PricingTab from './tabs/PricingTab.jsx'
import GreeksTab from './tabs/GreeksTab.jsx'
import PathsTab from './tabs/PathsTab.jsx'
import VolatilityTab from './tabs/VolatilityTab.jsx'
import { runSimulation, fetchGreeks } from './api.js'

// Default simulation parameters
const DEFAULT_PARAMS = {
  S0: 100,
  K: 100,
  r: 0.05,
  sigma: 0.20,
  T: 1.0,
  nPaths: 1000,
  lam: 1.0,
  muJ: -0.10,
  sigmaJ: 0.15,
  optionType: 'call',
  nSamplePaths: 30,
}

export default function App() {
  const [activeTab, setActiveTab] = useState('pricing')
  const [params, setParams] = useState(DEFAULT_PARAMS)

  // Simulation results — shared across all tabs
  const [pricingResults, setPricingResults] = useState(null)
  const [greeksResults, setGreeksResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleParamChange = useCallback((key, value) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }, [])

  const handleRun = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // Run pricing simulation + greeks in parallel
      const [simData, greekData] = await Promise.all([
        runSimulation(params),
        fetchGreeks(params),
      ])
      setPricingResults(simData)
      setGreeksResults(greekData)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [params])

  const tabProps = { params, loading, error }

  return (
    <div className="flex flex-col h-screen bg-surface-lowest overflow-hidden">
      <NavBar activeTab={activeTab} onTabChange={setActiveTab} />

      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar — simulation parameters */}
        <SimulationParams
          params={params}
          onChange={handleParamChange}
          onRun={handleRun}
          loading={loading}
        />

        {/* Main content area */}
        <main className="flex-1 overflow-y-auto p-6">
          {error && (
            <div className="mb-4 px-4 py-3 rounded-lg bg-error-container/20 border border-error/30 text-error text-sm font-mono">
              <span className="material-symbols-outlined text-base mr-2">error</span>
              {error}
            </div>
          )}

          {activeTab === 'pricing' && (
            <PricingTab {...tabProps} results={pricingResults} />
          )}
          {activeTab === 'greeks' && (
            <GreeksTab {...tabProps} results={greeksResults} />
          )}
          {activeTab === 'paths' && (
            <PathsTab {...tabProps} results={pricingResults} />
          )}
          {activeTab === 'volatility' && (
            <VolatilityTab {...tabProps} />
          )}
        </main>
      </div>
    </div>
  )
}
