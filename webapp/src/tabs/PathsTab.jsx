import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Grid, Text, Line } from '@react-three/drei'
import * as THREE from 'three'

// ── Helper: normalise a path array to [-1, 1] scene coordinates ──────────────

function normalisePaths(paths, t, S0) {
  if (!paths?.length || !t?.length) return []

  const allPrices = paths.flat()
  const pMin = Math.min(...allPrices)
  const pMax = Math.max(...allPrices)
  const pRange = pMax - pMin || 1

  const tMax = t[t.length - 1] || 1

  return paths.map((path, pathIdx) =>
    path.map((price, ti) => {
      const x = (t[ti] / tMax) * 4 - 2       // time → [-2, 2]
      const y = ((price - pMin) / pRange) * 2  // price → [0, 2]
      const z = (pathIdx / paths.length) * 3 - 1.5  // path spread → [-1.5, 1.5]
      return [x, y, z]
    })
  )
}

// ── Single animated path ──────────────────────────────────────────────────────

function PathLine({ points, color, opacity = 0.7 }) {
  if (!points?.length) return null
  return (
    <Line
      points={points}
      color={color}
      lineWidth={1}
      transparent
      opacity={opacity}
    />
  )
}

// ── Strike plane (horizontal slice at K price) ────────────────────────────────

function StrikePlane({ yNorm }) {
  return (
    <mesh position={[0, yNorm, 0]} rotation={[-Math.PI / 2, 0, 0]}>
      <planeGeometry args={[4, 3]} />
      <meshBasicMaterial
        color="#4edea3"
        transparent
        opacity={0.04}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

// ── Scene (inside Canvas) ─────────────────────────────────────────────────────

function Scene({ gbmPaths, mjdPaths, t, S0, K, showGbm, showMjd }) {
  const groupRef = useRef()

  // Slow auto-rotation when not being orbited
  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.1) * 0.15
    }
  })

  const gbmNorm = useMemo(
    () => normalisePaths(gbmPaths, t, S0),
    [gbmPaths, t, S0]
  )
  const mjdNorm = useMemo(
    () => normalisePaths(mjdPaths, t, S0),
    [mjdPaths, t, S0]
  )

  // Normalise K to scene y coordinate
  const allPrices = [...(gbmPaths?.flat() || []), ...(mjdPaths?.flat() || [])]
  const pMin = allPrices.length ? Math.min(...allPrices) : 0
  const pMax = allPrices.length ? Math.max(...allPrices) : 1
  const pRange = pMax - pMin || 1
  const kNorm = allPrices.length ? ((K - pMin) / pRange) * 2 : 1

  return (
    <group ref={groupRef}>
      {/* Floor grid */}
      <Grid
        args={[4, 3]}
        position={[0, -0.05, 0]}
        cellSize={0.5}
        cellColor="rgba(67,70,85,0.4)"
        sectionColor="rgba(67,70,85,0.6)"
        fadeDistance={12}
        infiniteGrid={false}
      />

      {/* Strike price plane */}
      {allPrices.length > 0 && <StrikePlane yNorm={kNorm} />}

      {/* GBM paths */}
      {showGbm && gbmNorm.map((pts, i) => (
        <PathLine
          key={`gbm-${i}`}
          points={pts}
          color="#b4c5ff"
          opacity={0.5}
        />
      ))}

      {/* MJD paths */}
      {showMjd && mjdNorm.map((pts, i) => (
        <PathLine
          key={`mjd-${i}`}
          points={pts}
          color="#FF6B35"
          opacity={0.5}
        />
      ))}

      {/* Axis labels */}
      <Text position={[2.4, -0.2, 0]} fontSize={0.12} color="#c3c6d7" anchorX="center">
        T=1
      </Text>
      <Text position={[-2.4, -0.2, 0]} fontSize={0.12} color="#c3c6d7" anchorX="center">
        T=0
      </Text>
    </group>
  )
}

// ── Stats overlay ─────────────────────────────────────────────────────────────

function StatsOverlay({ gbmPaths, mjdPaths, S0 }) {
  if (!gbmPaths?.length) return null

  const gbmTerminal = gbmPaths.map(p => p[p.length - 1])
  const mjdTerminal = mjdPaths.map(p => p[p.length - 1])

  const avgGbm = gbmTerminal.reduce((a, b) => a + b, 0) / gbmTerminal.length
  const avgMjd = mjdTerminal.reduce((a, b) => a + b, 0) / mjdTerminal.length
  const pItmGbm = gbmTerminal.filter(p => p > S0).length / gbmTerminal.length

  return (
    <div className="absolute top-3 right-3 glass-card p-3 text-xs font-mono space-y-1.5">
      <p className="label-xs mb-2">Simulation Stats</p>
      <div className="flex justify-between gap-4">
        <span className="text-on-surface-variant">Active Paths</span>
        <span className="text-primary">{gbmPaths.length + mjdPaths.length}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-on-surface-variant">GBM Terminal Avg</span>
        <span className="text-[#b4c5ff]">${avgGbm.toFixed(2)}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-on-surface-variant">MJD Terminal Avg</span>
        <span className="text-[#FF6B35]">${avgMjd.toFixed(2)}</span>
      </div>
      <div className="flex justify-between gap-4">
        <span className="text-on-surface-variant">P(ITM) GBM</span>
        <span className="text-tertiary">{(pItmGbm * 100).toFixed(1)}%</span>
      </div>
    </div>
  )
}

// ── Main tab ──────────────────────────────────────────────────────────────────

export default function PathsTab({ results, params, loading }) {
  const [showGbm, setShowGbm] = useState(true)
  const [showMjd, setShowMjd] = useState(true)

  const sp = results?.sample_paths
  const gbmPaths = sp?.gbm || []
  const mjdPaths = sp?.mjd || []
  const t = sp?.t || []

  const hasData = gbmPaths.length > 0

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="font-headline font-bold text-2xl text-on-surface">3D Path Visualization</h1>
          <p className="text-sm text-on-surface-variant mt-1">
            Monte Carlo price paths in 3D space — time × price × path index.
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showGbm}
              onChange={e => setShowGbm(e.target.checked)}
              className="rounded"
            />
            <span className="w-3 h-0.5 bg-[#b4c5ff] rounded-full inline-block" />
            <span className="text-on-surface-variant">GBM</span>
          </label>
          <label className="flex items-center gap-2 text-sm cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showMjd}
              onChange={e => setShowMjd(e.target.checked)}
              className="rounded"
            />
            <span className="w-3 h-0.5 bg-[#FF6B35] rounded-full inline-block" />
            <span className="text-on-surface-variant">MJD</span>
          </label>
        </div>
      </div>

      {/* Three.js canvas */}
      <div className="glass-card relative flex-1 min-h-[500px] overflow-hidden">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface-lowest/60 z-10 rounded-xl">
            <div className="flex items-center gap-2 text-sm text-on-surface-variant">
              <span className="material-symbols-outlined animate-spin-slow text-primary">sync</span>
              Simulating paths…
            </div>
          </div>
        )}

        {!hasData && !loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-on-surface-variant/50 z-10">
            <span className="material-symbols-outlined text-5xl">view_in_ar</span>
            <p className="text-sm">Run a simulation to render 3D paths</p>
          </div>
        )}

        <Canvas
          camera={{ position: [3, 2, 4], fov: 50 }}
          gl={{ antialias: true, alpha: true }}
          style={{ background: 'transparent' }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[5, 5, 5]} intensity={0.8} />

          {hasData && (
            <Scene
              gbmPaths={showGbm ? gbmPaths : []}
              mjdPaths={showMjd ? mjdPaths : []}
              t={t}
              S0={params?.S0 || 100}
              K={params?.K || 100}
              showGbm={showGbm}
              showMjd={showMjd}
            />
          )}

          <OrbitControls
            enablePan={false}
            minDistance={2}
            maxDistance={10}
            autoRotate={!hasData}
            autoRotateSpeed={0.5}
          />
        </Canvas>

        {hasData && (
          <StatsOverlay
            gbmPaths={gbmPaths}
            mjdPaths={mjdPaths}
            S0={params?.S0 || 100}
          />
        )}

        {/* Canvas hints */}
        <div className="absolute bottom-3 left-3 text-xs text-on-surface-variant/40 font-mono space-y-0.5">
          <p>Drag to orbit · Scroll to zoom</p>
          <p className="text-tertiary/60">Green plane = strike (K)</p>
        </div>
      </div>
    </div>
  )
}
