import { useEffect, useId, useMemo, useRef, useState } from 'react'

const SATURATION = 65535

export const SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300'] as const
export const MAX_OVERLAY = SERIES_COLORS.length

export interface Spectrum {
  title: string
  shift: number[]
  counts: number[]
  exposureS?: number
  laserMw?: number
  laserPct?: number
  accumulations?: number
  stage?: { x: number; y: number }
  time?: Date
  saturated: boolean
}

const cache = new Map<string, Promise<Spectrum>>()

function load(jsonUrl: string): Promise<Spectrum> {
  if (!cache.has(jsonUrl)) {
    const p = fetch(jsonUrl)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json() })
      .then(parseSpectrum)
    p.catch(() => cache.delete(jsonUrl))
    cache.set(jsonUrl, p)
  }
  return cache.get(jsonUrl)!
}

function parseSpectrum(doc: any): Spectrum {
  const r = doc?.result ?? {}
  const xs: number[] = r['raman_shift_cm-1'] ?? []
  const ys: number[] = r.data ?? []
  if (!xs.length || xs.length !== ys.length) throw new Error('No calibrated spectrum in file')
  const order = xs.map((_, i) => i).sort((a, b) => xs[a] - xs[b])
  const counts = order.map(i => ys[i])
  return {
    title: doc.title ?? '',
    shift: order.map(i => xs[i]),
    counts,
    exposureS: r.exposure_time,
    laserMw: r.laser_power_mw,
    laserPct: r.laser_power_pct,
    accumulations: r.num_accumulations ?? r.num_acc,
    stage: doc.meta?.x != null ? { x: doc.meta.x, y: doc.meta.y } : undefined,
    time: doc.timestamp ? new Date(doc.timestamp) : undefined,
    saturated: counts.some(v => v >= SATURATION),
  }
}

/** Load one measurement JSON. */
export function useSpectrum(jsonUrl?: string) {
  const [state, setState] = useState<{ data?: Spectrum; error?: string }>({})
  useEffect(() => {
    if (!jsonUrl) { setState({ error: 'no data file' }); return }
    let alive = true
    load(jsonUrl).then(
      data => alive && setState({ data }),
      err => alive && setState({ error: String(err?.message ?? err) }),
    )
    return () => { alive = false }
  }, [jsonUrl])
  return state
}

/** Load several measurement JSONs (undefined where not loaded or failed). */
export function useSpectra(jsonUrls: string[]) {
  const [loaded, setLoaded] = useState<Record<string, Spectrum>>({})
  const key = jsonUrls.join('|')
  useEffect(() => {
    let alive = true
    for (const u of jsonUrls) {
      load(u).then(d => alive && setLoaded(prev => (prev[u] ? prev : { ...prev, [u]: d })), () => {})
    }
    return () => { alive = false }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])
  return jsonUrls.map(u => loaded[u])
}

/** Return nice tick values (1, 2, 5 × 10^n). */
function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo
  if (!(span > 0)) return [lo]
  const raw = span / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) ?? raw
  const out: number[] = []
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10))
  return out
}

/** Format a tick label with decimals set by the tick spacing. */
function fmtTick(v: number, step: number): string {
  if (Math.abs(v) >= 10000 && step >= 1000) return `${+(v / 1000).toFixed(1)}k`
  const decimals = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)))
  return v.toFixed(decimals)
}

/** Return the index in sorted xs closest to x. */
function nearest(xs: number[], x: number): number {
  let lo = 0, hi = xs.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs[mid] < x) lo = mid; else hi = mid
  }
  return Math.abs(xs[lo] - x) <= Math.abs(xs[hi] - x) ? lo : hi
}

/** Find a few prominent peaks within the visible range. */
function findPeaks(xs: number[], ys: number[], x0: number, x1: number, max = 5): number[] {
  const W = 6, BASE = 30
  const idx = xs.map((_, i) => i).filter(i => xs[i] >= x0 && xs[i] <= x1)
  if (idx.length < 2 * W + 1) return []
  const vis = idx.map(i => ys[i])
  const range = Math.max(...vis) - Math.min(...vis)
  const cands: { i: number; prom: number }[] = []
  for (const i of idx) {
    let isMax = true
    for (let k = Math.max(0, i - W); k <= Math.min(ys.length - 1, i + W); k++) {
      if (ys[k] > ys[i]) { isMax = false; break }
    }
    if (!isMax) continue
    const left = Math.min(...ys.slice(Math.max(0, i - BASE), i + 1))
    const right = Math.min(...ys.slice(i, i + BASE + 1))
    const prom = ys[i] - Math.max(left, right)
    if (prom > range * 0.08) cands.push({ i, prom })
  }
  return cands.sort((a, b) => b.prom - a.prom).slice(0, max).map(c => c.i)
}

export interface ChartSeries { key: string; label: string; color: string; data: Spectrum }

interface SpectrumChartProps {
  series: ChartSeries[]
  height?: number
  normalize?: boolean
}

const M = { top: 14, right: 12, bottom: 34, left: 46 }

export default function SpectrumChart({ series, height = 220, normalize = false }: SpectrumChartProps) {
  const clipId = 'spc' + useId().replace(/:/g, '')
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  const [zoom, setZoom] = useState<[number, number] | null>(null)
  const [hoverX, setHoverX] = useState<number | null>(null)
  const [drag, setDrag] = useState<[number, number] | null>(null)

  const lines = useMemo(() => series.map(s => {
    if (!normalize) return { ...s, ys: s.data.counts }
    const lo = Math.min(...s.data.counts), hi = Math.max(...s.data.counts)
    return { ...s, ys: s.data.counts.map(v => (hi > lo ? (v - lo) / (hi - lo) : 0)) }
  }), [series, normalize])

  const full: [number, number] = useMemo(() => [
    Math.min(...series.map(s => s.data.shift[0])),
    Math.max(...series.map(s => s.data.shift[s.data.shift.length - 1])),
  ], [series])
  const [x0, x1] = zoom ?? full

  const [y0, y1] = useMemo(() => {
    let lo = Infinity, hi = -Infinity
    for (const l of lines) l.data.shift.forEach((x, i) => {
      if (x >= x0 && x <= x1) { lo = Math.min(lo, l.ys[i]); hi = Math.max(hi, l.ys[i]) }
    })
    if (!isFinite(lo)) return [0, 1]
    const span = hi - lo || 1
    return [lo - span * 0.05, hi + span * 0.16]
  }, [lines, x0, x1])

  const iw = Math.max(0, width - M.left - M.right), ih = height - M.top - M.bottom
  const sx = (x: number) => M.left + ((x - x0) / (x1 - x0)) * iw
  const sy = (y: number) => M.top + ih - ((y - y0) / (y1 - y0)) * ih
  const invX = (px: number) => x0 + ((px - M.left) / iw) * (x1 - x0)

  const paths = lines.map(l => {
    let d = ''
    l.data.shift.forEach((x, i) => {
      if (x < x0 || x > x1) return
      d += `${d ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(l.ys[i]).toFixed(1)}`
    })
    return d
  })

  const peaks = useMemo(() => {
    if (lines.length !== 1) return []
    const l = lines[0]
    const picked: number[] = []
    for (const i of findPeaks(l.data.shift, l.ys, x0, x1)) {
      if (picked.every(j => Math.abs(sx(l.data.shift[j]) - sx(l.data.shift[i])) > 36)) picked.push(i)
    }
    return picked
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lines, x0, x1, width, y0, y1])

  const localX = (e: React.PointerEvent) => e.clientX - (e.currentTarget as Element).getBoundingClientRect().left
  const onMove = (e: React.PointerEvent) => {
    const px = Math.min(Math.max(localX(e), M.left), M.left + iw)
    setHoverX(invX(px))
    if (drag) setDrag([drag[0], px])
  }
  const onUp = () => {
    if (drag && Math.abs(drag[1] - drag[0]) > 6) {
      const a = invX(Math.min(...drag)), b = invX(Math.max(...drag))
      setZoom([a, b])
    }
    setDrag(null)
  }

  const tip = hoverX == null ? null : (() => {
    const rows = lines.map(l => {
      const i = nearest(l.data.shift, hoverX)
      return { key: l.key, label: l.label, color: l.color, x: l.data.shift[i], y: l.ys[i] }
    })
    return { x: rows[0].x, rows }
  })()

  const xt = niceTicks(x0, x1, Math.max(2, Math.floor(iw / 70)))
  const yt = niceTicks(y0, y1, Math.max(2, Math.floor(ih / 40)))

  return (
    <div ref={wrapRef} className="relative select-none" style={{ height }}>
      {width > 0 && (
        <svg
          width={width} height={height}
          className="block touch-none cursor-crosshair"
          onPointerMove={onMove}
          onPointerLeave={() => { setHoverX(null); setDrag(null) }}
          onPointerDown={e => { const px = localX(e); setDrag([px, px]) }}
          onPointerUp={onUp}
          onDoubleClick={() => setZoom(null)}
          role="img"
          aria-label={`Raman spectrum chart: ${series.map(s => s.label).join(', ')}`}
        >
          <defs>
            <clipPath id={clipId}><rect x={M.left} y={M.top} width={iw} height={ih} /></clipPath>
          </defs>

          {yt.map(v => (
            <g key={`y${v}`}>
              <line x1={M.left} x2={M.left + iw} y1={sy(v)} y2={sy(v)} stroke="#eef0f3" />
              <text x={M.left - 6} y={sy(v)} dy="0.32em" textAnchor="end" className="fill-gray-400 text-[10px] font-mono">
                {fmtTick(v, yt.length > 1 ? yt[1] - yt[0] : 1)}
              </text>
            </g>
          ))}
          {xt.map(v => (
            <text key={`x${v}`} x={sx(v)} y={M.top + ih + 14} textAnchor="middle" className="fill-gray-400 text-[10px] font-mono">
              {v}
            </text>
          ))}
          <line x1={M.left} x2={M.left + iw} y1={M.top + ih} y2={M.top + ih} stroke="#d1d5db" />
          <text x={M.left + iw / 2} y={height - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">
            Raman shift (cm⁻¹)
          </text>

          <g clipPath={`url(#${clipId})`}>
            {paths.map((d, k) => (
              <path key={lines[k].key} d={d} fill="none" stroke={lines[k].color} strokeWidth={1.75} strokeLinejoin="round" />
            ))}
            {peaks.map(i => {
              const l = lines[0], x = sx(l.data.shift[i]), y = sy(l.ys[i])
              return (
                <g key={`p${i}`}>
                  <line x1={x} x2={x} y1={y - 3} y2={y - 9} stroke="#9ca3af" />
                  <text x={x} y={y - 12} textAnchor="middle" className="fill-gray-700 text-[10px] font-mono">
                    {l.data.shift[i].toFixed(0)}
                  </text>
                </g>
              )
            })}
            {tip && (
              <>
                <line x1={sx(tip.x)} x2={sx(tip.x)} y1={M.top} y2={M.top + ih} stroke="#6b7280" strokeDasharray="3 3" />
                {tip.rows.map(r => (
                  <circle key={r.key} cx={sx(r.x)} cy={sy(r.y)} r={4} fill={r.color} stroke="#fff" strokeWidth={2} />
                ))}
              </>
            )}
            {drag && Math.abs(drag[1] - drag[0]) > 2 && (
              <rect x={Math.min(...drag)} y={M.top} width={Math.abs(drag[1] - drag[0])} height={ih} fill="#2f49d9" opacity={0.08} />
            )}
          </g>
        </svg>
      )}

      {tip && (
        <div
          className="absolute pointer-events-none bg-white/95 border border-gray-200 rounded-lg shadow-sm px-2.5 py-1.5 text-xs"
          style={{
            top: M.top,
            ...(sx(tip.x) > width / 2 ? { right: width - sx(tip.x) + 10 } : { left: sx(tip.x) + 10 }),
          }}
        >
          <div className="text-gray-500 font-mono">{tip.x.toFixed(1)} cm⁻¹</div>
          {tip.rows.map(r => (
            <div key={r.key} className="flex items-center gap-1.5 whitespace-nowrap">
              <span className="inline-block w-3 h-0.5 rounded" style={{ background: r.color }} />
              <span className="font-mono font-semibold text-gray-900">{normalize ? r.y.toFixed(3) : r.y.toFixed(0)}</span>
              {tip.rows.length > 1 && <span className="text-gray-500 truncate max-w-[140px]">{r.label}</span>}
            </div>
          ))}
        </div>
      )}

      {zoom && (
        <button
          onClick={() => setZoom(null)}
          className="absolute top-1 right-2 text-[11px] text-raman-600 bg-white/90 border border-gray-200 rounded px-1.5 py-0.5 hover:bg-white"
        >
          Reset zoom
        </button>
      )}
    </div>
  )
}

export function SpectrumMeta({ data }: { data: Spectrum }) {
  const chips = [
    data.exposureS != null && `${data.exposureS} s`,
    data.accumulations != null && data.accumulations > 1 && `×${data.accumulations}`,
    data.laserMw != null ? `${data.laserMw} mW laser` : data.laserPct != null && `${data.laserPct}% laser`,
    data.stage && `X ${data.stage.x.toFixed(3)} · Y ${data.stage.y.toFixed(3)} mm`,
    data.time && data.time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
  ].filter(Boolean) as string[]
  return (
    <div className="flex flex-wrap gap-1.5">
      {data.saturated && (
        <span className="text-[11px] px-1.5 py-0.5 rounded bg-red-50 text-red-700 border border-red-200" title="Some points hit the detector limit (65535) — peak tops are clipped.">
          ⚠ Saturated
        </span>
      )}
      {chips.map(c => (
        <span key={c} className="text-[11px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 font-mono">{c}</span>
      ))}
    </div>
  )
}
