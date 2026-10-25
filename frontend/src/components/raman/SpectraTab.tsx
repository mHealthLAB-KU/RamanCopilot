import { useEffect, useMemo, useState } from 'react'
import type { ChatMessage } from '../../chatStore'
import SpectrumChart, { MAX_OVERLAY, SERIES_COLORS, SpectrumMeta, useSpectra } from './SpectrumChart'

const keyOf = (m: ChatMessage) => m.jsonUrl || m.imageUrl || m.text

export default function SpectraTab({ spectra }: { spectra: ChatMessage[] }) {
  const plottable = spectra.filter(m => m.jsonUrl)
  const [slots, setSlots] = useState<Record<string, number>>({})
  const [normalize, setNormalize] = useState(false)

  const newest = plottable[0]?.jsonUrl
  useEffect(() => {
    if (!newest) return
    setSlots(prev => (newest in prev ? prev : withSlot(prev, newest)))
  }, [newest])

  const selected = plottable.filter(m => m.jsonUrl! in slots)
  const loaded = useSpectra(selected.map(m => m.jsonUrl!))
  const series = useMemo(() => selected.flatMap((m, i) => loaded[i] ? [{
    key: m.jsonUrl!, label: m.text || `Spectrum ${i + 1}`,
    color: SERIES_COLORS[slots[m.jsonUrl!]], data: loaded[i]!,
  }] : []), [selected, loaded, slots])

  const toggle = (url: string) => setSlots(prev => {
    if (url in prev) { const { [url]: _, ...rest } = prev; return rest }
    return Object.keys(prev).length >= MAX_OVERLAY ? prev : withSlot(prev, url)
  })

  if (!spectra.length) return <p className="text-sm text-gray-400 text-center py-10">No spectra in this session.</p>

  return (
    <div className="space-y-3">
      <div className="bg-white border border-gray-200 rounded-xl p-3">
        <div className="flex items-center justify-between mb-1 px-1">
          <span className="text-sm font-semibold text-gray-900">
            {series.length > 1 ? `${series.length} spectra overlaid` : series[0]?.label || 'Spectrum'}
          </span>
          {series.length > 1 && (
            <label className="flex items-center gap-1.5 text-xs text-gray-600 cursor-pointer">
              <input type="checkbox" checked={normalize} onChange={e => setNormalize(e.target.checked)} />
              Normalize
            </label>
          )}
        </div>
        {series.length === 1 && <div className="px-1 mb-1"><SpectrumMeta data={series[0].data} /></div>}
        {series.length
          ? <SpectrumChart series={series} height={250} normalize={normalize && series.length > 1} />
          : <div className="h-[250px] flex items-center justify-center text-sm text-gray-400">
              {selected.length ? 'Loading…' : 'Select spectra below to plot them.'}
            </div>}
        <div className="px-1 mt-1 text-[11px] text-gray-400">Drag to zoom · double-click to reset</div>
      </div>

      <div className="bg-white border border-gray-200 rounded-xl divide-y divide-gray-100 overflow-hidden">
        {spectra.map((m, i) => {
          const url = m.jsonUrl
          const slot = url ? slots[url] : undefined
          const full = slot === undefined && Object.keys(slots).length >= MAX_OVERLAY
          return (
            <div key={keyOf(m) ?? i} className="flex items-center gap-3 px-3 py-2.5 text-sm">
              <button
                type="button"
                disabled={!url || full}
                onClick={() => url && toggle(url)}
                className="flex items-center gap-3 min-w-0 flex-1 text-left disabled:cursor-not-allowed"
                title={!url ? 'Older measurement — image only' : full ? `Up to ${MAX_OVERLAY} spectra can be overlaid` : 'Show / hide in the chart'}
              >
                <span
                  className="w-3.5 h-3.5 rounded-sm border shrink-0"
                  style={slot !== undefined
                    ? { background: SERIES_COLORS[slot], borderColor: SERIES_COLORS[slot] }
                    : { borderColor: '#d1d5db' }}
                />
                <span className={`truncate ${url ? 'text-gray-800' : 'text-gray-400'}`}>{m.text || 'Spectrum'}</span>
              </button>
              <div className="flex gap-2 text-xs text-raman-600 shrink-0">
                {m.csvUrl && <a href={m.csvUrl} download className="hover:underline">CSV</a>}
                {m.imageUrl && <a href={m.imageUrl} download className="hover:underline">PNG</a>}
                {m.zipUrl && <a href={m.zipUrl} download className="hover:underline">ZIP</a>}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/** Return the first free color slot. */
function withSlot(prev: Record<string, number>, url: string): Record<string, number> {
  const used = new Set(Object.values(prev))
  const free = SERIES_COLORS.findIndex((_, i) => !used.has(i))
  return free < 0 ? prev : { ...prev, [url]: free }
}
