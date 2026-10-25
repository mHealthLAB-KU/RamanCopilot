import { useState, ReactNode } from 'react'
import CameraView, { ObjectiveSelector } from './raman/CameraView'
import { SpectrumCard, toolLabel } from './ChatThread'
import SpectraTab from './raman/SpectraTab'
import type { ChatMessage } from '../chatStore'
import type { HwStatus } from '../hw'

const TABS = ['Live view', 'Spectra', 'Log', 'Parameters'] as const
type Tab = typeof TABS[number]

interface RightPanelProps {
  className?: string
  messages: ChatMessage[]
  status: HwStatus
  onMoved: () => void
  parameters: ReactNode
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className="font-mono text-sm text-gray-900 mt-0.5">{value}</div>
    </div>
  )
}

export default function RightPanel({
  className = 'flex', messages, status, onMoved, parameters,
}: RightPanelProps) {
  const [tab, setTab] = useState<Tab>('Live view')
  const spectra = messages.filter(m => m.kind === 'spectrum').reverse()
  const log = messages
    .filter(m => m.kind === 'progress')
    .flatMap(m => (m.steps || []).map((s, j) => ({ node: m.stepNodes?.[j], text: s })))
  const fmt = (v?: number | null) => (v == null ? '—' : v.toFixed(3))
  const { stage, ccd, laser, optics } = status
  const objective = optics ? { current: optics.objective, available: optics.available } : null

  return (
    <aside className={`${className} flex-col w-full lg:w-[40%] lg:min-w-[360px] lg:max-w-[540px] shrink-0 lg:border-l border-gray-200 bg-[#f4f5f7] min-h-0`}>
      <div className="p-4 pb-0">
        <div className="grid grid-cols-4 p-1 bg-gray-200/60 rounded-xl text-sm">
          {TABS.map(t => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`py-1.5 rounded-lg whitespace-nowrap ${tab === t ? 'bg-white text-gray-900 font-medium shadow-sm' : 'text-gray-500 hover:text-gray-800'}`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <div className={tab === 'Live view' ? 'space-y-4' : 'hidden'}>
          <div>
            <div className="h-[340px]">
              <CameraView aim={optics?.aim ?? null} onMoved={onMoved} />
            </div>
            <div className="mt-2 flex items-center justify-between gap-3">
              <span className="text-xs text-gray-500">Click to move stage · orange ring = laser spot</span>
              <ObjectiveSelector objective={objective} onChanged={onMoved} />
            </div>
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-gray-900">Last spectrum</span>
              {spectra[0]?.text && <span className="text-xs text-gray-500 truncate ml-3">{spectra[0].text}</span>}
            </div>
            {spectra[0]
              ? <SpectrumCard key={spectra[0].jsonUrl || spectra[0].imageUrl} msg={spectra[0]} height={170} bare />
              : <div className="h-32 flex items-center justify-center text-sm text-gray-400">No spectrum yet</div>}
          </div>

          <div className="bg-white border border-gray-200 rounded-xl p-4 grid grid-cols-3 gap-y-4 gap-x-3">
            <Readout label="Stage X (mm)" value={fmt(stage?.x)} />
            <Readout label="Stage Y (mm)" value={fmt(stage?.y)} />
            <Readout label="Focus Z (mm)" value={fmt(stage?.z)} />
            <Readout label="Exposure (s)" value={String(ccd?.exposure_s ?? '—')} />
            <Readout label="Accumulations" value={String(ccd?.num_acc ?? '—')} />
            <Readout label="Laser power (mW)"
                     value={laser?.power_mw != null ? `${+laser.power_mw.toPrecision(3)} (${+(laser.power_pct ?? 0).toFixed(1)}%)` : '—'} />
          </div>
        </div>

        {tab === 'Spectra' && <SpectraTab spectra={spectra} />}

        {tab === 'Log' && (
          log.length
            ? (
              <div className="bg-white border border-gray-200 rounded-xl divide-y divide-gray-100 text-xs">
                {log.map((l, i) => (
                  <div key={i} className="px-4 py-2.5">
                    {l.node && <div className="font-medium text-gray-900">{toolLabel(l.node)}</div>}
                    <div className="font-mono text-gray-600 break-words">{l.text}</div>
                  </div>
                ))}
              </div>
            )
            : <p className="text-sm text-gray-400 text-center py-10">No agent activity yet.</p>
        )}

        {tab === 'Parameters' && parameters}
      </div>
    </aside>
  )
}
