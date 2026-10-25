import { useCallback, useEffect, useRef, useState } from 'react'
import axios from 'axios'
import { Activity, Crosshair, Grid2x2, ScanLine, Loader2 } from 'lucide-react'
import { OFFLINE, fetchHwStatus, type Device, type HwStatus } from '../hw'

export type HardwareStatus = HwStatus

const POLL_MS = 1000

/** Poll the hardware status (/api/hw/status). */
export function useHardwareStatus(): [HwStatus, () => void] {
  const [status, setStatus] = useState<HwStatus>(OFFLINE)

  const poll = useCallback(async () => {
    try {
      const next = await fetchHwStatus()
      setStatus(prev => ({
        ...next,
        ccd: next.ccd && next.ccd.temp_status == null && prev.ccd
          ? { ...next.ccd, temp_status: prev.ccd.temp_status, temperature: next.ccd.temperature ?? prev.ccd.temperature }
          : next.ccd,
      }))
    } catch {
      setStatus(OFFLINE)
    }
  }, [])

  useEffect(() => {
    poll()
    const id = window.setInterval(poll, POLL_MS)
    return () => window.clearInterval(id)
  }, [poll])

  return [status, poll]
}

type Tone = 'ok' | 'warn' | 'off'

function ccdLabel(s: HardwareStatus): [string, Tone] {
  if (!s.connected.ccd) return ['Offline', 'off']
  switch (s.ccd?.temp_status) {
    case 'STABILIZED':     return ['Cooled', 'ok']
    case 'NOT_REACHED':    return ['Cooling', 'warn']
    case 'NOT_STABILIZED': return ['Stabilizing', 'warn']
    case 'DRIFT':          return ['Drift', 'warn']
    case 'OFF':            return ['Cooler off', 'warn']
    default:               return ['Connected', 'ok']
  }
}

const ICON: Record<Device, typeof Activity> = { laser: Activity, ccd: ScanLine, camera: Crosshair, stage: Grid2x2 }

function deviceRows(s: HardwareStatus): { key: Device; name: string; label: string; tone: Tone; icon: typeof Activity }[] {
  const on = (d: Device, yes: string): [string, Tone] => s.connected[d] ? [yes, 'ok'] : ['Offline', 'off']
  const [ccdText, ccdTone] = ccdLabel(s)
  const rows: [Device, string, [string, Tone]][] = [
    ['laser', 'Laser', on('laser', 'Ready')],
    ['ccd', 'Spectrometer', [ccdText, ccdTone]],
    ['camera', 'Camera', on('camera', 'Live')],
    ['stage', 'Stage', on('stage', 'Connected')],
  ]
  return rows.map(([key, name, [label, tone]]) => ({ key, name, label, tone, icon: ICON[key] }))
}

const DOT: Record<Tone, string> = { ok: 'bg-green-600', warn: 'bg-amber-500', off: 'bg-gray-300' }
const TEXT: Record<Tone, string> = { ok: 'text-green-700', warn: 'text-amber-700', off: 'text-gray-400' }

/** Hardware status chips in the header (dots only when compact). */
export function StatusPills({ status, compact = false }: { status: HardwareStatus; compact?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      {deviceRows(status).map(d => (
        <span
          key={d.key}
          title={`${d.name}: ${d.label}`}
          className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-gray-200 bg-white text-xs text-gray-800 whitespace-nowrap"
        >
          <span className={`w-1.5 h-1.5 rounded-full ${DOT[d.tone]}`} />
          <d.icon className={`w-3.5 h-3.5 text-gray-600 ${compact ? 'xl:hidden' : 'md:hidden'}`} />
          <span className={compact ? 'hidden xl:inline' : 'hidden md:inline'}>{d.name}</span>
          {!compact && <span className={`hidden lg:inline font-medium ${TEXT[d.tone]}`}>{d.label}</span>}
        </span>
      ))}
    </div>
  )
}

const CONNECT_URL: Record<Device, string> = {
  ccd: '/api/ccd/connect',
  camera: '/api/camera/connect',
  stage: '/api/stage/connect',
  laser: '/api/laser/connect',
}

const CARDS: { key: Device; title: string; icon: typeof Activity; detail: (s: HardwareStatus) => string }[] = [
  {
    key: 'ccd', title: 'RamanCCD', icon: ScanLine,
    detail: s => s.ccd?.temperature != null ? `Andor EMCCD · ${s.ccd.temperature} °C` : 'Andor EMCCD',
  },
  { key: 'camera', title: 'Camera', icon: Crosshair, detail: () => 'TUCam live view' },
  { key: 'stage', title: 'Stage', icon: Grid2x2, detail: () => 'Tango XYZ stage' },
  { key: 'laser', title: 'Laser', icon: Activity, detail: () => '532 nm excitation' },
]

let autoConnectDone = false
const AUTO_ORDER: Device[] = ['camera', 'stage', 'laser', 'ccd']

/** Hardware cards on the home screen that connect disconnected instruments and offer retries. */
export function HardwareCards({ status, onChanged }: { status: HardwareStatus; onChanged: () => void }) {
  const [busy, setBusy] = useState<Device | null>(null)
  const [errors, setErrors] = useState<Partial<Record<Device, string>>>({})
  const [laserPort, setLaserPort] = useState('COM4')
  const rows = Object.fromEntries(deviceRows(status).map(r => [r.key, r]))

  const connect = useCallback(async (d: Device, port: string) => {
    setBusy(d)
    setErrors(e => ({ ...e, [d]: undefined }))
    try {
      await axios.post(CONNECT_URL[d], d === 'laser' ? { port } : {})
    } catch (e: any) {
      setErrors(prev => ({ ...prev, [d]: e?.response?.data?.detail ?? e?.message ?? 'Connection failed' }))
    } finally {
      setBusy(null)
      onChanged()
    }
  }, [onChanged])

  const statusRef = useRef(status)
  statusRef.current = status
  useEffect(() => {
    if (autoConnectDone || !status.online) return
    autoConnectDone = true
    ;(async () => {
      for (const d of AUTO_ORDER) {
        if (!statusRef.current.connected[d]) await connect(d, 'COM4')
      }
    })()
  }, [status.online, connect])

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {CARDS.map(({ key, title, icon: Icon, detail }) => {
        const row = rows[key]
        return (
          <div key={key} className="bg-white border border-gray-200 rounded-xl p-5 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <Icon className="w-6 h-6 text-raman-500" />
              <span className={`inline-flex items-center gap-1.5 text-xs font-medium ${TEXT[row.tone]}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${DOT[row.tone]}`} />
                {row.label}
              </span>
            </div>
            <div className="font-semibold text-gray-900">{title}</div>
            <div className="text-sm text-gray-500">{detail(status)}</div>
            {errors[key] && <div className="text-xs text-red-600 truncate" title={errors[key]}>{errors[key]}</div>}
            {status.online && !status.connected[key] && (
              <div className="mt-1 flex items-center gap-2">
                {key === 'laser' && (
                  <input
                    value={laserPort}
                    onChange={e => setLaserPort(e.target.value.toUpperCase())}
                    className="w-20 px-2 py-1.5 text-xs font-mono border border-gray-200 rounded-lg focus:outline-none focus:border-raman-300"
                    aria-label="Laser port"
                  />
                )}
                <button
                  onClick={() => connect(key, laserPort)}
                  disabled={busy !== null}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium border border-gray-200 rounded-lg hover:bg-gray-50 disabled:opacity-50"
                >
                  {busy === key && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
                  {busy === key ? 'Connecting…' : 'Connect'}
                </button>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
