import { useEffect, useRef, useState, type ReactNode } from 'react'
import { applyHwSettings, type HwStatus } from '../../hw'
import { ObjectiveSelector } from './CameraView'

const LBL = 'block text-xs text-gray-500 mb-0.5 truncate'
const INP = 'w-full px-2 py-1 text-xs border border-gray-300 rounded focus:ring-1 focus:ring-raman-500 bg-white disabled:opacity-40 disabled:cursor-not-allowed'

const OWN_CHANGE_MS = 3000
const FLASH_MS = 1800

type Commit = (key: string, changes: Record<string, unknown>) => void

interface FieldState { errors: Record<string, string>; pending: Record<string, boolean>; ownAt: Record<string, number> }

/** Return true briefly when the value was changed externally (agent or instrument). */
function useExternalFlash(live: unknown, ownAt: number | undefined): boolean {
  const prev = useRef(live)
  const [flash, setFlash] = useState(false)
  useEffect(() => {
    if (prev.current !== undefined && prev.current !== live && Date.now() - (ownAt ?? 0) > OWN_CHANGE_MS) {
      setFlash(true)
      const t = window.setTimeout(() => setFlash(false), FLASH_MS)
      prev.current = live
      return () => window.clearTimeout(t)
    }
    prev.current = live
  }, [live, ownAt])
  return flash
}

function Field({ label, fkey, st, live, children, wide = false }: {
  label: string; fkey: string; st: FieldState; live: unknown; children: ReactNode; wide?: boolean
}) {
  const flash = useExternalFlash(live, st.ownAt[fkey])
  return (
    <label className={`block min-w-0 rounded transition-colors duration-700 ${wide ? 'col-span-2' : ''} ${flash ? 'bg-amber-100' : ''}`}
           title={flash ? 'Changed outside this panel (agent or instrument)' : undefined}>
      <span className={LBL} title={label}>
        {label}{st.pending[fkey] && <span className="ml-1 text-gray-400">…</span>}
      </span>
      {children}
      {st.errors[fkey] && <span className="block mt-0.5 text-[11px] text-red-600 leading-snug">{st.errors[fkey]}</span>}
    </label>
  )
}

/** Format mW with 3 significant digits. */
const fmtMw = (v: number) => String(+v.toPrecision(4))

/** Parse a power input: '4.5' -> 4.5 mW, '10%' -> full_mw × 0.1 mW. */
function parsePower(s: string, fullMw?: number): number {
  const t = s.trim()
  if (!t.endsWith('%')) return Number(t)
  const pct = Number(t.slice(0, -1).trim())
  return fullMw && t.length > 1 ? pct * fullMw / 100 : NaN
}

/** Numeric input sent on Enter or blur (Esc cancels); out-of-range values are rejected locally. */
function NumberInput({ live, step, min, range, disabled, onCommit, onReject, fmt, parse }: {
  live: number | null | undefined; step: number; min?: number; range?: [number, number]; disabled?: boolean
  onCommit: (v: number) => void; onReject?: (msg: string) => void
  fmt?: (v: number) => string
  parse?: (s: string) => number
}) {
  const [draft, setDraft] = useState<string | null>(null)
  const show = fmt ?? String
  const commit = () => {
    if (draft === null) return
    const v = (parse ?? Number)(draft)
    setDraft(null)
    if (draft.trim() === '' || !Number.isFinite(v) || v === live) return
    if (range && (v < range[0] || v > range[1])) {
      onReject?.(`Allowed: ${show(range[0])} – ${show(range[1])}`)
      return
    }
    onCommit(v)
  }
  return (
    <input
      type={parse ? 'text' : 'number'} inputMode={parse ? 'decimal' : undefined}
      className={INP} step={step} min={min ?? range?.[0]} max={range?.[1]} disabled={disabled}
      value={draft ?? (live == null ? '' : show(live))}
      onChange={e => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={e => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
        if (e.key === 'Escape') { setDraft(null); (e.target as HTMLInputElement).blur() }
      }}
    />
  )
}

function Select({ live, options, disabled, onCommit }: {
  live: string | number | null | undefined; options: [string | number, string][]; disabled?: boolean
  onCommit: (v: string) => void
}) {
  const opts = live != null && !options.some(([v]) => String(v) === String(live))
    ? [...options, [live, String(live)] as [string | number, string]] : options
  return (
    <select className={INP} disabled={disabled || live == null} value={live ?? ''} onChange={e => onCommit(e.target.value)}>
      {live == null && <option value="">—</option>}
      {opts.map(([v, l]) => <option key={String(v)} value={String(v)}>{l}</option>)}
    </select>
  )
}

function Toggle({ live, onCommit, confirmOff }: { live: boolean | null | undefined; onCommit: (v: boolean) => void; confirmOff?: string }) {
  return (
    <button
      type="button"
      disabled={live == null}
      onClick={() => {
        if (live && confirmOff && !window.confirm(confirmOff)) return
        onCommit(!live)
      }}
      className={`w-full px-2 py-1 text-xs rounded border text-left disabled:opacity-40 ${
        live ? 'bg-raman-50 border-raman-300 text-raman-700' : 'bg-white border-gray-300 text-gray-600'}`}
    >
      {live == null ? '—' : live ? 'On' : 'Off'}
    </button>
  )
}

function ReadOnly({ value, tone = 'text-gray-900' }: { value: string; tone?: string }) {
  return <div className={`px-2 py-1 text-xs font-mono border border-transparent ${tone} truncate`}>{value}</div>
}

function Section({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="text-xs font-semibold text-gray-900 mb-2">
        {title}{note && <span className="ml-2 font-normal text-gray-400">{note}</span>}
      </h3>
      <div className="grid grid-cols-2 gap-x-3 gap-y-2">{children}</div>
    </section>
  )
}

const TEMP_LABEL: Record<string, [string, string]> = {
  STABILIZED: ['Stabilized', 'text-green-700'],
  NOT_REACHED: ['Cooling', 'text-amber-700'],
  NOT_STABILIZED: ['Stabilizing', 'text-amber-700'],
  DRIFT: ['Drift', 'text-amber-700'],
  OFF: ['Cooler off', 'text-gray-500'],
}

const fmt = (v: number | null | undefined, d = 3) => (v == null ? '—' : v.toFixed(d))

interface ParameterPanelProps {
  status: HwStatus
  onChanged: () => void
  isAcquiring: boolean
  onAcquire: () => void
}

export default function ParameterPanel({ status, onChanged, isAcquiring, onAcquire }: ParameterPanelProps) {
  const [st, setSt] = useState<FieldState>({ errors: {}, pending: {}, ownAt: {} })

  const commit: Commit = async (key, changes) => {
    setSt(s => ({ errors: { ...s.errors, [key]: '' }, pending: { ...s.pending, [key]: true }, ownAt: { ...s.ownAt, [key]: Date.now() } }))
    let error = ''
    try {
      await applyHwSettings(changes)
    } catch (e: any) {
      error = e.message
    }
    setSt(s => ({ ...s, errors: { ...s.errors, [key]: error }, pending: { ...s.pending, [key]: false } }))
    onChanged()
  }
  const ccdSet = (key: string, v: Record<string, unknown>) => commit(`ccd.${key}`, { ccd: v })
  const reject = (key: string) => (msg: string) => setSt(s => ({ ...s, errors: { ...s.errors, [key]: msg } }))

  const { ccd, laser, camera, stage, optics } = status
  const accum = ccd?.acq_mode === 'accumulate' || ccd?.acq_mode === 'kinetic'
  const kinetic = ccd?.acq_mode === 'kinetic'
  const track = ccd?.read_mode === 'single_track'
  const [tempLabel, tempTone] = TEMP_LABEL[ccd?.temp_status ?? ''] ?? [ccd?.temp_status ?? '—', 'text-gray-500']

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 text-xs space-y-5">
      <section>
        <h3 className="text-xs font-semibold text-gray-900 mb-2">Objective</h3>
        <ObjectiveSelector objective={optics ? { current: optics.objective, available: optics.available } : null} onChanged={onChanged} />
      </section>

      <Section title="Acquisition" note={ccd ? undefined : 'CCD offline'}>
        <Field label="Mode" fkey="ccd.acq_mode" st={st} live={ccd?.acq_mode}>
          <Select live={ccd?.acq_mode} onCommit={v => ccdSet('acq_mode', { acq_mode: v })}
                  options={[['single', 'Single'], ['accumulate', 'Accumulate'], ['kinetic', 'Kinetic']]} />
        </Field>
        <Field label="Readout mode" fkey="ccd.read_mode" st={st} live={ccd?.read_mode}>
          <Select live={ccd ? ccd.read_mode : null} onCommit={v => ccdSet('read_mode', { read_mode: v })}
                  options={[['fvb', 'FVB'], ['single_track', 'Single track'], ['image', 'Image']]} />
        </Field>
        <Field label="Exposure (s)" fkey="ccd.exposure_s" st={st} live={ccd?.exposure_s}>
          <NumberInput live={ccd?.exposure_s} step={0.1} min={0.001} disabled={!ccd}
                       onCommit={v => ccdSet('exposure_s', { exposure_s: v })} />
        </Field>
        <Field label="Accumulations" fkey="ccd.num_acc" st={st} live={ccd?.num_acc}>
          <NumberInput live={ccd?.num_acc} step={1} min={1} disabled={!accum}
                       onCommit={v => ccdSet('num_acc', { num_acc: v })} />
        </Field>
        <Field label="Kinetic count" fkey="ccd.num_kin" st={st} live={ccd?.num_kin}>
          <NumberInput live={ccd?.num_kin} step={1} min={1} disabled={!kinetic}
                       onCommit={v => ccdSet('num_kin', { num_kin: v })} />
        </Field>
        <Field label="Kinetic cycle (s)" fkey="ccd.kinetic_cycle_s" st={st} live={ccd?.kinetic_cycle_s}>
          <NumberInput live={ccd?.kinetic_cycle_s} step={0.1} min={0} disabled={!kinetic}
                       onCommit={v => ccdSet('kinetic_cycle_s', { kinetic_cycle_s: v })} />
        </Field>
        {track && <>
          <Field label="Track center (row)" fkey="ccd.track_center" st={st} live={ccd?.track_center}>
            <NumberInput live={ccd?.track_center} step={1} min={1} onCommit={v => ccdSet('track_center', { track_center: v })} />
          </Field>
          <Field label="Track width (rows)" fkey="ccd.track_width" st={st} live={ccd?.track_width}>
            <NumberInput live={ccd?.track_width} step={1} min={1} onCommit={v => ccdSet('track_width', { track_width: v })} />
          </Field>
        </>}
        <Field label="Horizontal binning" fkey="ccd.hbin" st={st} live={ccd?.hbin}>
          <NumberInput live={ccd?.hbin} step={1} min={1} disabled={!ccd} onCommit={v => ccdSet('hbin', { hbin: v })} />
        </Field>
      </Section>

      <Section title="Detector">
        <Field label="Temperature" fkey="ccd.temperature" st={st} live={undefined}>
          <ReadOnly value={ccd?.temperature != null ? `${ccd.temperature} °C · ${tempLabel}` : '—'} tone={tempTone} />
        </Field>
        <Field label="Target (°C)" fkey="ccd.target_temp" st={st} live={ccd?.target_temp}>
          <NumberInput live={ccd?.target_temp} step={5} disabled={!ccd} onCommit={v => ccdSet('target_temp', { target_temp: Math.round(v) })} />
        </Field>
        <Field label="Cooler" fkey="ccd.cooler" st={st} live={ccd?.cooler}>
          <Toggle live={ccd?.cooler} onCommit={v => ccdSet('cooler', { cooler: v })}
                  confirmOff="Turn the CCD cooler off? Getting back to the target temperature takes several minutes." />
        </Field>
        <Field label="Pre-amp gain" fkey="ccd.preamp_index" st={st} live={ccd?.preamp_index}>
          <Select live={ccd?.preamp_index} onCommit={v => ccdSet('preamp_index', { preamp_index: Number(v) })}
                  options={(ccd?.preamp_gains ?? []).map((g, i) => [i, `×${g}`])} />
        </Field>
        <Field label="Vertical shift" fkey="ccd.vs_index" st={st} live={ccd?.vs_index}>
          <Select live={ccd?.vs_index} onCommit={v => ccdSet('vs_index', { vs_index: Number(v) })}
                  options={(ccd?.vs_speeds_us ?? []).map((s, i) => [i, `${s} µs`])} />
        </Field>
        <Field label="Horizontal readout" fkey="ccd.hs_index" st={st} live={ccd?.hs_index}>
          <Select live={ccd?.hs_index} onCommit={v => ccdSet('hs_index', { hs_index: Number(v) })}
                  options={(ccd?.hs_speeds_mhz ?? []).map((s, i) => [i, `${s} MHz`])} />
        </Field>
        <Field label="Shutter" fkey="ccd.shutter" st={st} live={ccd?.shutter}>
          <Select live={ccd?.shutter} onCommit={v => ccdSet('shutter', { shutter: v })}
                  options={[['auto', 'Auto'], ['open', 'Open'], ['close', 'Closed']]} />
        </Field>
      </Section>

      <Section title="Laser" note={laser ? undefined : 'offline'}>
        <Field label="State" fkey="laser.is_on" st={st} live={laser?.is_on}>
          <ReadOnly value={laser ? (laser.is_on ? 'Emitting' : 'Off') : '—'} tone={laser?.is_on ? 'text-red-600' : 'text-gray-900'} />
        </Field>
        <Field label="Beam if fired" fkey="laser.beam" st={st} live={laser?.beam}>
          <ReadOnly value={laser ? (laser.beam === 'measurement' ? 'Measurement' : 'Guide') : '—'} />
        </Field>
        <Field label={`Power (mW) · ${laser ? `${fmtMw(laser.power_range[0])}–${fmtMw(laser.power_range[1])}` : ''}`}
               fkey="laser.power_mw" st={st} live={laser?.power_mw}>
          <NumberInput live={laser?.power_mw} step={0.1} range={laser?.power_range} disabled={!laser}
                       fmt={fmtMw} parse={s => parsePower(s, laser?.full_mw)}
                       onReject={reject('laser.power_mw')}
                       onCommit={v => commit('laser.power_mw', { laser: { power_mw: v } })} />
          <span className="block mt-0.5 text-[11px] text-gray-400">
            {laser?.power_pct != null ? `${+laser.power_pct.toFixed(1)}% of ${laser.full_mw} mW · ` : ''}type “10%” for percent
          </span>
        </Field>
      </Section>

      <Section title="Camera" note={camera ? undefined : 'offline'}>
        <Field label="Exposure (ms)" fkey="camera.exposure_ms" st={st} live={camera?.exposure_ms}>
          <NumberInput live={camera?.exposure_ms} step={5} min={0.1} disabled={!camera || !!camera.auto_exposure}
                       onCommit={v => commit('camera.exposure_ms', { camera: { exposure_ms: v } })} />
        </Field>
        <Field label="Auto exposure" fkey="camera.auto_exposure" st={st} live={camera?.auto_exposure}>
          <Toggle live={camera?.auto_exposure} onCommit={v => commit('camera.auto_exposure', { camera: { auto_exposure: v } })} />
        </Field>
      </Section>

      <Section title="Stage" note={stage ? undefined : 'offline'}>
        <Field label="Position X / Y (mm)" fkey="stage.xy" st={st} live={undefined}>
          <ReadOnly value={`${fmt(stage?.x)} / ${fmt(stage?.y)}`} />
        </Field>
        <Field label="Focus Z (mm)" fkey="stage.z" st={st} live={undefined}>
          <ReadOnly value={fmt(stage?.z, 4)} />
        </Field>
        {(['x', 'y', 'z'] as const).map(axis => (
          <Field key={axis} label={`Speed ${axis.toUpperCase()} (mm/s)`} fkey={`stage.speed.${axis}`} st={st} live={stage?.speed?.[axis]}>
            <NumberInput live={stage?.speed?.[axis]} step={axis === 'z' ? 0.01 : 0.1} min={axis === 'z' ? 0.01 : 0.1}
                         range={stage?.speed_range?.[axis]} disabled={!stage?.speed}
                         onReject={reject(`stage.speed.${axis}`)}
                         onCommit={v => commit(`stage.speed.${axis}`, { stage: { speed: { [axis]: v } } })} />
          </Field>
        ))}
      </Section>

      <button
        onClick={onAcquire}
        disabled={isAcquiring || !ccd || !laser}
        className="w-full py-2.5 bg-raman-500 text-white text-sm font-semibold rounded-lg hover:bg-raman-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center gap-2 justify-center"
        title="Measure with the settings shown above"
      >
        {isAcquiring ? (
          <>
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin inline-block" />
            Acquiring…
          </>
        ) : (
          <>▶&nbsp;ACQUIRE</>
        )}
      </button>
    </div>
  )
}
