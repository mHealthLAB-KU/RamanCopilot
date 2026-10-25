import axios from 'axios'

export type Device = 'laser' | 'ccd' | 'camera' | 'stage'

export interface CcdState {
  temperature: number | null
  temp_status: string | null
  target_temp: number | null
  cooler: boolean | null
  exposure_s: number | null
  acq_mode: string | null
  num_acc: number | null
  num_kin: number | null
  kinetic_cycle_s: number | null
  read_mode: string
  hbin: number | null
  track_center: number | null
  track_width: number | null
  trigger: string | null
  shutter: string | null
  preamp_index: number | null
  preamp_gains: number[]
  vs_index: number | null
  vs_speeds_us: number[]
  hs_index: number | null
  hs_speeds_mhz: number[]
}

export interface LaserState {
  is_on: boolean
  beam: 'guide' | 'measurement'
  power_mw: number | null
  power_pct: number | null
  full_mw: number
  power_range: [number, number]
}

export interface CameraState { streaming: boolean; exposure_ms: number | null; auto_exposure: boolean | null }

export interface StageState {
  x?: number; y?: number; z?: number
  speed?: { x: number; y: number; z: number }
  speed_range?: Record<'x' | 'y' | 'z', [number, number]>
}

export interface OpticsState {
  objective: string
  available: string[]
  aim: { x: number; y: number; width: number; height: number }
  fov_mm: { x: number; y: number }
}

export interface HwStatus {
  online: boolean
  connected: Record<Device, boolean>
  ccd: CcdState | null
  laser: LaserState | null
  camera: CameraState | null
  stage: StageState | null
  optics: OpticsState | null
  ccd_init?: string
}

export const OFFLINE: HwStatus = {
  online: false,
  connected: { laser: false, ccd: false, camera: false, stage: false },
  ccd: null, laser: null, camera: null, stage: null, optics: null,
}

export async function fetchHwStatus(): Promise<HwStatus> {
  const { data } = await axios.get('/api/hw/status', { timeout: 2500 })
  return { ...data, online: true }
}

/** Apply some settings to the instruments; throws an Error with the server's reason on failure. */
export async function applyHwSettings(changes: Record<string, unknown>): Promise<void> {
  try {
    await axios.post('/api/hw/settings', changes)
  } catch (e: any) {
    const d = e?.response?.data?.detail
    throw new Error(typeof d === 'string' ? d : e?.message ?? 'Failed to apply')
  }
}
