import { useRef, useState, useCallback, useEffect } from 'react'
import axios from 'axios'
import { applyHwSettings } from '../../hw'

const STREAM_W = 1060
const STREAM_H = 800

export interface AimPoint { x: number; y: number; width: number; height: number }

export interface ObjectiveState { current: string; available: string[] }

interface CameraViewProps {
  aim?: AimPoint | null
  onMoved?: () => void
}

export default function CameraView({ aim, onMoved }: CameraViewProps) {
  const aimX = aim ? (aim.x / aim.width) * STREAM_W : STREAM_W / 2
  const aimY = aim ? (aim.y / aim.height) * STREAM_H : STREAM_H / 2
  const viewportRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })
  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const fit = () => {
      const scale = Math.min(el.clientWidth / STREAM_W, el.clientHeight / STREAM_H)
      setBox({ w: STREAM_W * scale, h: STREAM_H * scale })
    }
    fit()
    const ro = new ResizeObserver(fit)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const [clickPos, setClickPos] = useState<{ x: number; y: number } | null>(null)
  const [moving, setMoving] = useState(false)
  const [live, setLive] = useState(false)

  const handleClick = useCallback(
    async (e: React.MouseEvent<HTMLDivElement>) => {
      if (moving) return
      const rect = e.currentTarget.getBoundingClientRect()
      const relX = e.clientX - rect.left
      const relY = e.clientY - rect.top

      const px = (relX / rect.width) * STREAM_W
      const py = (relY / rect.height) * STREAM_H

      setClickPos({ x: relX, y: relY })
      setMoving(true)

      try {
        await axios.post('/api/stage/move-pixel', {
          px,
          py,
          stream_width: STREAM_W,
          stream_height: STREAM_H,
        })
        onMoved?.()
      } catch (err) {
        console.error('Stage move failed:', err)
      } finally {
        setTimeout(() => {
          setClickPos(null)
          setMoving(false)
        }, 500)
      }
    },
    [moving, onMoved],
  )

  return (
    <div className="bg-[#1f232b] rounded-xl overflow-hidden select-none flex flex-col h-full">
      <div ref={viewportRef} className="relative flex-1 min-h-0 overflow-hidden flex items-center justify-center">
      <div
        className="relative flex-shrink-0 cursor-crosshair overflow-hidden"
        style={{ width: box.w, height: box.h }}
        onClick={handleClick}
      >
        <div className="absolute inset-0 flex items-center justify-center bg-[#1f232b]">
          <span className="text-gray-500 text-sm pointer-events-none">Camera offline</span>
        </div>

        <img
          src="/api/camera/stream"
          className="absolute inset-0 w-full h-full object-contain"
          onError={e => { (e.currentTarget as HTMLImageElement).style.display = 'none'; setLive(false) }}
          onLoad={e => { (e.currentTarget as HTMLImageElement).style.display = ''; setLive(true) }}
          alt=""
          draggable={false}
        />

        {live && <div
          className="absolute pointer-events-none w-4 h-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-orange-400"
          style={{
            left: `${(aimX / STREAM_W) * 100}%`,
            top: `${(aimY / STREAM_H) * 100}%`,
            filter: 'drop-shadow(0 0 1px rgba(0,0,0,0.9))',
          }}
        />}

        {clickPos && (
          <div
            className="absolute pointer-events-none"
            style={{ left: clickPos.x - 10, top: clickPos.y - 10, width: 20, height: 20 }}
          >
            <div className="w-full h-full border-2 border-yellow-400 rounded-full animate-ping opacity-75" />
          </div>
        )}

        {moving && (
          <div className="absolute inset-0 bg-black/20 flex items-center justify-center pointer-events-none">
            <span className="text-white text-xs bg-black/60 px-3 py-1 rounded">Moving…</span>
          </div>
        )}
      </div>
      </div>
    </div>
  )
}

/** Objective magnification selector. */
export function ObjectiveSelector({ objective, onChanged }: { objective: ObjectiveState | null; onChanged?: () => void }) {
  const [switching, setSwitching] = useState<string | null>(null)
  if (!objective) {
    return <span className="text-xs text-gray-400 whitespace-nowrap">Objective — offline</span>
  }
  const select = async (name: string) => {
    if (switching || name === objective.current) return
    setSwitching(name)
    try {
      await applyHwSettings({ optics: { objective: name } })
      onChanged?.()
    } catch (err) {
      console.error('Objective change failed:', err)
    } finally {
      setSwitching(null)
    }
  }
  return (
    <div
      className="inline-flex p-0.5 bg-gray-200/70 rounded-lg"
      title="Objective currently mounted. Set it here after rotating the turret."
    >
      {objective.available.map(name => (
        <button
          key={name}
          type="button"
          disabled={switching !== null}
          onClick={() => select(name)}
          className={`px-3 py-1 rounded-md font-mono text-xs disabled:cursor-wait ${
            name === objective.current
              ? 'bg-white text-gray-900 font-medium shadow-sm'
              : 'text-gray-500 hover:text-gray-900 disabled:opacity-50'
          }`}
        >
          {switching === name ? '…' : name}
        </button>
      ))}
    </div>
  )
}
