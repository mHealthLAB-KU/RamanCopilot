import { useState, useRef, useEffect } from 'react'
import { Plus, ArrowUp, Loader2, X, FileSpreadsheet, AlertCircle } from 'lucide-react'
import axios from 'axios'

export interface Attachment {
  fileId: string
  filename: string
  bytes: number
}

interface PendingFile {
  key: number
  filename: string
  status: 'uploading' | 'done' | 'error'
  fileId?: string
  bytes?: number
  error?: string
}

interface SearchBarProps {
  onSubmit?: (command: string, attachments: Attachment[]) => void
  isLoading?: boolean
  placeholder?: string
  large?: boolean
}

const ACCEPT = '.csv,.tsv,.txt,.dat,.xlsx,.xls'

function formatBytes(n?: number): string {
  if (!n) return ''
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}

export default function SearchBar({
  onSubmit,
  isLoading = false,
  placeholder = 'Ask a follow-up or change the plan...',
  large = false,
}: SearchBarProps) {
  const [value, setValue] = useState('')
  const [focused, setFocused] = useState(false)
  const [pending, setPending] = useState<PendingFile[]>([])
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList?.length) return
    const picked = Array.from(fileList)
    const base = Date.now()

    setPending(prev => [
      ...prev,
      ...picked.map((f, i) => ({
        key: base + i, filename: f.name, status: 'uploading' as const,
      })),
    ])

    await Promise.all(picked.map(async (f, i) => {
      const key = base + i
      const form = new FormData()
      form.append('file', f)
      try {
        const { data } = await axios.post('/api/files/upload', form)
        setPending(prev => prev.map(p => p.key === key
          ? { ...p, status: 'done', fileId: data.file_id, bytes: data.bytes }
          : p))
      } catch (err: any) {
        setPending(prev => prev.map(p => p.key === key
          ? { ...p, status: 'error', error: err?.response?.data?.detail ?? 'Upload failed' }
          : p))
      }
    }))
  }

  const removePending = (key: number) =>
    setPending(prev => prev.filter(p => p.key !== key))

  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
    el.style.overflowY = el.scrollHeight > 200 ? 'auto' : 'hidden'
  }, [value])

  const uploading = pending.some(p => p.status === 'uploading')
  const ready = pending.filter(p => p.status === 'done')
  const canSubmit = (!!value.trim() || ready.length > 0) && !isLoading && !uploading

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return
    const command = value.trim()
    const attachments: Attachment[] = ready.map(p => ({
      fileId: p.fileId!, filename: p.filename, bytes: p.bytes ?? 0,
    }))
    setValue('')
    setPending([])
    onSubmit?.(command, attachments)
  }

  return (
    <form onSubmit={handleSubmit} className="w-full">
      <div
        className={`
          relative flex flex-col gap-2 px-5 pt-4 pb-3
          bg-white border rounded-2xl transition-shadow
          ${focused ? 'border-raman-300 shadow-md' : 'border-gray-200 shadow-sm'}
        `}
      >
        {pending.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {pending.map(p => (
              <span
                key={p.key}
                className={`inline-flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-lg text-xs border ${
                  p.status === 'error'
                    ? 'bg-red-50 border-red-200 text-red-700'
                    : 'bg-gray-50 border-gray-200 text-gray-700'
                }`}
                title={p.status === 'error' ? p.error : p.fileId}
              >
                {p.status === 'uploading'
                  ? <Loader2 className="w-3.5 h-3.5 animate-spin text-gray-400" />
                  : p.status === 'error'
                  ? <AlertCircle className="w-3.5 h-3.5" />
                  : <FileSpreadsheet className="w-3.5 h-3.5 text-raman-500" />}
                <span className="max-w-[180px] truncate">{p.filename}</span>
                {p.status === 'done' && (
                  <span className="text-gray-400">{formatBytes(p.bytes)}</span>
                )}
                <button
                  type="button"
                  onClick={() => removePending(p.key)}
                  className="p-0.5 hover:bg-gray-200 rounded"
                  aria-label={`Remove ${p.filename}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))}
          </div>
        )}

        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          className="w-full resize-none outline-none bg-transparent text-gray-900 placeholder-gray-400 max-h-[200px]"
          rows={large ? 2 : 1}
          autoComplete="off"
          autoCorrect="off"
          spellCheck={false}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit(e as unknown as React.FormEvent)
            }
          }}
        />

        <div className="flex items-center justify-between gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="hidden"
            onChange={e => { handleFiles(e.target.files); e.target.value = '' }}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="p-2 border border-gray-200 hover:bg-gray-50 rounded-lg"
            aria-label="Attach file"
            title="Attach data file (csv, excel, txt)"
          >
            <Plus className="w-4 h-4 text-gray-700" />
          </button>

          <button
            type="submit"
            disabled={!canSubmit}
            className="p-2.5 bg-raman-500 hover:bg-raman-600 disabled:opacity-40 text-white rounded-xl"
            aria-label="Send message"
            title={uploading ? 'Uploading files…' : undefined}
          >
            {isLoading || uploading
              ? <Loader2 className="w-5 h-5 animate-spin" />
              : <ArrowUp className="w-5 h-5" />
            }
          </button>
        </div>
      </div>
    </form>
  )
}
