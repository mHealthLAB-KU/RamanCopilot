import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Check, Loader2, Clock, AlertCircle } from 'lucide-react'
import type { ChatMessage } from '../chatStore'
import SpectrumChart, { SpectrumMeta, useSpectrum } from './raman/SpectrumChart'
import logoImage from '../logo/logo.png'

const MD_COMPONENTS = {
  p:          ({ node, ...p }: any) => <p className="mb-2 last:mb-0" {...p} />,
  ul:         ({ node, ...p }: any) => <ul className="list-disc pl-5 mb-2 last:mb-0 space-y-0.5" {...p} />,
  ol:         ({ node, ...p }: any) => <ol className="list-decimal pl-5 mb-2 last:mb-0 space-y-0.5" {...p} />,
  li:         ({ node, ...p }: any) => <li className="leading-relaxed" {...p} />,
  strong:     ({ node, ...p }: any) => <strong className="font-semibold" {...p} />,
  em:         ({ node, ...p }: any) => <em className="italic" {...p} />,
  h1:         ({ node, ...p }: any) => <h1 className="text-base font-bold mt-1 mb-1.5" {...p} />,
  h2:         ({ node, ...p }: any) => <h2 className="text-sm font-bold mt-1 mb-1.5" {...p} />,
  h3:         ({ node, ...p }: any) => <h3 className="text-sm font-semibold mt-1 mb-1" {...p} />,
  a:          ({ node, ...p }: any) => <a className="text-raman-600 underline" target="_blank" rel="noreferrer" {...p} />,
  code:       ({ node, ...p }: any) => <code className="px-1 py-0.5 rounded bg-black/5 font-mono text-[0.85em]" {...p} />,
  pre:        ({ node, ...p }: any) => <pre className="p-2 rounded bg-black/5 font-mono text-xs overflow-x-auto mb-2 [&>code]:p-0 [&>code]:bg-transparent [&>code]:text-[1em]" {...p} />,
  blockquote: ({ node, ...p }: any) => <blockquote className="border-l-2 border-gray-300 pl-2 text-gray-600" {...p} />,
  hr:         ({ node, ...p }: any) => <hr className="my-2 border-gray-200" {...p} />,
  table:      ({ node, ...p }: any) => (
    <div className="overflow-x-auto mb-2 last:mb-0">
      <table className="text-sm border-collapse" {...p} />
    </div>
  ),
  thead:      ({ node, ...p }: any) => <thead className="bg-gray-50" {...p} />,
  th:         ({ node, ...p }: any) => <th className="border border-gray-200 px-3 py-1.5 text-left font-semibold" {...p} />,
  td:         ({ node, ...p }: any) => <td className="border border-gray-200 px-3 py-1.5 align-top" {...p} />,
  del:        ({ node, ...p }: any) => <del className="text-gray-500" {...p} />,
}

const MD_REMARK_PLUGINS = [remarkGfm, remarkMath]
const MD_REHYPE_PLUGINS = [rehypeKatex]

function normalizeMath(src: string): string {
  return src
    .replace(/\\\[([\s\S]+?)\\\]/g, (_m, e) => `$$${e}$$`)
    .replace(/\\\(([\s\S]+?)\\\)/g, (_m, e) => `$${e}$`)
}

function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={MD_REMARK_PLUGINS} rehypePlugins={MD_REHYPE_PLUGINS} components={MD_COMPONENTS}>
      {normalizeMath(text)}
    </ReactMarkdown>
  )
}

/** Convert a snake_case tool name to a display label ("auto_focus" -> "Auto focus"). */
export function toolLabel(name?: string): string {
  if (!name) return 'Step'
  const s = name.replace(/_/g, ' ')
  return s.charAt(0).toUpperCase() + s.slice(1)
}

const SPECTRUM_LINE = '#2f49d9'

/** Spectrum card shared by the chat and the live view; plots the measurement JSON or falls back to the saved PNG. */
export function SpectrumCard({ msg, height = 220, bare = false }: { msg: ChatMessage; height?: number; bare?: boolean }) {
  const { data, error } = useSpectrum(msg.jsonUrl)
  return (
    <div className={bare ? 'space-y-2' : 'bg-white border border-gray-200 rounded-xl p-4 space-y-2'}>
      {!bare && msg.text && <div className="text-sm font-medium text-gray-900">{msg.text}</div>}
      {data && <SpectrumMeta data={data} />}
      {data
        ? <SpectrumChart height={height} series={[{ key: msg.jsonUrl!, label: msg.text, color: SPECTRUM_LINE, data }]} />
        : error && msg.imageUrl
          ? (
            <a href={msg.imageUrl} target="_blank" rel="noreferrer">
              <img src={msg.imageUrl} alt={msg.text || 'Spectrum'} className="rounded-lg max-w-full" />
            </a>
          )
          : <div className="flex items-center justify-center text-sm text-gray-400" style={{ height }}>Loading…</div>}
      <div className="flex gap-4 text-xs text-raman-600 flex-wrap">
        {msg.csvUrl && <a href={msg.csvUrl} download className="hover:underline">CSV</a>}
        {msg.imageUrl && <a href={msg.imageUrl} download className="hover:underline">PNG</a>}
        {msg.zipUrl && <a href={msg.zipUrl} download className="hover:underline">All files (zip)</a>}
      </div>
    </div>
  )
}

/** List of tool execution steps; the last step is shown as in progress while running. */
function StepList({ msg, running }: { msg: ChatMessage; running: boolean }) {
  const steps = msg.steps || []
  return (
    <div className="bg-white border border-gray-200 rounded-xl divide-y divide-gray-100 overflow-hidden">
      {steps.map((s, j) => {
        const live = running && j === steps.length - 1
        return (
          <div key={j} className={`flex items-start gap-4 px-5 py-3 text-sm ${live ? 'bg-orange-50' : ''}`}>
            {live
              ? <Loader2 className="w-4 h-4 mt-0.5 shrink-0 text-amber-700 animate-spin" />
              : <Check className="w-4 h-4 mt-0.5 shrink-0 text-green-700" />}
            {msg.stepNodes?.[j] && (
              <span className="w-40 shrink-0 font-medium text-gray-900 truncate" title={msg.stepNodes[j]}>
                {toolLabel(msg.stepNodes[j])}
              </span>
            )}
            <span className={`min-w-0 break-words ${live ? 'text-amber-800' : 'text-gray-600'}`}>{s}</span>
          </div>
        )
      })}
    </div>
  )
}

function AssistantBody({ msg, running }: { msg: ChatMessage; running: boolean }) {
  if (msg.kind === 'progress') return <StepList msg={msg} running={running} />
  if (msg.kind === 'spectrum') return <SpectrumCard msg={msg} />
  if (msg.isError) {
    return (
      <div className="flex gap-3 bg-red-50 border border-red-200 text-red-800 rounded-xl px-5 py-4 text-sm">
        <AlertCircle className="w-4 h-4 mt-0.5 shrink-0" />
        <div className="min-w-0"><Markdown text={msg.text || ''} /></div>
      </div>
    )
  }
  if (msg.kind === 'clarification') {
    return (
      <div className="bg-white border border-orange-200 rounded-xl px-5 py-4 text-sm">
        <div className="flex items-center gap-2 font-semibold text-gray-900 mb-1.5">
          <Clock className="w-4 h-4 text-amber-700" />
          Waiting for your input
        </div>
        <div className="text-gray-700"><Markdown text={msg.text || ''} /></div>
        <div className="mt-2 text-xs text-gray-400">Reply in the box below to continue.</div>
      </div>
    )
  }
  return <div className="text-[15px] text-gray-800 leading-relaxed"><Markdown text={msg.text || ''} /></div>
}

interface ChatThreadProps {
  messages: ChatMessage[]
  loading: boolean
}

export default function ChatThread({ messages, loading }: ChatThreadProps) {
  const lastUser = messages.map(m => m.role).lastIndexOf('user')
  const p = messages.map(m => m.kind).lastIndexOf('progress')
  const lastProgress = loading && p > lastUser ? p : -1

  return (
    <div className="space-y-4">
      {messages.map((msg, i) => {
        if (msg.role === 'user') {
          return (
            <div key={msg.id ?? i} className="flex justify-end pt-2">
              <div className="max-w-[85%] bg-[#e8ecf4] text-gray-900 px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed whitespace-pre-wrap">
                {msg.text}
              </div>
            </div>
          )
        }
        const showName = i === 0 || messages[i - 1].role === 'user'
        return (
          <div key={msg.id ?? i} className="space-y-2">
            {showName && (
              <div className="flex items-center gap-2 font-semibold text-gray-900">
                <img src={logoImage} alt="" className="w-5 h-5 object-contain" />
                RamanCopilot
              </div>
            )}
            <AssistantBody msg={msg} running={i === lastProgress} />
          </div>
        )
      })}
      {loading && lastProgress === -1 && (
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Loader2 className="w-4 h-4 animate-spin" />
          Thinking…
        </div>
      )}
    </div>
  )
}
