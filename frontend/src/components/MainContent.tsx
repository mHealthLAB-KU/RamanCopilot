import { useState, useRef, useEffect, useCallback } from 'react'
import { Menu, MessageSquare, Microscope } from 'lucide-react'
import axios from 'axios'
import SearchBar, { Attachment } from './SearchBar'
import ChatThread from './ChatThread'
import RightPanel from './RightPanel'
import { useHardwareStatus, StatusPills, HardwareCards } from './HardwareStatus'
import ParameterPanel from './raman/ParameterPanel'
import { chatTitle, type Chat, type ChatMessage } from '../chatStore'

interface MainContentProps {
  onMenuClick: () => void
  initialChat?: Chat
  onPersist?: (messages: ChatMessage[], sessionId: string) => void
}

export default function MainContent({
  onMenuClick,
  initialChat,
  onPersist,
}: MainContentProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(() => initialChat?.messages ?? [])
  const [chatLoading, setChatLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => initialChat?.sessionId ?? '')
  const [isAcquiring, setIsAcquiring] = useState(false)
  const [hw, refreshHw] = useHardwareStatus()
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, chatLoading])

  useEffect(() => {
    if (!onPersist) return
    const t = window.setTimeout(() => onPersist(messages, sessionId), 800)
    return () => window.clearTimeout(t)
  }, [messages, sessionId, onPersist])

  const handleChat = useCallback(async (command: string, attachments: Attachment[] = []) => {
    const attachNote = attachments.length
      ? '\n\n[Attached files]\n' +
        attachments.map(a => `- ${a.filename} (file_id: ${a.fileId})`).join('\n')
      : ''
    const text = command || (attachments.length ? 'Analyze the attached file(s).' : '')
    if (!text) return
    const fullMessage = text + attachNote

    setMessages(prev => [...prev, { role: 'user', text: fullMessage }])
    setChatLoading(true)

    const streamId = Date.now()
    let localSid = sessionId

    const appendStep = (line: string, node: string) => {
      setMessages(prev => {
        const idx = prev.findIndex(m => m.id === streamId)
        if (idx === -1) {
          return [...prev, {
            role: 'assistant', text: '', kind: 'progress', steps: [line], stepNodes: [node], id: streamId,
          }]
        }
        const copy = [...prev]
        copy[idx] = {
          ...copy[idx],
          steps: [...(copy[idx].steps || []), line],
          stepNodes: [...(copy[idx].stepNodes || []), node],
        }
        return copy
      })
    }

    const handleEvent = (type: string, data: any) => {
      if (data?.session_id) localSid = data.session_id
      switch (type) {
        case 'intent':
          break
        case 'chat':
          setSessionId(localSid)
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.reply || '',
          }])
          break
        case 'node':
          if (data?.message) appendStep(data.message, data.node ?? '')
          break
        case 'spectrum':
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.title || '', kind: 'spectrum',
            imageUrl: data.image_url, csvUrl: data.csv_url, jsonUrl: data.json_url,
            zipUrl: data.zip_url,
          }])
          break
        case 'clarification':
          setSessionId(localSid)
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.question || 'More information is needed.', kind: 'clarification',
          }])
          break
        case 'done': {
          setSessionId(localSid)
          const report = data.final_report || data.abort_reason || 'Experiment complete.'
          setMessages(prev => [...prev, {
            role: 'assistant', text: report, kind: 'report', isError: !!data.abort_reason,
          }])
          refreshHw()
          break
        }
        case 'error':
          setSessionId(localSid)
          setMessages(prev => [...prev, {
            role: 'assistant', text: data.detail || 'Experiment failed.', isError: true,
          }])
          break
      }
    }

    try {
      const resp = await fetch('/api/experiment/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: fullMessage, session_id: sessionId }),
      })
      if (!resp.ok || !resp.body) throw new Error(`Server error (${resp.status})`)

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const blocks = buffer.split('\n\n')
        buffer = blocks.pop() ?? ''
        for (const block of blocks) {
          if (!block.trim()) continue
          let etype = 'message'
          let dataStr = ''
          for (const line of block.split('\n')) {
            if (line.startsWith('event:')) etype = line.slice(6).trim()
            else if (line.startsWith('data:')) dataStr += line.slice(5).trim()
          }
          if (dataStr) {
            try { handleEvent(etype, JSON.parse(dataStr)) } catch {  }
          }
        }
      }
    } catch (err: any) {
      setSessionId(localSid)
      setMessages(prev => [...prev, {
        role: 'assistant', text: err?.message ?? 'Request failed', isError: true,
      }])
    } finally {
      setChatLoading(false)
    }
  }, [sessionId, refreshHw])

  const handleAcquire = useCallback(async () => {
    setIsAcquiring(true)
    try {
      const { data } = await axios.post('/api/spectrum/acquire', {})
      const saved = data?.saved
      if (saved && (saved.json_url || saved.image_url)) {
        setMessages(prev => [...prev, {
          role: 'assistant', text: saved.title || 'Manual acquisition', kind: 'spectrum',
          imageUrl: saved.image_url, csvUrl: saved.csv_url, jsonUrl: saved.json_url, zipUrl: saved.zip_url,
        }])
      }
      refreshHw()
    } catch (err: any) {
      const d = err?.response?.data?.detail
      const why = Array.isArray(d)
        ? d.map((e: any) => `${(e.loc ?? []).slice(-1)[0] ?? ''}: ${e.msg}`).join(' / ')
        : (d ?? err?.message ?? 'Unknown error')
      setMessages(prev => [...prev, {
        role: 'assistant', text: `Manual acquisition failed — ${why}`, isError: true,
      }])
    } finally {
      setIsAcquiring(false)
    }
  }, [refreshHw])

  const isEmpty = messages.length === 0
  const [narrowView, setNarrowView] = useState<'chat' | 'panel'>('chat')

  return (
    <div className="flex-1 flex flex-col overflow-hidden min-h-0">
      <header className={`shrink-0 flex items-center justify-between gap-4 px-6 py-4 ${isEmpty ? '' : 'bg-white border-b border-gray-200'}`}>
        <div className="flex-1 flex items-center gap-3 min-w-0">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 hover:bg-gray-100 rounded-lg"
            aria-label="Open sidebar"
          >
            <Menu className="w-5 h-5 text-gray-700" />
          </button>
          {!isEmpty && (
            <div className="min-w-0">
              <div className="font-semibold text-gray-900 truncate">{chatTitle(messages)}</div>
              <div className="text-sm text-gray-500 truncate"></div>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <StatusPills status={hw} compact={!isEmpty} />
          {!isEmpty && (
            <button
              onClick={() => setNarrowView(v => (v === 'chat' ? 'panel' : 'chat'))}
              className="lg:hidden inline-flex items-center gap-1.5 px-3 py-1.5 text-sm border border-gray-200 rounded-lg hover:bg-gray-50"
            >
              {narrowView === 'chat'
                ? <><Microscope className="w-4 h-4" />Instrument</>
                : <><MessageSquare className="w-4 h-4" />Chat</>}
            </button>
          )}
        </div>
      </header>

      <div className="flex-1 flex overflow-hidden min-h-0">
        <div className={`flex-1 flex-col min-w-0 min-h-0 ${!isEmpty && narrowView === 'panel' ? 'hidden lg:flex' : 'flex'}`}>
          {isEmpty ? (
            <div className="flex-1 overflow-y-auto px-6 py-12">
              <div className="max-w-5xl mx-auto">
                <h1 className="text-4xl font-bold tracking-tight text-gray-900 text-center">
                  What should we measure today?
                </h1>
                <p className="mt-4 text-lg text-gray-500 text-center">
                  Describe the sample and your goal. RamanCopilot plans the run and operates the microscope for you.
                </p>
                <div className="mt-10">
                  <SearchBar
                    onSubmit={handleChat}
                    isLoading={chatLoading}
                    large
                    placeholder="e.g. Find intact organoids in the current field and take one spectrum from each with the lowest exposure that still gives a usable signal."
                  />
                </div>
                <h2 className="mt-12 mb-3 text-sm font-medium text-gray-500">
                  Hardware{!hw.online && <span className="ml-2 text-red-600">· backend not reachable</span>}
                </h2>
                <HardwareCards status={hw} onChanged={refreshHw} />
              </div>
            </div>
          ) : (
            <>
              <div className="flex-1 overflow-y-auto px-6 py-8">
                <div className="max-w-3xl mx-auto">
                  <ChatThread messages={messages} loading={chatLoading} />
                  <div ref={bottomRef} />
                </div>
              </div>
              <div className="shrink-0 px-6 pb-5 pt-2">
                <div className="max-w-3xl mx-auto">
                  <SearchBar onSubmit={handleChat} isLoading={chatLoading} />
                </div>
              </div>
            </>
          )}
        </div>

        {!isEmpty && <RightPanel
          className={narrowView === 'panel' ? 'flex' : 'hidden lg:flex'}
          messages={messages}
          status={hw}
          onMoved={refreshHw}
          parameters={
            <ParameterPanel status={hw} onChanged={refreshHw} isAcquiring={isAcquiring} onAcquire={handleAcquire} />
          }
        />}
      </div>
    </div>
  )
}
