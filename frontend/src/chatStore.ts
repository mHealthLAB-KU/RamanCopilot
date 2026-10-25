import { useCallback, useEffect, useState } from 'react'

export interface ChatMessage {
  role: 'user' | 'assistant'
  text: string
  isError?: boolean
  id?: number
  kind?: 'progress' | 'clarification' | 'report' | 'spectrum'
  steps?: string[]
  stepNodes?: string[]
  imageUrl?: string
  csvUrl?: string
  jsonUrl?: string
  zipUrl?: string
}

export interface Chat {
  id: string
  title: string
  sessionId: string
  messages: ChatMessage[]
  updatedAt: number
}

const STORAGE_KEY = 'RamanCopilot.chats.v1'
const MAX_CHATS = 200

function genId(): string {
  return `c_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`
}

export function chatTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find(m => m.role === 'user')?.text?.trim()
  if (!firstUser) return 'New session'
  const oneLine = firstUser.split('\n')[0].trim()
  return oneLine.length > 40 ? oneLine.slice(0, 40) + '…' : oneLine || 'New session'
}

export function loadChats(): Chat[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const arr = raw ? JSON.parse(raw) : []
    if (!Array.isArray(arr)) return []
    return (arr as Chat[]).sort((a, b) => b.updatedAt - a.updatedAt)
  } catch {
    return []
  }
}

function persist(chats: Chat[]): Chat[] {
  let list = [...chats].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CHATS)
  while (list.length) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
      return list
    } catch {
      list = list.slice(0, list.length - 1)
    }
  }
  try { localStorage.removeItem(STORAGE_KEY) } catch {  }
  return []
}

export interface UseChats {
  chats: Chat[]
  activeId: string
  activeChat: Chat | undefined
  newChat: () => void
  selectChat: (id: string) => void
  deleteChat: (id: string) => void
  updateActive: (messages: ChatMessage[], sessionId: string) => void
}

/** Manage the chat list and the active conversation, persisted in localStorage. */
export function useChats(): UseChats {
  const [initial] = useState(() => {
    const loaded = loadChats()
    const empty = loaded.find(c => c.messages.length === 0)
    if (empty) return { chats: loaded, activeId: empty.id }
    const c: Chat = { id: genId(), title: 'New session', sessionId: '', messages: [], updatedAt: Date.now() }
    return { chats: [c, ...loaded], activeId: c.id }
  })
  const [chats, setChats] = useState<Chat[]>(initial.chats)
  const [activeId, setActiveId] = useState<string>(initial.activeId)

  useEffect(() => {
    if (!activeId && chats[0]) setActiveId(chats[0].id)
  }, [activeId, chats])

  useEffect(() => { persist(chats) }, [chats])

  const newChat = useCallback(() => {
    setChats(prev => {
      const emptyExisting = prev.find(c => c.messages.length === 0)
      if (emptyExisting) {
        setActiveId(emptyExisting.id)
        return prev
      }
      const c: Chat = { id: genId(), title: 'New session', sessionId: '', messages: [], updatedAt: Date.now() }
      setActiveId(c.id)
      return [c, ...prev]
    })
  }, [])

  const selectChat = useCallback((id: string) => setActiveId(id), [])

  const deleteChat = useCallback((id: string) => {
    setChats(prev => {
      const next = prev.filter(c => c.id !== id)
      if (next.length === 0) {
        const c: Chat = { id: genId(), title: 'New session', sessionId: '', messages: [], updatedAt: Date.now() }
        setActiveId(c.id)
        return [c]
      }
      setActiveId(cur => (cur === id ? next[0].id : cur))
      return next
    })
  }, [])

  const updateActive = useCallback((messages: ChatMessage[], sessionId: string) => {
    if (messages.length === 0) return
    setChats(prev => prev.map(c =>
      c.id === activeId
        ? { ...c, messages, sessionId, title: chatTitle(messages), updatedAt: Date.now() }
        : c,
    ))
  }, [activeId])

  const activeChat = chats.find(c => c.id === activeId)
  return { chats, activeId, activeChat, newChat, selectChat, deleteChat, updateActive }
}
