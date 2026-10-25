import { X, Plus, Trash2 } from 'lucide-react'
import type { Chat } from '../chatStore'
import logoImage from '../logo/logo.png'

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  chats: Chat[]
  activeChatId: string
  onNewChat: () => void
  onSelectChat: (id: string) => void
  onDeleteChat: (id: string) => void
}

export default function Sidebar({
  isOpen,
  onClose,
  chats,
  activeChatId,
  onNewChat,
  onSelectChat,
  onDeleteChat,
}: SidebarProps) {
  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          w-64 bg-white border-r border-gray-200
          transform transition-transform duration-200 ease-in-out
          flex flex-col
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
        aria-label="Navigation sidebar"
      >
        <div className="px-5 pt-6 pb-4 flex items-center justify-between">
          <button
            onClick={() => { onNewChat(); onClose() }}
            className="flex items-center gap-2.5 rounded-lg"
            aria-label="Go to home"
          >
            <img src={logoImage} alt="" className="w-6 h-6 object-contain" />
            <h1 className="text-lg font-semibold text-gray-900">RamanCopilot</h1>
          </button>
          <button
            onClick={onClose}
            className="lg:hidden p-1.5 hover:bg-gray-100 rounded-lg"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="px-3">
          <button
            onClick={() => { onNewChat(); onClose() }}
            className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm font-medium text-gray-900 border border-gray-200 rounded-lg hover:bg-gray-50"
          >
            <Plus className="w-4 h-4" />
            New session
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 pt-6 pb-3">
          <h2 className="px-3 pb-2 text-xs font-medium text-gray-500">Recent</h2>
          <nav className="space-y-0.5" aria-label="Recent sessions">
            {!chats.some(c => c.messages.length > 0) && (
              <p className="px-3 py-2 text-xs text-gray-400">No sessions yet.</p>
            )}
            {chats.filter(c => c.messages.length > 0).map(chat => {
              const active = chat.id === activeChatId
              return (
                <div
                  key={chat.id}
                  className={`group flex items-center rounded-lg ${active ? 'bg-gray-100' : 'hover:bg-gray-50'}`}
                >
                  <button
                    className={`flex-1 min-w-0 px-3 py-2 text-left text-sm truncate rounded-lg ${
                      active ? 'text-gray-900 font-medium' : 'text-gray-700'
                    }`}
                    onClick={() => { onSelectChat(chat.id); onClose() }}
                    title={chat.title}
                  >
                    {chat.title || 'New session'}
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDeleteChat(chat.id) }}
                    className="p-1.5 mr-1 text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 focus:opacity-100 rounded"
                    aria-label="Delete session"
                    title="Delete session"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              )
            })}
          </nav>
        </div>

      </aside>
    </>
  )
}
