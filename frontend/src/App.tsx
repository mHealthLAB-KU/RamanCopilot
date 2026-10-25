import { useState } from 'react'
import Sidebar from './components/Sidebar'
import MainContent from './components/MainContent'
import { useChats } from './chatStore'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const { chats, activeId, activeChat, newChat, selectChat, deleteChat, updateActive } = useChats()

  return (
    <div className="flex h-screen overflow-hidden bg-[#f4f5f7]">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-raman-500 focus:text-white focus:rounded"
      >
        Skip to content
      </a>

      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        chats={chats}
        activeChatId={activeId}
        onNewChat={newChat}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
      />

      <main id="main-content" className="flex-1 flex flex-col overflow-hidden min-w-0">
        <MainContent
          key={activeId}
          onMenuClick={() => setSidebarOpen(true)}
          initialChat={activeChat}
          onPersist={updateActive}
        />
      </main>
    </div>
  )
}

export default App
