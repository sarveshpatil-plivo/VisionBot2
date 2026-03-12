import { useRef, useEffect, useState } from 'react'
import { Message } from './components/Message'
import { useChat } from './hooks/useChat'
import { useSession } from './hooks/useSession'

const EXAMPLE_QUERIES = [
  'Why are DTMF tones not detected during IVR navigation?',
  'SMS not delivered to numbers in India — common causes?',
  'SIP registration failing with 403 Forbidden',
  'WebRTC call drops after 30 seconds',
]

export default function App() {
  const [sessionId, resetSession] = useSession()
  const { messages, isStreaming, sendMessage, sendFeedback } = useChat(sessionId)
  const [input, setInput] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    const q = input.trim()
    if (!q || isStreaming) return
    setInput('')
    sendMessage(q)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-3">
        <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold text-sm">
          S
        </div>
        <div className="flex-1">
          <div className="font-semibold text-gray-900">SupportIQ</div>
          <div className="text-xs text-gray-500">Ticket intelligence · Internal</div>
        </div>
        {messages.length > 0 && (
          <button
            onClick={resetSession}
            className="text-sm text-gray-500 hover:text-gray-800 border border-gray-200 rounded-lg px-3 py-1.5 hover:bg-gray-50 transition-colors"
          >
            New chat
          </button>
        )}
      </header>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-16">
              <div className="text-4xl mb-4">🔍</div>
              <h1 className="text-2xl font-bold text-gray-800 mb-2">SupportIQ</h1>
              <p className="text-gray-500 mb-8 max-w-md mx-auto">
                Ask anything about past resolved support tickets. Every answer is cited.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl mx-auto">
                {EXAMPLE_QUERIES.map(q => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); sendMessage(q) }}
                    className="text-left text-sm bg-white border border-gray-200 rounded-xl px-4 py-3 hover:border-indigo-300 hover:bg-indigo-50 transition-colors text-gray-700"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <Message
              key={msg.id}
              message={msg}
              onFeedback={
                msg.role === 'assistant' && !msg.loading
                  ? (helpful) => {
                      const userMsg = messages[i - 1]
                      const ticketIds = msg.citations?.flatMap(c => c.ticket_id ? [c.ticket_id] : []) ?? []
                      sendFeedback(userMsg?.content ?? '', helpful, ticketIds)
                    }
                  : undefined
              }
            />
          ))}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-4 py-4">
        <div className="max-w-3xl mx-auto flex gap-3">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about a support issue..."
            rows={1}
            className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
          <button
            onClick={handleSend}
            disabled={isStreaming || !input.trim()}
            className="bg-indigo-600 text-white rounded-xl px-4 py-2.5 text-sm font-medium hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
          >
            {isStreaming ? '...' : 'Ask'}
          </button>
        </div>
        <div className="max-w-3xl mx-auto mt-1.5 text-xs text-gray-400 text-center">
          Internal tool · Answers based on resolved Zendesk tickets
        </div>
      </div>
    </div>
  )
}
