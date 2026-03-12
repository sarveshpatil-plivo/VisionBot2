import { useState, useCallback, useEffect } from 'react'
import { MESSAGES_KEY } from './useSession'

const API_URL = '/api'
const TOKEN = import.meta.env.VITE_API_TOKEN ?? ''

export interface Citation {
  source: 'ticket' | 'docs'
  // Ticket
  ticket_id?: string
  subject?: string
  resolution_type?: string
  csat_score?: number
  zendesk_url?: string
  // Docs
  page_title?: string
  section_title?: string
  url?: string
  // Common
  excerpt?: string
}

export interface RelatedTicket {
  ticket_id: string
  subject: string
  resolution_type?: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  confidence_score?: number
  confidence_factors?: Record<string, string | number>
  suggested_action?: string
  related_tickets?: RelatedTicket[]
  awaiting_clarification?: boolean
  loading?: boolean
}

export function useChat(sessionId: string) {
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const stored = localStorage.getItem(MESSAGES_KEY(sessionId))
      if (stored) return (JSON.parse(stored) as Message[]).filter(m => !m.loading)
    } catch {}
    return []
  })
  const [isStreaming, setIsStreaming] = useState(false)

  // When sessionId changes (new chat), load messages for the new session
  useEffect(() => {
    try {
      const stored = localStorage.getItem(MESSAGES_KEY(sessionId))
      if (stored) {
        setMessages((JSON.parse(stored) as Message[]).filter(m => !m.loading))
      } else {
        setMessages([])
      }
    } catch {
      setMessages([])
    }
  }, [sessionId])

  // Persist messages to localStorage whenever they change (skip in-progress loading bubbles)
  useEffect(() => {
    const toSave = messages.filter(m => !m.loading)
    if (toSave.length > 0) {
      localStorage.setItem(MESSAGES_KEY(sessionId), JSON.stringify(toSave))
    }
  }, [messages, sessionId])

  const sendMessage = useCallback(async (question: string) => {
    if (isStreaming) return

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: question }
    const botId = crypto.randomUUID()
    const botMsg: Message = { id: botId, role: 'assistant', content: '', loading: true }

    setMessages(prev => [...prev, userMsg, botMsg])
    setIsStreaming(true)

    try {
      const resp = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${TOKEN}`,
        },
        body: JSON.stringify({ question, session_id: sessionId, metadata_filters: {} }),
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const reader = resp.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let content = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = JSON.parse(line.slice(6))

          if (payload.type === 'token') {
            content += payload.content
            setMessages(prev =>
              prev.map(m => m.id === botId ? { ...m, content, loading: false } : m)
            )
          } else if (payload.type === 'done') {
            setMessages(prev =>
              prev.map(m =>
                m.id === botId
                  ? {
                      ...m,
                      content,
                      loading: false,
                      citations: payload.citations ?? [],
                      confidence_score: payload.confidence_score,
                      confidence_factors: payload.confidence_factors,
                      suggested_action: payload.suggested_action,
                      related_tickets: payload.related_tickets ?? [],
                      awaiting_clarification: payload.awaiting_clarification,
                    }
                  : m
              )
            )
          } else if (payload.type === 'error') {
            setMessages(prev =>
              prev.map(m =>
                m.id === botId
                  ? { ...m, content: 'Something went wrong. Please try again.', loading: false }
                  : m
              )
            )
          }
        }
      }
    } catch (err) {
      setMessages(prev =>
        prev.map(m =>
          m.id === botId
            ? { ...m, content: 'Connection error. Please try again.', loading: false }
            : m
        )
      )
    } finally {
      setIsStreaming(false)
    }
  }, [isStreaming, sessionId])

  const sendFeedback = useCallback(async (question: string, helpful: boolean, ticketIds: string[]) => {
    await fetch(`${API_URL}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${TOKEN}` },
      body: JSON.stringify({ session_id: sessionId, question, helpful, ticket_ids: ticketIds }),
    })
  }, [sessionId])

  return { messages, isStreaming, sendMessage, sendFeedback }
}
