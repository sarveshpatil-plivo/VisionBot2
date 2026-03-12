import { useState } from 'react'

const SESSION_KEY = 'supportiq_session_id'
export const MESSAGES_KEY = (id: string) => `supportiq_messages_${id}`

/**
 * Stable session ID persisted to localStorage.
 * Survives page refresh — same session ID = LangGraph restores multi-turn memory.
 * resetSession() clears the current chat and starts a fresh session.
 */
export function useSession(): [string, () => void] {
  const [sessionId, setSessionId] = useState<string>(() => {
    const stored = localStorage.getItem(SESSION_KEY)
    if (stored) return stored
    const newId = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, newId)
    return newId
  })

  const resetSession = () => {
    localStorage.removeItem(MESSAGES_KEY(sessionId))
    const newId = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, newId)
    setSessionId(newId)
  }

  return [sessionId, resetSession]
}
