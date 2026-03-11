import { useRef } from 'react'

/** Stable session ID for multi-turn memory — persists for tab lifetime. */
export function useSession(): string {
  const id = useRef<string>(crypto.randomUUID())
  return id.current
}
