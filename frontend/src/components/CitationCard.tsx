import { useState } from 'react'
import type { Citation } from '../hooks/useChat'

interface Props {
  citation: Citation
  index: number
}

export function CitationCard({ citation, index }: Props) {
  const [expanded, setExpanded] = useState(false)
  const isDocs = citation.source === 'docs'

  const title = isDocs
    ? `${citation.page_title}${citation.section_title ? ` › ${citation.section_title}` : ''}`
    : `#${citation.ticket_id} — ${citation.subject}`

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden text-sm">
      <button
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 hover:bg-gray-100 text-left"
        onClick={() => setExpanded(e => !e)}
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className={`text-xs font-mono px-1.5 py-0.5 rounded shrink-0 ${
            isDocs
              ? 'bg-green-100 text-green-700'
              : 'bg-blue-100 text-blue-700'
          }`}>
            {isDocs ? 'docs' : `[${index}]`}
          </span>
          <span className="font-medium text-gray-800 truncate">
            {title}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          {!isDocs && citation.csat_score && (
            <span className="text-xs text-gray-500">CSAT {citation.csat_score}</span>
          )}
          {!isDocs && citation.resolution_type && (
            <span className="text-xs bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded">
              {citation.resolution_type.replace(/_/g, ' ')}
            </span>
          )}
          <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-3 py-2 bg-white space-y-2">
          {citation.excerpt && (
            <blockquote className="border-l-2 border-blue-300 pl-3 text-gray-600 italic">
              {citation.excerpt}
            </blockquote>
          )}
          {isDocs && citation.url && (
            <a
              href={citation.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-green-600 hover:underline text-xs"
            >
              View in Plivo Docs →
            </a>
          )}
          {!isDocs && citation.zendesk_url && (
            <a
              href={citation.zendesk_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline text-xs"
            >
              View in Zendesk →
            </a>
          )}
        </div>
      )}
    </div>
  )
}
