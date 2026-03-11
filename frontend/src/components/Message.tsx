import ReactMarkdown from 'react-markdown'
import type { Message as MessageType } from '../hooks/useChat'
import { CitationCard } from './CitationCard'
import { ConfidenceBadge } from './ConfidenceBadge'
import { SuggestedAction } from './SuggestedAction'
import { RelatedTickets } from './RelatedTickets'

interface Props {
  message: MessageType
  onFeedback?: (helpful: boolean) => void
}

export function Message({ message, onFeedback }: Props) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-xl bg-blue-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm">
          {message.content}
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div className="flex gap-3 max-w-3xl">
      <div className="shrink-0 w-7 h-7 bg-indigo-100 rounded-full flex items-center justify-center text-sm mt-0.5">
        🤖
      </div>
      <div className="flex-1 space-y-3">
        {/* Loading indicator */}
        {message.loading && !message.content && (
          <div className="flex gap-1 items-center text-gray-400 text-sm py-1">
            <span className="animate-bounce">●</span>
            <span className="animate-bounce [animation-delay:0.1s]">●</span>
            <span className="animate-bounce [animation-delay:0.2s]">●</span>
          </div>
        )}

        {/* Answer */}
        {message.content && (
          <div className="prose prose-sm max-w-none text-gray-800">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Confidence + Suggested Action */}
        {!message.loading && message.confidence_score !== undefined && (
          <div className="flex flex-wrap items-start gap-2">
            <ConfidenceBadge
              score={message.confidence_score}
              factors={message.confidence_factors ?? {}}
            />
          </div>
        )}

        {message.suggested_action && (
          <SuggestedAction action={message.suggested_action} />
        )}

        {/* Citations */}
        {message.citations && message.citations.length > 0 && (
          <div className="space-y-1.5">
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Sources
            </div>
            {message.citations.map((c, i) => (
              <CitationCard key={c.ticket_id} citation={c} index={i + 1} />
            ))}
          </div>
        )}

        {/* Related Tickets */}
        {message.related_tickets && message.related_tickets.length > 0 && (
          <RelatedTickets tickets={message.related_tickets} />
        )}

        {/* Feedback */}
        {!message.loading && onFeedback && (
          <div className="flex items-center gap-2 pt-1">
            <span className="text-xs text-gray-400">Was this helpful?</span>
            <button
              onClick={() => onFeedback(true)}
              className="text-xs text-gray-400 hover:text-green-600 transition-colors"
            >👍</button>
            <button
              onClick={() => onFeedback(false)}
              className="text-xs text-gray-400 hover:text-red-500 transition-colors"
            >👎</button>
          </div>
        )}
      </div>
    </div>
  )
}
