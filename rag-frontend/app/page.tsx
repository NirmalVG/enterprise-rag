"use client"

import { useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

type Message = { role: "user" | "assistant"; content: string }

export default function Home() {
  const [question, setQuestion] = useState("")
  const [history, setHistory] = useState<Message[]>([])
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim()) return

    const userMsg = question
    setQuestion("")
    setLoading(true)

    // 1. Add the user's question AND an empty assistant placeholder to the UI
    const currentHistory: Message[] = [
      ...history,
      { role: "user", content: userMsg },
      { role: "assistant", content: "" }, // Placeholder for the stream
    ]
    setHistory(currentHistory)

    try {
      const backendUrl =
        process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
      const res = await fetch(`${backendUrl}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // Send the previous history (excluding the new question) to the backend
        body: JSON.stringify({ question: userMsg, chat_history: history }),
      })

      if (!res.body) throw new Error("No response body")

      // 2. Set up the stream reader
      const reader = res.body.getReader()
      const decoder = new TextDecoder("utf-8")
      setLoading(false) // Turn off the bounce animation once data arrives

      let done = false
      while (!done) {
        const { value, done: readerDone } = await reader.read()
        done = readerDone

        if (value) {
          const chunk = decoder.decode(value, { stream: true })

          // 3. Append the new chunk using strict immutable state updates
          setHistory((prev) => {
            const updated = [...prev]

            // Create a brand new copy of the last message to satisfy React Strict Mode
            const lastMessage = { ...updated[updated.length - 1] }

            lastMessage.content += chunk
            updated[updated.length - 1] = lastMessage

            return updated
          })
        }
      }
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-neutral-950 text-neutral-100 font-sans">
      <div className="w-full max-w-4xl flex flex-col h-[90vh]">
        {/* Header */}
        <div className="text-center mb-8 shrink-0">
          <h1 className="text-4xl font-bold tracking-tight mb-2 text-white">
            Enterprise RAG
          </h1>
          <p className="text-neutral-400">
            Hybrid Search, Semantic Chunking, & Memory
          </p>
        </div>

        {/* Chat Log */}
        <div className="flex-1 overflow-y-auto space-y-6 mb-4 p-6 rounded-2xl bg-neutral-900 border border-neutral-800 scrollbar-hide">
          {history.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-neutral-500 italic space-y-4">
              <p>Ask a question about your documents...</p>
            </div>
          ) : (
            history.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] p-5 rounded-2xl ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white shadow-md rounded-br-sm"
                      : "bg-neutral-800 text-neutral-200 shadow-md rounded-bl-sm overflow-x-auto"
                  }`}
                >
                  {msg.role === "user" ? (
                    <p className="whitespace-pre-wrap leading-relaxed">
                      {msg.content}
                    </p>
                  ) : (
                    // Markdown Renderer for AI responses (Tables, Lists, Bold, etc.)
                    <div className="prose prose-invert prose-blue max-w-none break-words">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Loading Animation */}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-neutral-800 p-5 rounded-2xl rounded-bl-sm flex space-x-2 items-center h-12 shadow-md">
                <div className="w-2 h-2 bg-neutral-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-neutral-500 rounded-full animate-bounce delay-100"></div>
                <div className="w-2 h-2 bg-neutral-500 rounded-full animate-bounce delay-200"></div>
              </div>
            </div>
          )}
        </div>

        {/* Input Form */}
        <form
          onSubmit={handleSubmit}
          className="relative flex items-center shrink-0"
        >
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Search your knowledge base..."
            className="w-full bg-neutral-900 border border-neutral-700 text-white rounded-full py-4 pl-6 pr-32 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all shadow-lg"
            disabled={loading}
            autoFocus
          />
          <button
            type="submit"
            disabled={loading || !question.trim()}
            className="absolute right-2 top-2 bottom-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-full px-8 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
          >
            Ask
          </button>
        </form>
      </div>
    </main>
  )
}
