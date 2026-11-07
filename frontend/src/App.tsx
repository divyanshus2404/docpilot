import { useEffect, useRef, useState } from "react";
import { Citation, DocInfo, deleteDocument, health, listDocuments, streamChat, uploadDocuments } from "./api";

type Message = {
  role: "user" | "assistant";
  text: string;
  steps?: string[];
  citations?: Citation[];
  pending?: boolean;
};

const EXAMPLES = [
  "How much paid annual leave do employees get?",
  "Can I paste client data into ChatGPT?",
  "What is the hotel limit in metro cities?",
];

export default function App() {
  const [docs, setDocs] = useState<DocInfo[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [canUpload, setCanUpload] = useState(true);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshDocs = () => listDocuments().then(setDocs).catch(() => setError("Backend not reachable"));

  useEffect(() => {
    refreshDocs();
    health()
      .then((h) => {
        setModel(`${h.llm_provider} · ${h.llm_model} · ${h.embedding_model}`);
        setCanUpload(h.uploads_enabled);
      })
      .catch(() => {});
  }, []);

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  const updateLast = (fn: (m: Message) => Message) =>
    setMessages((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])]);

  async function send(question: string) {
    if (!question.trim() || busy) return;
    setInput("");
    setError(null);
    setBusy(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: question },
      { role: "assistant", text: "", steps: [], pending: true },
    ]);
    try {
      await streamChat(question, threadId, {
        onThread: setThreadId,
        onStep: (s) => updateLast((m) => ({ ...m, steps: [...(m.steps ?? []), s] })),
        onToken: (t) => updateLast((m) => ({ ...m, text: m.text + t })),
        onDone: (answer, citations) => updateLast((m) => ({ ...m, text: answer, citations, pending: false })),
      });
    } catch (e) {
      updateLast((m) => ({ ...m, text: "Something went wrong.", pending: false }));
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function onUpload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setError(null);
    try {
      await uploadDocuments(files);
      await refreshDocs();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">◆</span> DocPilot
        </div>
        <p className="tagline">Agentic RAG over your documents</p>

        {canUpload ? (
        <label className={`upload ${uploading ? "disabled" : ""}`}>
          {uploading ? "Indexing…" : "+ Upload PDF / MD / TXT"}
          <input type="file" multiple accept=".pdf,.md,.txt" onChange={(e) => onUpload(e.target.files)} hidden />
        </label>
        ) : (
          <div className="muted small">Demo mode: ask about the sample policies below.</div>
        )}

        <div className="doclist">
          <div className="section-title">Knowledge base</div>
          {docs.length === 0 && <div className="muted">No documents yet</div>}
          {docs.map((d) => (
            <div key={d.source} className="doc">
              <div>
                <div className="doc-name">{d.source}</div>
                <div className="muted small">{d.chunks} {d.chunks === 1 ? "chunk" : "chunks"}</div>
              </div>
              {canUpload && (
                <button className="icon" title="Remove" onClick={() => deleteDocument(d.source).then(refreshDocs)}>
                  ×
                </button>
              )}
            </div>
          ))}
        </div>

        <button
          className="secondary"
          onClick={() => {
            setMessages([]);
            setThreadId(null);
          }}
        >
          New conversation
        </button>
        {model && <div className="muted small model">{model}</div>}
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty">
              <h1>Ask your documents anything</h1>
              <p className="muted">Answers come only from your files, with sources. Follow-up questions work too.</p>
              <div className="examples">
                {EXAMPLES.map((q) => (
                  <button key={q} onClick={() => send(q)}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) => (
            <div key={i} className={`msg ${m.role}`}>
              {m.role === "assistant" && m.steps && m.steps.length > 0 && (
                <details className="trace" open={m.pending}>
                  <summary>{m.pending ? "Thinking…" : `Agent trace · ${m.steps.length} steps`}</summary>
                  <ol>
                    {m.steps.map((s, j) => (
                      <li key={j}>{s}</li>
                    ))}
                  </ol>
                </details>
              )}
              <div className="bubble">{m.text || (m.pending ? <span className="dots">●●●</span> : "")}</div>
              {m.citations && m.citations.length > 0 && (
                <div className="citations">
                  {m.citations.map((c) => (
                    <button key={c.id} className="chip" onClick={() => setOpenCitation(c)}>
                      [{c.id}] {c.source}
                      {c.page ? ` · p.${c.page}` : ""}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {error && <div className="error">{error}</div>}

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={docs.length ? "Ask a question…" : "Upload a document first, or try an example"}
            disabled={busy}
          />
          <button type="submit" disabled={busy || !input.trim()}>
            Send
          </button>
        </form>
      </main>

      {openCitation && (
        <div className="modal-backdrop" onClick={() => setOpenCitation(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <strong>
                [{openCitation.id}] {openCitation.source}
                {openCitation.page ? ` · page ${openCitation.page}` : ""}
              </strong>
              <span className="muted small">similarity {openCitation.score.toFixed(2)}</span>
            </div>
            <pre>{openCitation.content}</pre>
            <button className="secondary" onClick={() => setOpenCitation(null)}>
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
