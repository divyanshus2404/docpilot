export type Citation = {
  id: number;
  source: string;
  page: number | null;
  score: number;
  content: string;
};

export type DocInfo = { source: string; chunks: number };

export type StreamHandlers = {
  onThread: (threadId: string) => void;
  onStep: (text: string) => void;
  onToken: (text: string) => void;
  onDone: (answer: string, citations: Citation[]) => void;
};

export async function listDocuments(): Promise<DocInfo[]> {
  const r = await fetch("/api/documents");
  return (await r.json()).documents;
}

export async function uploadDocuments(files: FileList): Promise<void> {
  const form = new FormData();
  Array.from(files).forEach((f) => form.append("files", f));
  const r = await fetch("/api/documents", { method: "POST", body: form });
  if (!r.ok) throw new Error((await r.json()).detail ?? "Upload failed");
}

export async function deleteDocument(source: string): Promise<void> {
  await fetch(`/api/documents/${encodeURIComponent(source)}`, { method: "DELETE" });
}

export async function health(): Promise<{ llm_provider: string; llm_model: string; embedding_model: string; uploads_enabled: boolean }> {
  return (await fetch("/api/health")).json();
}

/** POSTs a question and parses the Server-Sent Events stream from /api/chat/stream. */
export async function streamChat(question: string, threadId: string | null, h: StreamHandlers) {
  const r = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: threadId }),
  });
  if (!r.ok || !r.body) throw new Error("Chat request failed");

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const raw of events) {
      const event = raw.match(/^event: (.*)$/m)?.[1];
      const data = JSON.parse(raw.match(/^data: (.*)$/m)?.[1] ?? "{}");
      if (event === "thread") h.onThread(data.thread_id);
      else if (event === "step") h.onStep(data.text);
      else if (event === "token") h.onToken(data.text);
      else if (event === "done") h.onDone(data.answer, data.citations);
    }
  }
}
