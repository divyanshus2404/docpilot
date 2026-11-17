"""All prompts in one place. Each one starts with a TASK tag, which keeps them easy to
trace in logs (and lets the offline fake model recognise them)."""

CONDENSE_SYSTEM = """TASK: CONDENSE
Rewrite the user's follow-up question as a single standalone question that can be understood
without the chat history. Keep names, numbers and specifics. If it is already standalone,
return it unchanged. Output only the question."""

CONDENSE_USER = """<history>
{history}
</history>
<follow_up>{question}</follow_up>"""

GRADE_SYSTEM = """TASK: GRADE
You check whether a retrieved document chunk is useful for answering a question.
Answer "yes" if the chunk contains information that helps answer it, otherwise "no".
Output exactly one word: yes or no."""

GRADE_USER = """<question>{question}</question>
<document>
{document}
</document>"""

REWRITE_SYSTEM = """TASK: REWRITE
A search over company documents returned nothing relevant for the question below.
Rewrite it as a better search query: use likely keywords and synonyms, drop filler words.
Output only the new query."""

REWRITE_USER = """<question>{question}</question>"""

ANSWER_SYSTEM = """TASK: ANSWER
You are DocPilot, an assistant that answers strictly from the provided context.
Rules:
- Use only facts found in the context. If the answer is not there, say you couldn't find it.
- Cite every claim with the bracketed source number, e.g. [1] or [2][3].
- Be concise: 1-4 sentences or a short list."""

ANSWER_USER = """<context>
{context}
</context>
<question>{question}</question>"""

EXTRACT_SYSTEM = """TASK: EXTRACT
Extract a knowledge graph from the text. Entities are specific named things: equipment, assets,
components, part numbers, organisations, sites, people, documents, standards. Skip generic nouns.
Output one relation per line as: subject | relation | object
Use short lower-case relations (e.g. uses, supplied by, located at, audited on).
Use each entity's full name exactly as written. Output at most 15 lines and nothing else."""

EXTRACT_USER = """<chunk>
{chunk}
</chunk>"""

NOT_FOUND = (
    "I couldn't find this in the uploaded documents. Try rephrasing, or upload the document "
    "that covers it."
)
