# Decision log

Each entry records a choice that is not obvious from the code, and why it was made.

## D1. Source layout and tooling

A `src/` layout with one installable package, `grounded_rag`. Ruff handles linting and formatting, `mypy --strict` runs on `src/`, and tests run under pytest with `asyncio_mode = "auto"`. Tests that need PostgreSQL carry the `db` marker and run when `TEST_DATABASE_URL` is set. Tests that need real OpenAI or Pinecone credentials carry the `live` marker and run only when `RUN_LIVE=1`.

## D2. Settings without an environment prefix

Settings are read from plain names (`DATABASE_URL`, `OPENAI_API_KEY`) because those are the names operators already know from the providers' own tooling. Empty values are treated as unset so a blank line in `.env` falls back to the default. A test asserts that every field of `Settings` appears in `.env.example`.

## D3. Errors carry stable codes, status mapping happens once

`GroundedRagError` subclasses carry a stable `code`. They know nothing about HTTP. The mapping from code to status lives in the API layer only.

## D4. Logging

structlog with JSON lines in production and console output for development. Request ids are bound through context variables so every line written during a request carries one. A processor redacts values stored under sensitive keys as a last line of defence; raw user text is described by length and hash unless `LOG_CONTENT=true`.

## D5. Compose credentials

The Compose file uses fixed local-only PostgreSQL credentials and publishes the port on the loopback interface only. The deployment guide says to replace them for anything else.

## D6. Content-addressed documents

A document id is derived from the corpus and the SHA-256 of its normalised text, so ingesting the same content twice is a no-op, and a changed file becomes a new document. `ingest --replace` removes older documents with the same source path after the new one is ready. Chunk ids are `<doc_id>-<ordinal>` and double as vector ids, which makes upserts and deletes idempotent.

## D7. Ingestion order and resumability

Documents are written to PostgreSQL as `pending`, vectors are upserted, and only then is the document marked `ready`. Retrieval ignores anything that is not `ready`. A failed run leaves a `pending` document that the next run completes with the same ids. Deletion marks the document `deleting` first (which hides it), removes vectors, then removes rows, so it can be repeated until it succeeds. Changing the roles of a ready document updates PostgreSQL first, because it is authoritative, and then refreshes vector metadata.

## D8. Chunking

Chunks follow paragraph and heading boundaries. A heading is merged with the paragraph below it so it is never left alone at the end of a chunk. Paragraphs are packed up to a target size; a paragraph above the maximum is cut at sentence boundaries with a small overlap. Overlap is only used inside oversized paragraphs, because paragraph-aligned chunks already begin at natural boundaries. Chunks are trimmed and `chunk.text == document_text[start_char:end_char]` always holds; property-style tests check ordering, size bounds and that no non-whitespace character is lost. Repeated paragraphs stay as separate chunks because each has its own offsets; retrieval collapses exact duplicates.

## D9. Context header on every chunk

Each chunk has a header made of the document title and section name. The header is prepended to the text for embedding and included in the PostgreSQL full-text vector, but citations and offsets refer to the chunk text alone. This lets a question that names a document find chunks of that document in a multi-document corpus, which plain chunk text would not do.

## D10. Pinecone over REST

The Pinecone adapter calls the REST API with `httpx` instead of the SDK. That keeps the dependency list small, keeps everything asynchronous, and shares one retry helper with the OpenAI client. The API version is sent in a header and is a setting. The adapter is tested against a mock transport that checks request shapes; it has not been run against a live index. Deleting a namespace's vectors that no longer exists is treated as success so deletion stays repeatable.

## D11. OpenAI over REST

Chat and embedding calls use `httpx` directly for the same reasons. Model identifiers and prices are settings, not code. The embedding request passes `dimensions`, which the `text-embedding-3` models accept; a model that does not accept it would need that line changed.

## D12. API keys are hashed with SHA-256

Keys are 256-bit random values, so there is nothing to brute-force and a slow password hash would only add latency to every request. The database stores the hash and a short display prefix; the plaintext is shown once at creation.

## D13. Layout adjustments

Additions to the layout in the project brief: `container.py` (composition root shared by the CLI and the API), `identity.py` (key generation and hashing), `http_retry.py` (shared retry policy), `llm/openai_http.py` (transport), `domain/usage.py`, and `tests/fakes.py`. None of them change a public boundary.

## D14. Dependency notes

`httpx` is both the provider transport and the test client. `pypdf` is pure Python, so the image needs no system libraries. `alembic` runs migrations from a one-shot Compose service. SQLAlchemy is used in its 2.0 style (typed `Mapped` columns, `async_sessionmaker`); the installed release is the 2.1 series.

## D15. Access control is checked three times, and the last check is the one that counts

1. The Pinecone query carries a metadata filter on `allowed_roles`.
2. The PostgreSQL full-text query joins `documents` and filters on `allowed_roles` and `status = 'ready'`.
3. After fusion and before anything reaches the model, every candidate from both retrievers is looked up again in PostgreSQL with the same filter, and the roles are compared once more in Python. A faulty lookup therefore cannot widen access.

Vector metadata can be stale (a role change, a failed re-upsert), which is why the third check exists. Stage traces are filtered to chunks that passed it, and the count of rejected candidates is logged but never returned, because a count would reveal that a document the caller cannot read matched the question. Tests assert that a forbidden document gives the same result as a missing one.

## D16. Retrieval details

Dense and lexical retrieval run concurrently. Lexical search turns the question into an OR query over its words, so a long natural-language question still matches chunks that contain only some of them; ranking uses `ts_rank_cd` with length normalisation. Reciprocal rank fusion merges the lists because it needs no score calibration. The authoritative check runs over the union of both candidate lists, so the traces can be shown safely. Passages that repeat verbatim within a document are collapsed to the best-ranked copy. The lexical query ranks every matching row, which is fine at the size of the evaluation corpus and is listed as a limitation for large corpora.

## D17. Reranking and query rewriting use the chat model

The reranker is listwise: one call sees the opening of each candidate and returns an order. Numbers the model omits or repeats are repaired so a sloppy answer never loses a candidate. A reranker returns an order rather than scores, so the `rerank` stage in a trace shows positional scores. Rewriting is a separate small call that sees the question, earlier queries and the grader's feedback. A rewrite that repeats an earlier query ends the loop instead of paying for the same search again.

## D18. Structured outputs and spend accounting

Chat calls that need JSON send a strict JSON schema derived from a pydantic model; keywords that strict mode rejects are stripped from the schema and enforced by pydantic on the reply. A reply that fails validation still counts toward the request's spend, because the call was billed. Every call is checked against a per-request token and cost budget before it is made, so one call can overshoot the budget; the loop then abstains with `budget_exceeded`.

## D19. The answer loop

- The state is a dataclass and each node returns the fields it changes. A new graph is compiled per request because the nodes close over request-specific services.
- Passages are labelled `P1`, `P2`, ... in prompts because models copy short labels more reliably than chunk ids. The generator returns claims, each with its own citations, so an unsupported claim can be dropped without discarding the rest. The final answer text is the claims joined with numbered markers.
- Verification checks each quote against the chunk it names after lower-casing, collapsing whitespace and folding typographic quotes and dashes. A quote must be at least eight characters. The cited chunk must be one that was shown to the model.
- One repair attempt is allowed. Afterwards unsupported citations are dropped, claims left without a genuine citation are dropped, and if nothing remains the loop abstains with `unsupported_citations`.
- Citations in the response quote the stored text, not the model's version, and carry offsets into the document.
- When retrieval finds nothing the caller may read, the loop abstains with `no_relevant_context` at once; rewriting cannot help.
- Retrieved text is delimited in the prompt, closing tags inside it are defused, the system prompts say to treat it as data, and the generator has no tools.

## D20. Evaluation design

- Questions name the contract they ask about (for example `Which law governs the "Acme Supply Agreement"?`) because the corpus holds many contracts. This makes the document easy to find and the clause hard to find; it is not a test of picking the right contract unprompted. The document title is part of each chunk's context header so the name can match.
- Quality is measured as an administrator who can read every contract, so all of them compete as distractors. Security probes ask as principals who cannot read the target contract.
- Every contract is ingested for every run, so dev and test runs see the same corpus.
- A retrieval hit means a chunk of the right contract overlaps a gold span. Retrieval figures use the first retrieval of each question; a separate figure reports whether the final context, after any rewrite, contains the gold span.
- Rates carry 95% Wilson intervals. Percentiles use the nearest-rank definition.
- All model and embedding calls go through a disk cache keyed by a hash of the request. A replay reports its token counts, priced at the current rates, and credits back the original call's latency, so a cached run reports comparable cost and latency. The spend guard sits below the cache, so it counts only live calls.
- The judge model is a setting. It defaults to the answer model, which the report states, and the documentation explains why a different model is better.
- The dry-run estimate uses fixed guesses for prompt sizes and retry rates. It is labelled as an estimate everywhere it appears.

## D21. Interfaces that exist, and why

`VectorStore` (Pinecone and an in-memory fake for tests), `Embedder` (OpenAI, a cached wrapper, and a fake), `ChatClient` (OpenAI, a cached wrapper, and scripted fakes), `Reranker` (LLM and keep-order), `ChunkStore` (PostgreSQL and a fake that can be told to ignore access rules, to prove the later checks hold). Each has at least two implementations or exists to let tests inject a fault. Nothing else is abstracted.

## D22. Further layout adjustments

`cli_common.py` and `cli_eval.py` split the command line by concern. `agent/assemble.py`, `agent/schemas.py` and `agent/config.py` keep the graph nodes short. `retrieval/rewrite.py` holds query rewriting. `evals/` gained `guard.py`, `environment.py`, `estimate.py` and `probes.py`. Named evaluation configurations live in `evals/configs/*.toml`.

## D23. Hardening added after the first full pass

- Request bodies are limited in size, by declared length and by streamed bytes, before they are parsed.
- Database errors map to `503 dependency_unavailable` with a generic message. Other unexpected exceptions map to `500 internal_error`, built inside the request middleware so the response carries the request id.
- Every piece of untrusted text embedded in a prompt, not only retrieved passages, has its delimiter tags neutralised.
- Saving a document never reopens one that is already `ready` or `deleting`. Two ingestions of the same content at once used to be able to hide a ready document, and the loser failed with a misleading error.
- Tests marked `live` check the OpenAI and Pinecone adapters against the real services. They are skipped by default and are the way to confirm the assumptions that the mock transports encode.
