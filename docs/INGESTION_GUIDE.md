# Ingestion Guide (owner only)

`python -m src.ingestion` is the **only** way the book enters the system. There is no upload endpoint,
no upload widget, no book selector and no HTTP route that triggers ingestion.

## Options

| Command | Effect |
|---|---|
| `python -m src.ingestion` | Index if anything changed; otherwise report `up_to_date`. |
| `--detect-chapters-only` | Write a draft `data/config/chapters.json`, write a report, stop (no embedding). |
| `--strict` | Refuse to index unless `chapters.json` has `"reviewed": true`. |
| `--force` | Rebuild even if the fingerprint is unchanged. |
| `--keep-old` | Do not delete the previous index collection after switching. |

## Pipeline

1. **Validate PDF**: exists; opens (encrypted/corrupt → clear error); text extracted **page by page** with pypdf.
   A PDF is rejected as "scanned" only if **fewer than 20 % of pages** have ≥100 characters. A blank cover or a
   few image pages only produce a warning.
2. **Clean text**: tab/space normalisation (this style of PDF emits tabs between words), reflow of hard-wrapped lines into
   paragraphs (heuristic), conservative removal of bare page numbers and short lines repeated on ≥30 % of pages
   (reported, never silent). Hyphenation is **not** rewritten, to avoid corrupting real hyphenated words.
3. **Chapters** (see below), producing segments split at chapter/section starts, **including mid-page**.
4. **Chunk** each segment: LangChain `RecursiveCharacterTextSplitter` (paragraph → line → sentence → clause → word),
   1000 chars / 150 overlap by default, never across pages or chapters. Page-tail fragments under `MIN_CHUNK_CHARS` (300) are merged into the previous chunk (exact source text, exact offsets). Token counts come from the embedding model's own
   tokenizer; anything over the model limit (256 for all-MiniLM-L6-v2) is split further, never truncated.
5. **Embed** locally with SentenceTransformers (normalised vectors, same preprocessing as questions). Each chunk is embedded as `<chapter> - <section>\n<text>` so passages carry their context; the token check includes that prefix, and the stored/cited text is only the book text.
6. **Build a new Chroma collection** named after a fingerprint, verify count + self-retrieval, then **atomically**
   rewrite `runtime/active_index.json`. Only then is the previous collection deleted.
7. **Report** → `runtime/reports/ingestion_report.json|md`.

## Chapter detection rules

Interpretation of the required priority:

1. A mapping file marked `"reviewed": true` is the owner's decision and is used verbatim (manual).
2. Otherwise a draft is generated from **PDF bookmarks** (outline) when valid.
3. Otherwise a **heading regex** proposes chapters (`Chapter 3`, `CHAPTER IV: Title`, `Part Two`, …). It requires a
   separator or a bare number line, so prose like "Chapter 3 explains…" and arbitrary uppercase lines are ignored. Heuristic output is always flagged for review.
4. Otherwise chunks are labelled **"Chapter unavailable"**; titles are never invented.

Mid-page starts: each entry's `anchor` (heading text, punctuation-insensitive) or explicit `anchor_offset` locates the start
inside the page; text before it stays with the previous chapter. If an anchor isn't found, the entry is assumed to start at
the top of its page and a warning is added.

Front/back matter whose bookmark matches common names (title page, contents, copyright, "also by…", about the author…)
is auto-marked `"indexed": false`. Change that flag if you disagree.

## Page numbers

Every chunk stores the zero-based `pdf_page_index` and one-based `pdf_page_number` (what a PDF viewer shows). A
**printed page label** is stored only if the PDF defines real `/PageLabels`; pypdf's synthesised "1, 2, 3…" labels are discarded.
Your supplied PDF has no real labels, so citations show **"PDF page N"** only.

## Idempotency and change detection

The index **fingerprint** hashes: pipeline version, book id, PDF SHA-256, chunk size/overlap, the effective chapter
mapping, and the embedding model name/dimension/max length/sentence-transformers version/preprocessing version.
Same fingerprint + healthy collection ⇒ nothing happens. Any change ⇒ a *replacement* collection is built first; if the build
fails, the old index and pointer remain untouched. Chunk IDs are deterministic: `<book>-p<page>-<n>-<sha1[:8]>`.

## Replacing / re-indexing the fixed book

1. Stop the API (`Ctrl+C`; or `docker compose stop api ui`).
2. Replace `data/books/selected_book.pdf`.
3. `python -m src.ingestion --detect-chapters-only`, review `chapters.json`, set `"reviewed": true`
   (a *reviewed* file from the old book would be applied to the new one, so delete or regenerate it first!).
4. `python -m src.ingestion` and read the report.
5. Re-run `python -m src.calibrate` with a new evaluation set; thresholds from the old book may not fit.
6. Start the API again.

## Known limitations

* Paragraph reconstruction from hard-wrapped lines is heuristic; some headings get merged with the first paragraph.
* Two-column layouts, tables and footnotes may be read in a jumbled order by pypdf.
* Excerpts shown to users are the **cleaned** text (whitespace-normalised), not byte-identical PDF text.
* Your PDF shows a duplicated passage on PDF page 6 in the extracted text layer; duplicates are de-duplicated at retrieval time.
