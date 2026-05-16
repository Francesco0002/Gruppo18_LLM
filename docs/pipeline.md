# Pipeline del Corpus DIEM

Questo documento descrive la pipeline che prepara i documenti per la futura
fase di indexing del chatbot RAG.

## Obiettivo

La pipeline trasforma pagine e PDF ufficiali DIEM in file Markdown locali,
misurabili e pronti per chunking, embedding e indicizzazione vettoriale.

```text
discover.py
  -> discovered_urls.jsonl + raw_html/
scrape.py
  -> raw markdown HTML + markdown pulito + manifest.jsonl
extract_pdf.py
  -> raw markdown PDF + markdown pulito + manifest.jsonl
ingest.py
  -> marcatura duplicati + stats.json corrente + storico run
chunking.py
  -> chunks.jsonl + stats.json dei chunk
vector_store.py
  -> embedding dei chunk + Chroma vector store
retrieval.py
  -> BM25 + dense retrieval + fusione RRF dei risultati
rag_chain.py
  -> prompt RAG + generazione risposta con LLM locale
chatbot.py
  -> interfaccia CLI per interrogare il chatbot
```

## 1. Discovery

Comando:

```bash
python src/discover.py
```

Responsabilità:

- legge i seed da `data/urls.txt`;
- opzionalmente espande i seed tramite sitemap;
- visita URL HTML con BFS;
- applica filtri di scope e query rumorose;
- rispetta `robots.txt` se `respect_robots_txt: true`;
- salva HTML grezzo in `data/raw_html/<sh>/<hash>.html`;
- scrive `data/discovered_urls.jsonl`;
- salva `data/checkpoint.json` per riprendere un crawl interrotto;
- salva `data/discovery_state.json` per mantenere tra run la frontier non ancora
  visitata e la memoria degli URL/documenti già noti;
- applica `refresh_after_days` per ricontrollare periodicamente URL già visti
  senza consumare sempre il budget sui medesimi URL recenti.

`checkpoint.json` è uno stato intra-run: se il processo si interrompe, permette
di riprendere la stessa esecuzione. Salva queue, contatori del run e solo il
delta di memoria rispetto allo stato persistente iniziale (`new_known_urls` e
`new_known_documents`). `discovery_state.json` è invece uno stato inter-run:
conserva la frontier residua e la memoria cumulativa necessaria a proseguire la
copertura in run successive. I limiti `max_total_urls` e `per_domain_limits`
restano budget del singolo run, non contatori globali.
`discovered_urls.jsonl` resta lo snapshot degli URL prodotti dalla run corrente:
serve come input immediato agli step di scraping HTML ed estrazione PDF, mentre
la memoria cumulativa vive in `discovery_state.json`.

Il checkpoint usa `status="in_progress"` durante il run e
`status="completed"` a chiusura. Solo il checkpoint finale riporta anche
`stop_reason`, così si distingue tra arresto per `max_total_urls`, coda
esaurita (`queue_exhausted`) e coda senza URL più eleggibili
(`no_eligible_urls`).

Quando una run non ha URL eleggibili, le statistiche della discovery riportano
anche `skipped_by_reason`: ad esempio `recently_known` segnala URL già noti e
ancora dentro la finestra `refresh_after_days`, quindi esclusi senza fetch nella
run corrente.

Le statistiche in `data/processed/stats.json` includono anche
`discovery.coverage`, che misura la copertura cumulativa della BFS per singola
depth fino a `max_depth`. Per ogni livello espone gli URL ancora pendenti a
quella profondità (`pending_at_depth`), quelli ancora pendenti a profondità
inferiori (`pending_below_depth`), se il livello è già sigillato (`sealed`) e
se è completo (`complete`). Una depth è completa solo quando non restano URL
pendenti né a quel livello né sotto di esso; il verdetto globale è `complete`
solo se tutte le depth configurate sono complete e la run termina con
`queue_exhausted`.

Lo storico di ciascuna run conserva anche
`frontier_before_by_depth`, `frontier_after_by_depth` e
`processed_html_by_depth`, così run successive con lo stesso `max_depth`
mostrano se una frontiera si sta svuotando o se continuano a emergere nuovi URL.
I PDF restano inclusi nelle statistiche generali, ma non riaprono la BFS HTML:
quando sono linkati da una pagina al limite possono comparire a
`depth = max_depth + 1`.

I PDF linkati dagli HTML vengono registrati subito in
`discovered_urls.jsonl`. Se `robots.txt` consente il download, ricevono
`status="pending_download"`; altrimenti ricevono `status="robots_denied"`.

## 2. Scraping HTML

Comando:

```bash
python src/scrape.py
```

Responsabilità:

- legge gli HTML `status="ok"` e `indexable=true` da `discovered_urls.jsonl`;
- legge il file HTML grezzo già salvato, senza riscaricare la pagina;
- genera Markdown raw con Crawl4AI;
- salva il raw in `data/processed/raw_markdown/<sh>/<hash>.md`;
- pulisce boilerplate conservativo e aggiunge front matter YAML;
- salva il Markdown indicizzabile in `data/processed/markdown/<sh>/<hash>.md`;
- aggiunge record a `data/processed/manifest.jsonl`;
- salta URL HTML già processati negli ultimi 7 giorni.

## 3. Estrazione PDF

Comando:

```bash
python src/extract_pdf.py
```

Responsabilità:

- legge PDF `status="pending_download"` da `discovered_urls.jsonl`;
- scarica i PDF in `data/raw_pdf/<sh>/<hash>.pdf`;
- converte i PDF in Markdown raw con `pymupdf4llm`;
- salva il raw in `data/processed/raw_markdown/<sh>/<hash>.md`;
- normalizza artefatti PDF evidenti e aggiunge front matter YAML;
- salva il Markdown indicizzabile in `data/processed/markdown/<sh>/<hash>.md`;
- aggiunge record a `data/processed/manifest.jsonl`;
- salta PDF già processati negli ultimi 7 giorni.

I PDF `robots_denied` restano tracciati in `discovered_urls.jsonl`, ma non
vengono scaricati.

## 4. Ingest

Comando completo:

```bash
python src/ingest.py
```

Comando per rigenerare solo duplicati e statistiche:

```bash
python src/ingest.py --stats-only
```

Responsabilità:

- coordina discovery, scraping HTML ed estrazione PDF;
- marca documenti duplicati tramite `content_hash`, considerando lo stato
  corrente del manifest;
- genera `data/processed/stats.json`;
- salva una copia storica in `data/processed/runs/<crawl_run_id>/stats.json`.

`ingest.py` non decide i limiti del crawl: i parametri effettivi sono sempre
letti da `config.yaml`.

`data/processed/manifest.jsonl` resta append-only: se un URL fallisce e poi
riesce in una run successiva, conserva entrambe le righe. Le statistiche finali
usano invece solo l'ultimo record disponibile per ogni `source+url`, così
`failed` vecchi non falsano lo stato attuale del corpus.

La futura fase di indexing userà la stessa regola tramite
`pipeline_io.latest_records_by_url()`: leggerà lo storico `manifest.jsonl`,
terrà solo lo stato corrente di ogni documento, poi indicizzerà solo record
`status="ok"`, `indexable=true`, `text_extracted=true`, non duplicati e con
`index_markdown_path` presente.

## 5. Chunking

Comando:

```bash
python src/chunking.py
```

Responsabilità:

- legge data/processed/manifest.jsonl;
- usa `pipeline_io.latest_records_by_url()` per considerare solo lo stato corrente del manifest append-only;
- seleziona solo documenti con status="ok", indexable=true, text_extracted=true, non duplicati e con index_markdown_path presente;
legge i Markdown puliti da data/processed/markdown/;
- rimuove il front matter YAML iniziale, perché i metadati sono già presenti nel manifest;
- rimuove eventuali header contestuali già presenti per evitare duplicazioni;
- normalizza piccoli artefatti testuali prima del chunking;
- divide i documenti prima per sezioni Markdown e poi, quando necessario, con split a dimensione controllata;
- aggiunge a ogni chunk un header contestuale con titolo, breadcrumb e fonte;
- salva i chunk in `data/processed/chunks/chunks.jsonl`;
- salva statistiche del chunking in `data/processed/chunks/stats.json`.

Ogni chunk conserva il testo da indicizzare insieme ai metadati principali del documento sorgente, tra cui chunk_id, document_hash, source_url, title, breadcrumb, chunk_index, text_hash e numero di caratteri.

La fase successiva usa `chunks.jsonl` per generare gli embedding e popolare il vector store.

## 6. Vector Store

Creazione del vector store:

```bash
python src/vector_store.py --reset
```

Query di test:
```bash
python src/vector_store.py --query "Quali corsi di laurea offre il DIEM?"
```
Responsabilità:

- legge `data/processed/chunks/chunks.jsonl`;
- converte ogni chunk in un documento LangChain con testo e metadati;
- genera gli embedding tramite un modello HuggingFace multilingua;
- indicizza i vettori nel database Chroma;
- salva il vector store in `data/vectorstore/chroma/`;
- salva statistiche in `data/vectorstore/stats.json`;
- consente query semantiche di test tramite parametro --query.

Il vector store non deve essere versionato su Git perché è un artefatto generato localmente.

## 7. Retrieval Ibrido

Comando:

```bash
python src/retrieval.py --query "Quali corsi di laurea offre il DIEM?"
```
Comando con numero finale di risultati personalizzato:

```bash
python src/retrieval.py --query "Quali sono gli orari di ricevimento del professor Mario Vento?" --final-k 3
```
Responsabilità:

- legge i chunk da `data/processed/chunks/chunks.jsonl`;
- esegue retrieval lessicale tramite BM25;
- esegue dense retrieval tramite il vector store Chroma creato da `vector_store.py`;
- combina i risultati dei due retriever tramite Reciprocal Rank Fusion;
- rimuove risultati ridondanti provenienti dallo stesso URL;
- applica un rerank leggero basato sui metadati e sul tipo di query;
- stampa i chunk finali con titolo, URL, breadcrumb, chunk ID e anteprima del contenuto.

Il retrieval ibrido migliora la robustezza rispetto alla sola ricerca vettoriale:

- BM25 è utile per nomi propri, docenti, sigle, codici corso, URL e parole chiave esatte;
- il dense retrieval è utile per domande formulate in linguaggio naturale e semanticamente simili ai documenti;
- la fusione RRF evita di confrontare direttamente score eterogenei prodotti da BM25 e Chroma.

Il rerank leggero sui metadati gestisce alcuni casi frequenti:

- domande su corsi di laurea e offerta formativa;
- domande sugli orari di ricevimento di un docente specifico;
- domande generiche sulla didattica;
- penalizzazione di pagine in lingua inglese quando la query è in italiano;
- penalizzazione di pagine relative ad anni accademici vecchi se la query non specifica un anno;
- deduplica dei risultati provenienti dallo stesso URL.

Questa fase non genera ancora la risposta finale: produce i chunk più rilevanti che saranno poi forniti al modello LLM nella fase RAG completa.

## 8. RAG Generation

Comando:

```bash
python src/chatbot.py
```

Comando con modello e numero di chunk personalizzati:

```bash
python src/chatbot.py --model llama3.2:3b --final-k 3
```

Responsabilità:

- riceve una domanda utente da terminale;
- usa `retrieval.py` per recuperare i chunk più rilevanti dal corpus DIEM;
- costruisce un contesto compatto usando titolo, URL, breadcrumb, chunk ID e contenuto dei chunk;
- genera un prompt RAG vincolato alle fonti recuperate;
- invia il prompt a un modello instruct locale tramite Ollama;
- restituisce una risposta in italiano insieme alle fonti utilizzate.

La generazione avviene tramite il modulo `src/rag_chain.py`, che implementa la pipeline:

```text
domanda utente
  -> hybrid retrieval
  -> costruzione contesto
  -> prompt RAG
  -> chiamata Ollama
  -> risposta + fonti
```

Il prompt impone al modello di usare esclusivamente il contesto fornito.
Se il contesto non contiene informazioni sufficienti, il chatbot deve dichiarare che l'informazione non è disponibile nelle fonti DIEM indicizzate.
Se la domanda è fuori dominio rispetto al DIEM, il chatbot deve segnalarlo invece di produrre una risposta non fondata.

Le variabili principali sono configurate tramite `.env`:

```env
OLLAMA_MODEL=llama3.2:3b
OLLAMA_ENDPOINT=http://localhost:11434/api/generate
RAG_FINAL_K=3
RAG_MAX_CONTEXT_CHARS=6000
```

Questa fase completa la pipeline RAG end-to-end: i chunk recuperati dal retrieval ibrido vengono usati come contesto per generare una risposta controllata e accompagnata dalle fonti.

## Output Principali

| File / cartella | Contenuto |
|---|---|
| `data/discovered_urls.jsonl` | Manifest della discovery URL. |
| `data/raw_html/` | HTML grezzo indicizzabile. |
| `data/raw_pdf/` | PDF scaricati. |
| `data/processed/raw_markdown/` | Markdown estratto prima della pulizia. |
| `data/processed/markdown/` | Markdown pulito e indicizzabile da HTML e PDF. |
| `data/processed/manifest.jsonl` | Manifest append-only dei documenti processati. |
| `data/processed/stats.json` | Ultime statistiche generate sullo stato corrente. |
| `data/processed/runs/<crawl_run_id>/stats.json` | Copia storica delle statistiche di un run. |
| `data/checkpoint.json` | Stato per riprendere la discovery. |
| `data/processed/chunks/chunks.jsonl` | Chunk contestuali pronti per embedding e indicizzazione vettoriale. |
| `data/processed/chunks/stats.json` | Statistiche del chunking: numero chunk, lunghezze, domini e documenti più frammentati. |
| `data/vectorstore/chroma/` | Vector store Chroma generato dagli embedding dei chunk. |
| `data/vectorstore/stats.json` | Statistiche del vector store e modello embedding usato. |

## Status Discovery

| Status | Significato |
|---|---|
| `ok` | HTML indicizzabile salvato su disco. |
| `not_indexable` | HTML in scope ma escluso dall'indice. |
| `pending_download` | PDF rilevato e scaricabile nella fase PDF. |
| `robots_denied` | Risorsa rilevata ma bloccata da `robots.txt`. |
| `duplicate_redirect` | Redirect verso documento già visto. |
| `duplicate_canonical` | Canonical già visto. |
| `redirected_out_of_scope` | Redirect fuori scope. |
| `failed` | Errore HTTP o di rete. |

## Scope URL

| Dominio / risorsa | Regola |
|---|---|
| `www.diem.unisa.it` | Ammesso. |
| `docenti.unisa.it` | Solo se scoperto da DIEM o da profilo docente in scope. |
| `corsi.unisa.it` | Solo percorsi corso/codici DIEM configurati. |
| `uploads` PDF | Rilevati; scaricati solo se `robots.txt` lo permette. |

## Note Operative

Ogni record processed conserva `raw_markdown_path` e `index_markdown_path`.
I campi di qualità
(`clean_status`, `clean_warnings`, `raw_markdown_chars`,
`clean_markdown_chars`, `removed_chars_ratio`) descrivono quanto è stata
modificata l'estrazione raw.

I Markdown puliti sotto 100 caratteri restano tracciati nel manifest, ma sono
marcati `text_extracted=false` e `indexable=false` per evitare chunk di sola
navigazione o pagine tecniche quasi vuote.

Per ripartire da zero:

```bash
make clean
python src/ingest.py
```

Per continuare incrementalmente:

```bash
python src/ingest.py
```

Per aggiornare solo statistiche e duplicati:

```bash
python src/ingest.py --stats-only
```
