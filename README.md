# Gruppo18_LLM

Chatbot RAG per rispondere a domande sulle informazioni ufficiali del DIEM
usando pagine web e documenti pubblici del sito UNISA/DIEM.

## Struttura

```text
src/
  discover.py             Orchestratore BFS della discovery URL
  discovery_processor.py  Processing del singolo URL scoperto
  discovery_fetch.py      HTTP, rate limit, robots.txt e sitemap
  discovery_io.py         Config, checkpoint e output discovery
  discovery_models.py     Dataclass condivise
  html_utils.py           Utility HTML
  url_filters.py          Regole di scope e filtri URL
  scrape.py               HTML raw -> Markdown raw + pulito
  extract_pdf.py          PDF -> Markdown raw + pulito
  ingest.py               Pipeline completa + duplicati + stats
  pipeline_io.py          Utility comuni per JSONL, hash e file
  chunking.py             Markdown pulito -> chunk contestuali per RAG
  vector_store.py         Chunk -> embedding -> Chroma vector store
  retrieval.py            Retrieval ibrido BM25 + dense + RRF
  rag_chain.py            Pipeline RAG: retrieval + prompt + chiamata LLM
  app.py                  Interfaccia grafica Chainlit
  chatbot_cli.py          Chatbot CLI da terminale
  legacy/                 Prototipi non più usati

.chainlit/              Configurazione interfaccia Chainlit
public/                 Asset UI Chainlit: logo, favicon, avatar

data/                     Dati locali e output del crawl
docs/pipeline.md          Dettaglio della pipeline dati
config.example.yaml       Template di configurazione
Makefile                  Shortcut dei comandi principali
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml
cp .env.example .env
```

Modifica `config.yaml` per impostare limiti di crawl, domini ammessi,
profondità, rate limit, refresh degli URL già noti e path degli output.

Modifica `.env` per impostare la chiave API Groq e il modello usato nella fase RAG:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=qwen/qwen3-32b
GROQ_TIMEOUT_SECONDS=60
GROQ_MAX_RETRIES=3
GROQ_JSON_MODE=true
RAG_FINAL_K=10
RAG_MAX_CONTEXT_CHARS=12000
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_BACKEND=torch
EMBEDDING_ONNX_MODEL_PATH=
EMBEDDING_ONNX_PROVIDER=CPUExecutionProvider
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=16
VECTORSTORE_BATCH_SIZE=256
DENSE_INDEX_PROFILE=core
DENSE_PDF_MIN_YEAR=2020
DENSE_MAX_CHUNKS_PER_PDF=16
RETRIEVAL_RERANK_K=20
RERANKER_ENABLED=false
RERANKER_MODEL=mixedbread-ai/mxbai-rerank-base-v2
RERANKER_FALLBACK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANKER_WEIGHT=0.60
HYBRID_WEIGHT=0.40
RERANKER_BATCH_SIZE=4
```
Per usare la generazione RAG è necessario disporre di una API key Groq valida.

## Comandi Principali

Discovery:

```bash
python src/discover.py
```

Conversione HTML e PDF in Markdown raw e Markdown pulito:

```bash
python src/scrape.py
python src/extract_pdf.py
```

Pipeline completa:

```bash
python src/ingest.py
```

Solo marcatura duplicati e statistiche su dati già prodotti:

```bash
python src/ingest.py --stats-only
```

Chunking dei Markdown puliti:

```bash
python src/chunking.py
```

Creazione del vector store Chroma:

```bash
python src/vector_store.py --reset
```

Dopo aver cambiato `EMBEDDING_MODEL`, ricrea sempre il vector store con `--reset`.

Query di test sul vector store:

```bash
python src/vector_store.py --query "Quali corsi di laurea offre il DIEM?"
```

Retrieval ibrido BM25 + dense:

```bash
python src/retrieval.py --query "Quali corsi di laurea offre il DIEM?" --final-k 5
```

Chatbot RAG da terminale:

```bash
python src/chatbot_cli.py
```

Esempio con modello Groq e numero di chunk personalizzato:

```bash
python src/chatbot_cli.py --model qwen/qwen3-32b --final-k 3
```

Interfaccia grafica Chainlit:

```bash
chainlit run src/app.py -w
```

Ripartenza pulita:

```bash
make clean
python src/ingest.py
```



## Documentazione

La descrizione completa di flusso dati, output, status, gestione PDF,
`robots.txt`, checkpoint e duplicati è in:

```text
docs/pipeline.md
```

## Note

`data/processed/markdown/` contiene il Markdown pulito da indicizzare;
`data/processed/markdown_raw/` conserva l'estrazione originale per debug;
`data/processed/chunks/` contiene i chunk contestuali prodotti per embedding e retrieval;
`data/vectorstore/` contiene il vector store Chroma generato localmente.

Il modulo `src/retrieval.py` implementa il retrieval ibrido combinando:
- BM25, per ricerca lessicale basata su parole chiave;
- dense retrieval, tramite embedding e Chroma;
- Reciprocal Rank Fusion, per fondere i ranking;
- candidate pool allargato su query originale ed espansa;
- deduplica per URL o per entità strutturata, per esempio `publication_id`;
- routing per famiglie informative del sito: docenti, ricevimento, pubblicazioni, laboratori/strumentazione, corsi, accesso, statistiche, Erasmus, dottorati, bandi/regolamenti, news e contatti;
- rerank leggero basato su metadata, tipo chunk ed entità;
- rerank neurale opzionale con `mixedbread-ai/mxbai-rerank-base-v2` o fallback veloce `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

Il modulo `src/rag_chain.py` implementa la pipeline RAG completa:
- recupera i chunk più rilevanti tramite `retrieval.py`;
- costruisce un contesto compatto da passare al modello LLM;
- genera un prompt vincolato alle fonti DIEM;
- chiama un modello LLM tramite API Groq;
- interpreta gli indici dei documenti usati dal modello;
- restituisce risposta e fonti utilizzate.

Il modulo `src/chatbot_cli.py` fornisce una semplice interfaccia da terminale per interrogare il chatbot.
La generazione è vincolata al contesto recuperato: se le fonti non contengono informazioni sufficienti,
il chatbot deve dichiararlo invece di inventare una risposta.
Il modulo `src/app.py` fornisce l’interfaccia grafica Chainlit.

## Pipeline RAG aggiornata

La pipeline ora lavora in più stadi:

1. **Chunking multi-rappresentazione**: `src/chunking.py` salva `body_text`, `locator_text` e `text_for_display`. L'embedding primario usa solo il corpo, mentre il locator contiene metadati compatti come corso, anno, curriculum, docente, fonte e topic.
2. **Vector store body + locator**: `src/vector_store.py` crea due collection Chroma: `diem_knowledge` per il contenuto e `diem_knowledge_locator` per i locator compatti. Dopo questo aggiornamento va eseguito `python src/vector_store.py --reset`.
3. **Candidate generation evidence-first**: `src/retrieval.py` combina BM25, dense body, dense locator ed evidenze strutturate da `src/structured_evidence.py`, guidate dal planner leggero `src/query_planner.py`.
4. **Fusione senza boost opachi**: i ranking vengono fusi con RRF; le evidenze strutturate vengono solo portate in testa quando la query richiede chiaramente tabelle/listati ufficiali, senza generare risposte predefinite.
5. **Reranking neurale opzionale**: `src/reranking.py` passa i migliori candidati al cross-encoder e combina score neurale e score ibrido in modo più conservativo.
6. **Generazione**: `src/rag_chain.py` usa sempre il retrieval principale. Le risposte estrattive per ricevimento/pubblicazioni sono post-processing sulle evidenze recuperate, non bypass del retrieval.

`EMBEDDING_MAX_SEQ_LENGTH` resta un limite di sicurezza per modelli a contesto lungo su MPS. Il chunking spezza le schede sintetiche lunghe a monte, quindi nel corpus corrente il limite a 1024 non tronca i chunk generati.

In entrambi i casi, dopo aver cambiato `EMBEDDING_MODEL`, `EMBEDDING_TRUNCATE_DIM`, `EMBEDDING_MAX_SEQ_LENGTH` o lo schema dei chunk, ricrea Chroma con `python src/vector_store.py --reset`.

Su Mac M1 puoi usare il Granite 97M esportato in ONNX quantizzato ARM64:

```bash
pip install 'sentence-transformers[onnx]' onnx
python src/export_embedding_onnx.py \
  --model ibm-granite/granite-embedding-97m-multilingual-r2 \
  --output models/granite-embedding-97m-multilingual-r2-onnx-arm64-int8
```

Poi imposta:

```env
EMBEDDING_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2
EMBEDDING_BACKEND=onnx
EMBEDDING_ONNX_MODEL_PATH=models/granite-embedding-97m-multilingual-r2-onnx-arm64-int8
EMBEDDING_ONNX_PROVIDER=CPUExecutionProvider
EMBEDDING_DEVICE=cpu
```

Per ONNX quantizzato su Apple Silicon è preferibile `CPUExecutionProvider`: MPS non è usato da ONNX Runtime e CoreML può introdurre tempi di compilazione o incompatibilità con modelli int8.

Il profilo `DENSE_INDEX_PROFILE=core` riduce il vector store dense: HTML, catalogo corsi, docenti/rubrica entrano sempre; i PDF entrano solo se recenti, regolamenti o bandi recenti. Per i PDF inclusi, `DENSE_MAX_CHUNKS_PER_PDF` limita quanti chunk entrano in Chroma. BM25 continua comunque a leggere tutti i chunk, quindi i PDF esclusi dal dense index non spariscono dalla pipeline.

Profili disponibili:

```env
DENSE_INDEX_PROFILE=core   # consigliato
DENSE_INDEX_PROFILE=no_pdf # massimo risparmio: nessun PDF in Chroma
DENSE_INDEX_PROFILE=all    # vecchio comportamento: tutto in Chroma
```

Il reranker neurale può essere pesante soprattutto al primo avvio. Per una demo fluida:

```env
RERANKER_ENABLED=false
```

Per una via intermedia su M1:

```env
RERANKER_ENABLED=true
RETRIEVAL_RERANK_K=20
RERANKER_DEVICE=auto
RERANKER_BATCH_SIZE=1
```

Con `mixedbread-ai/mxbai-rerank-base-v2`, `RERANKER_DEVICE=auto` usa Apple MPS quando disponibile. Su Mac con 8 GB, mantieni `RERANKER_BATCH_SIZE=1` e aumenta solo dopo un benchmark locale.

Per massima qualità compatibile con la macchina, lascia `RETRIEVAL_RERANK_K=20` e usa il reranker solo per benchmark o demo ragionate. La scelta pratica è: reranker off durante sviluppo/UI live, reranker on quando vuoi misurare la qualità finale.

## Valutazione retrieval

Il file `eval/golden_questions.jsonl` contiene domande di test con pattern URL attesi. Le query senza fonti attese, ad esempio fuori dominio o ricevimento generico, sono incluse per documentare i casi guardrail ma non entrano nelle metriche retrieval.

Il file `eval/golden_questions_topic_coverage.jsonl` aggiunge una suite multi-topic per piani di studio, corsi, docenti, ricevimento, pubblicazioni, ricerca, terza missione, international, dottorati, laboratori, documenti e qualità/statistiche.

Per controllare che manifest e chunk coprano tutte le aree core:

```bash
python src/coverage_audit.py --fail-on-gaps
```

Esegui una valutazione veloce della configurazione corrente:

```bash
python src/evaluate_retrieval.py --current-only
```

Esegui il confronto A/B tra pipeline senza e con reranker:

```bash
python src/evaluate_retrieval.py
```

Output generati:

```text
eval/retrieval_report.json
eval/retrieval_report.md
```

Metriche:

- `recall@5`: quota di domande in cui almeno una fonte corretta compare nei primi 5 risultati.
- `recall@10`: come sopra, ma sui primi 10. Con `RAG_FINAL_K=7`, aiuta a capire se la fonte è vicina al contesto passato al chatbot anche quando non è nei primissimi risultati.
- `mrr@10`: premia fonti corrette molto in alto. Valore vicino a 1 significa che la fonte corretta è spesso prima.
- `ndcg@5` e `ndcg@10`: misurano la qualità dell'ordine dei risultati, non solo la presenza di almeno una fonte corretta.
- `evaluated`: numero di domande con fonte attesa.
- `skipped`: domande senza fonte attesa, escluse dalle metriche retrieval.

Interpretazione consigliata:

- Se `recall@5` migliora con il reranker, il cross-encoder sta aiutando davvero.
- Se `recall@10` resta stabile ma `ndcg@5` migliora, il reranker sta riordinando meglio i candidati.
- Se `recall@10` cala, il problema non è il reranker ma la candidate generation o la deduplica.
- Se le query docenti/orari falliscono, controllare prima se le pagine `docenti.unisa.it/.../home` sono nel corpus: il reranker non può recuperare documenti non indicizzati.

`config.yaml`, `.env`, virtual environment, file in `data/`, indici e API key
non devono essere versionati su Git.
