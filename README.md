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
RAG_FINAL_K=5
RAG_MAX_CONTEXT_CHARS=6000
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=64
VECTORSTORE_BATCH_SIZE=512
DENSE_INDEX_PROFILE=core
DENSE_PDF_MIN_YEAR=2024
DENSE_MAX_CHUNKS_PER_PDF=16
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
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
python src/retrieval.py --query "Quali corsi di laurea offre il DIEM?"
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
- deduplica per URL;
- rerank leggero basato sui metadati;
- rerank neurale opzionale con `BAAI/bge-reranker-v2-m3`.

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

1. **Chunking e embedding**: `src/chunking.py` produce chunk contestuali; `src/vector_store.py` indicizza in Chroma solo il profilo dense configurato da `DENSE_INDEX_PROFILE`, di default `core`.
2. **Candidate generation**: `src/retrieval.py` interroga BM25 e Chroma sia con la query originale sia con la query espansa. La query originale protegge nomi propri, sigle e codici; quella espansa migliora il richiamo sui domini noti.
3. **Fusione e filtri**: i ranking vengono fusi con RRF, poi corretti con segnali di metadati, fonte e freschezza. PDF e pagine storiche vengono penalizzati quando la domanda non chiede esplicitamente bandi, regolamenti, PDF o anni.
4. **Reranking neurale opzionale**: `src/reranking.py` passa i migliori candidati al cross-encoder `BAAI/bge-reranker-v2-m3` e combina score neurale e score ibrido.
5. **Generazione**: `src/rag_chain.py` chiede a Groq un JSON con risposta, fonti usate e citazioni inline. Se JSON mode fallisce, resta il fallback compatibile con `FONTI_USATE`.

Su Mac M1 con 8 GB il collo di bottiglia principale è spesso l'embedding durante la reindicizzazione, non solo il reranker. Per una build stabile usa una configurazione conservativa:

```env
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=4
VECTORSTORE_BATCH_SIZE=64
```

Se anche così la build è troppo lenta o instabile, usa un embedding più leggero per la fase di test:

```env
EMBEDDING_MODEL=intfloat/multilingual-e5-base
EMBEDDING_TRUNCATE_DIM=
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=8
VECTORSTORE_BATCH_SIZE=128
```

In entrambi i casi, dopo aver cambiato `EMBEDDING_MODEL` o `EMBEDDING_TRUNCATE_DIM`, ricrea Chroma con `python src/vector_store.py --reset`.

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
RETRIEVAL_RERANK_K=10
RERANKER_BATCH_SIZE=4
```

Per massima qualità offline, lasciare il reranker attivo e accettare più latenza. La scelta pratica è: reranker off durante sviluppo/UI live, reranker on per benchmark e demo ragionate.

## Valutazione retrieval

Il file `eval/golden_questions.jsonl` contiene domande di test con pattern URL attesi. Le query senza fonti attese, ad esempio fuori dominio o ricevimento generico, sono incluse per documentare i casi guardrail ma non entrano nelle metriche retrieval.

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

- `recall@5`: quota di domande in cui almeno una fonte corretta compare nei primi 5 risultati. È la metrica più importante per il chatbot, perché `RAG_FINAL_K` di default è 5.
- `recall@10`: come sopra, ma sui primi 10. Se è alto e `recall@5` è basso, il retriever trova la fonte ma il ranking va migliorato.
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
