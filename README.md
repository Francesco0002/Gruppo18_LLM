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
  legacy/                 Prototipi non più usati

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
profondità, rate limit e path degli output.

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

Query di test sul vector store:

```bash
python src/vector_store.py --query "Quali corsi di laurea offre il DIEM?"
```

Retrieval ibrido BM25 + dense:

```bash
python src/retrieval.py --query "Quali corsi di laurea offre il DIEM?"
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
`data/processed/raw_markdown/` conserva l'estrazione originale per debug;
`data/processed/chunks/` contiene i chunk contestuali prodotti per embedding e retrieval;
`data/vectorstore/` contiene il vector store Chroma generato localmente.

Il modulo `src/retrieval.py` implementa il retrieval ibrido combinando:
- BM25, per ricerca lessicale basata su parole chiave;
- dense retrieval, tramite embedding e Chroma;
- Reciprocal Rank Fusion, per fondere i ranking;
- deduplica per URL;
- rerank leggero basato sui metadati, utile per favorire pagine pertinenti in base alla query.

`config.yaml`, `.env`, virtual environment, file in `data/`, indici e API key
non devono essere versionati su Git.
