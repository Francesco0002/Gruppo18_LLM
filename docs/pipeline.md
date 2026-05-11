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
  -> markdown HTML + manifest.jsonl
extract_pdf.py
  -> markdown PDF + manifest.jsonl
ingest.py
  -> marcatura duplicati + stats.json corrente + storico run
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
- salva `data/checkpoint.json` per riprendere un crawl interrotto.

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
- genera Markdown pulito con Crawl4AI;
- salva Markdown in `data/processed/markdown/<sh>/<hash>.md`;
- salva un campione di Markdown raw in `data/processed/markdown_raw/`;
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
- converte i PDF in Markdown con `pymupdf4llm`;
- salva Markdown in `data/processed/markdown/<sh>/<hash>.md`;
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
`status="ok"`, non duplicati e con `markdown_path` presente.

## Output Principali

| File / cartella | Contenuto |
|---|---|
| `data/discovered_urls.jsonl` | Manifest della discovery URL. |
| `data/raw_html/` | HTML grezzo indicizzabile. |
| `data/raw_pdf/` | PDF scaricati. |
| `data/processed/markdown/` | Markdown finale da HTML e PDF. |
| `data/processed/manifest.jsonl` | Manifest append-only dei documenti processati. |
| `data/processed/stats.json` | Ultime statistiche generate sullo stato corrente. |
| `data/processed/runs/<crawl_run_id>/stats.json` | Copia storica delle statistiche di un run. |
| `data/checkpoint.json` | Stato per riprendere la discovery. |

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
