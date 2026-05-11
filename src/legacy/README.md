# Legacy

Codice prototipale conservato come riferimento storico, non più usato
nella pipeline corrente.

- `crawler.py` — primo prototipo del crawler basato su `crawl4ai`.
  La pipeline attuale parte da `src/discover.py` e usa `httpx` +
  filtri custom in `src/url_filters.py`. Questo file è qui solo per
  consultazione: non importarlo dal codice nuovo.
