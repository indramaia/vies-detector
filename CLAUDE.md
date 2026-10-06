# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Viés Detector — end-to-end pipeline that detects and communicates editorial bias in Brazilian news outlets, using a fine-tuned BERTimbau model, a curated ideological spectrum, and a public REST API. TCC (thesis) project — USP MBA IA & Big Data (Indra Seixas Neiva, 2026). Code comments, logs, and docstrings are in Portuguese; keep new contributions consistent with that unless told otherwise.

Live demo: biasradar.lovable.app · API: vies-detector.onrender.com · Model: huggingface.co/IndraSeixas/bertimbau-bias

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab

# DB (defaults to local sqlite:///vies_detector.db if DATABASE_URL unset)
python scripts/setup_db.py

# Run the full pipeline once (collect → classify → aggregate → contextualize)
python scripts/run_pipeline.py

# Run the API locally
python api/app.py

# Tests (pytest config lives in pyproject.toml: testpaths=tests, coverage on by default)
pytest
pytest tests/test_aggregation.py
pytest tests/test_aggregation.py::test_name -v

# Train/fine-tune the classifier (needs data/factnews.csv — not in repo)
# Must run as a module (-m), not as a script path: train.py uses a relative
# import (`from .model_loader import ...`) that only resolves when the
# classifier package is loaded, which `python classifier/train.py` does not do.
python -m classifier.train --data data/factnews.csv --output models/bertimbau-bias
python -m classifier.train --data data/factnews.csv --output models/bertimbau-bias --seeds 42 123 456

# Regenerate the ideological spectrum chart (docs/spectrum.png)
python scripts/generate_spectrum_chart.py

# One-off article analysis (same path used by the analyze_url.yml workflow)
python scripts/analyze_single_url.py --url <url> --url_hash <sha256>

# One-time SQLite → Neon Postgres migration
python scripts/migrate_sqlite_to_neon.py
```

There is no separate lint command configured; match existing style (docstring-style module headers with `─` divider lines, Portuguese comments explaining *why*).

## Architecture

Five layers, each its own top-level package, run in sequence by `pipeline/main_flow.py::run_pipeline()`:

```
collector → classifier → aggregation → ideological → api
  RSS+SHA256   BERTimbau    BiasScore     [-1,+1]     Flask+cache
```

- **`collector/`** — `sources.py` (RSS/homepage-scrape catalog per outlet), `rss_fetcher.py` (parallel per-article fetch/scrape orchestration), `article_scraper.py` (requests+BeautifulSoup HTML scraping), `deduplicator.py` (SHA-256 of URL, dedup against DB hashes), `preprocessor.py` (NLTK Punkt sentence segmentation + boilerplate cleanup). LGPD constraint: only a ≤500-char snippet is ever stored, never the full article body.
- **`classifier/`** — `model_loader.py` loads BERTimbau fine-tuned as a process-wide singleton (`@lru_cache`), falling back to downloading `IndraSeixas/bertimbau-bias` from the HuggingFace Hub when no local `models/bertimbau-bias` exists. `sentence_classifier.py` batches inference + softmax. `train.py` is the fine-tuning script — standalone offline tool, not invoked by the pipeline. `WeightedTrainer` supports a `--loss {ce,focal}` flag (both using the same inverse-frequency `CLASS_WEIGHTS`) to switch between weighted CrossEntropy and Focal Loss for class-imbalance mitigation, plus label smoothing, post-hoc threshold calibration (`tune_thresholds`), and optional multi-seed averaging. A controlled same-seed comparison found Focal Loss does **not** outperform weighted CE on this corpus once both are threshold-calibrated (CE: Macro-F1 0.825 vs Focal: 0.819) — CE ponderada + threshold calibration is the chosen config (see README.md "Mitigação de Desbalanceamento").
- **`aggregation/`** — `bias_score.py` computes `BiasScore = Σ(CLASS_WEIGHT[label] × rs_factor) / n_sentences` ∈ [0,2] per article (weights: factual=0, enviesada=1, fortemente_enviesada=2), with reported-speech sentences (quotes, attribution verbs like "disse/afirmou") discounted via `rs_factor=0.4` since the bias there belongs to the quoted source, not the outlet. Only the first 100 sentences of an article are classified. `window_aggregator.py` rolls per-outlet stats over a time window (default 30 days). `topic_clusterer.py` does TF-IDF clustering for `/api/stories`, `/api/topics/<slug>`, and `/api/articles/<hash>/similar`.
- **`ideological/`** — `reference_map.py` loads a curated, static `data/ideological_references.json` (score [-1,+1] per outlet, sourced from Manchetômetro, GPOPAI/USP, Intervozes — see README for full methodology and literature). `spectrum.py` builds `IdeologicalContext` combining BiasScore with that static ideology score, including an ethics caveat. This score is **not** model-computed; it's a hand-curated reference map.
- **`pipeline/main_flow.py`** — orchestrates everything as discrete `task_*` functions (no Prefect despite old docstring references; `prefect` is commented out of requirements.txt). Key behaviors to preserve when touching this file:
  - Runs `task_purge_old_data` (deletes articles/sentences older than `_RETENTION_DAYS=45`) and reads existing hashes in one short-lived DB session, closes it, *then* runs the multi-minute RSS collection — an open connection during collection gets killed by Neon on idle timeout.
  - `task_update_home_summary` keeps `total_articles`/`total_sentences`/`total_vehicles` as **cumulative counters** (adds deltas) rather than `COUNT(*)`, specifically because retention purges would otherwise make these showcase metrics regress on the frontend.
  - Uses `bulk_insert_mappings` instead of ORM objects to cut DB round-trips from ~10k to 2 per run.
- **`api/app.py`** — single-file Flask app. Core design is resilience against Render free-tier cold starts (15 min idle → sleep, 30-90s cold start):
  1. In-memory per-worker TTL cache (`_CACHE` dict) — vehicles/stats/spectrum TTL 12h (tied to the pipeline's 12h cadence), stories/topics 30 min, articles 30 min.
  2. Stale-While-Revalidate (`_serve_swr`): expired cache is still served immediately while a background thread refreshes it (at most one refresh thread per key via `_SWR_IN_PROGRESS`).
  3. Static fallback from `ideological_references.json` when the DB is empty/unreachable (`_fallback_vehicles`), so the frontend never sees an empty list.
  4. `/api/warmup` is the keep-alive target (pinged every 14 min by `keepalive.yml`) — it only touches Neon if a cache entry is actually expired, to conserve Neon's free-tier compute quota.
  - `POST /api/analyze` / `GET /api/analyze/<hash>` implement on-demand single-URL analysis: it registers an `OnDemandRequest` row and dispatches the `analyze_url.yml` GitHub Actions workflow via the GitHub API (needs `GH_TOKEN`), then the client polls for completion.
  - Only `api/`, `aggregation/`, `ideological/`, `scripts/` (not `collector/`, `classifier/`, `pipeline/`) are shipped to production (see Dockerfile/render.yaml `buildFilter`) — the API never runs model inference itself, only reads what the pipeline already computed.

### Database (`scripts/setup_db.py`)

SQLAlchemy models, `DATABASE_URL` env var selects backend (defaults to local `sqlite:///vies_detector.db`; production uses Neon Postgres). `PIPELINE_MODE=1` switches the engine to `NullPool` (fresh TCP connection per `get_session()`, used by one-shot GitHub Actions scripts) vs. the API's `QueuePool` with keepalives. Tables: `articles` (metadata + BiasScore, no full article text — LGPD), `sentences` (per-sentence label/scores, FK to articles), `vehicle_indices` (one row per outlet, upserted each pipeline run), `home_summary` (single cumulative-totals row, id=1), `on_demand_requests` (lifecycle for `/api/analyze`).

### Operational harness

There is no server/cron the project owns — every recurring piece of work is a free-tier external service pinging another free-tier external service, and most changes in this repo's history are tuning those services against each other's quotas rather than application features. The moving parts:

- **`.github/workflows/pipeline.yml`** — the actual collect→classify→aggregate flow (`run_pipeline.py`), on a GitHub Actions cron (`0 */12 * * *`, i.e. every 12h) plus `workflow_dispatch` for manual runs. Each run downloads the model fresh from the HF Hub — there's no persistent runner. `PIPELINE_MODE=1` is set here so DB connections use `NullPool`.
- **`.github/workflows/analyze_url.yml`** — same shape, but triggered externally via `workflow_dispatch` by `api/app.py`'s `POST /api/analyze` (through the GitHub REST API, needs a `GH_TOKEN` with Actions write scope), for one-off single-URL analysis outside the 12h cadence.
- **`.github/workflows/keepalive.yml`** — a GitHub Actions cron (`*/14 * * * *`) that curls `/api/warmup` on the Render API, with 3 retries at 40s to ride out cold starts. This is what keeps the Render free dyno from spinning down (15 min idle timeout) — not the app itself.
- **`scripts/setup_cronjob.py`** — a one-time setup script to register the *same* keep-alive ping on cron-job.org instead of/in addition to GitHub Actions (GitHub Actions cron schedules are "best effort" and can silently lag by many minutes, so cron-job.org is the more reliable belt-and-suspenders option). Not part of any automated flow — run manually when the external cron job needs to be (re)created.
- **Neon Postgres free tier** is the actual constraint everything above is designed around: 512MB storage cap and a monthly compute-hour quota. `/api/warmup` deliberately avoids a `SELECT 1` keepalive query so Neon can auto-suspend between pipeline runs — only an *expired cache entry* triggers a real DB read. `task_purge_old_data` (45-day retention) exists because `sentences` text was growing unbounded and hit ~490MB before anyone noticed.
- **Render free tier** is the other constraint: cold start (30-90s) after 15 min idle is what the whole TTL+SWR+fallback cache stack in `api/app.py` is mitigating, and what `keepalive.yml` is preventing from happening at all for real users.

Net effect: three independent schedules (pipeline 12h, GitHub keepalive 14min, optional cron-job.org keepalive) plus an in-process cache all have to be read together to understand why the API behaves as it does at any given moment — if something looks stale or slow, check which of these last fired before assuming an app bug.

Deployment: pipeline runs only via GitHub Actions, never locally/continuously. API deploys to Render (`render.yaml`/`Procfile`), with a Fly.io config (`fly.toml`) and Dockerfile as alternates — all running the same `gunicorn api.app:app` single-worker/multi-thread setup.

### Tests

`conftest.py` provides `test_engine` (in-memory SQLite, session-scoped) and `test_session` (per-test, auto-rollback) fixtures built on `scripts.setup_db.Base` — tests never touch the real database. One test file per layer: `test_collector.py`, `test_classifier.py`, `test_aggregation.py`, `test_ideological.py`.

## Recent work (current branch: `feature/pipeline-png`)

The last stretch of work has been operational hardening of the pipeline/Neon/Render triangle rather than new features, in roughly this order:

1. `0239ba9` — capped classification at the first 100 sentences/article (`_MAX_SENTENCES_CLASSIFY`) and added logging for scraped-vs-fallback rates.
2. `89f29e5` — broke a circular `torch` import by moving `SentenceResult` into `aggregation/bias_score.py`, so the API package can import aggregation code without pulling in the ML stack.
3. `494412e` → `4bfcf46` — added reported-speech detection (`rs_factor`) to BiasScore, then built the 4-layer API cache (pre-warm on startup, in-memory TTL, Stale-While-Revalidate, static JSON fallback) to survive Render cold starts.
4. `36a559e` — made `keepalive.yml` retry 3× with 40s backoff instead of failing on the first cold-start 503.
5. `a9b2cfc`, `e12b244` — fixed/replaced per-outlet scraping (notably R7, whose RSS feed was discontinued — now scraped from its homepage instead).
6. `e759b1a` — added `/api/topics/<slug>` (curated synonym map for topic-filtered story clusters).
7. `f5febf7` — added `POST /api/analyze` + `/api/analyze/<hash>` for on-demand single-URL analysis, dispatched through `analyze_url.yml`.
8. `819850e`/`9c670f7` → `1dfc7ab` → `273b1d1` → `d98f5c4` — the capacity-management arc: collection cadence went 6h → 12h, keepalive interval 5min → 15min (per a stated quota change effective 2026-06-01), cache TTLs raised to 6h then 12h, and `task_purge_old_data` (45-day retention) plus the cumulative-counter fix for `home_summary` were added after the Neon DB silently grew to ~490MB/512MB and started threatening to break the pipeline outright.
9. (uncommitted, working tree) — thesis-side classifier experiment, not a pipeline change: added `--loss {ce,focal}` to `classifier/train.py` and ran a controlled same-seed comparison. Also added `scripts/benchmark_llm_baseline.py`, a one-off zero-shot Gemini benchmark against the same FactNews test split, for a "compared to what?" baseline in the defense — unrelated to the Neon/Render capacity work above, and not wired into any workflow.

**Where things stand:** the free-tier resource ceiling (Neon 512MB / compute-hours, Render spin-down) has been the dominant force shaping the architecture, and the retention+cumulative-counter fix in the HEAD commit is the most load-bearing recent change — any new work that adds DB writes or changes query windows should re-check it doesn't reintroduce unbounded growth or make `home_summary` regress. No open threads or TODOs are recorded in-repo beyond the "trabalhos futuros" list inside `classifier/train.py` (domain-adaptive pretraining, larger backbone, LLM-augmented minority-class data, pseudo-labeling) — none of which are implemented.
