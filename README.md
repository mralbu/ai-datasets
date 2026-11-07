# AI Datasets

[![Live site](https://img.shields.io/badge/live%20site-mralbu.github.io%2Fai--datasets-2f5bd3?style=flat-square)](https://mralbu.github.io/ai-datasets/)
[![Deploy](https://img.shields.io/github/actions/workflow/status/mralbu/ai-datasets/deploy.yml?branch=main&label=deploy&style=flat-square)](https://github.com/mralbu/ai-datasets/actions/workflows/deploy.yml)

A small static site showing what classic AI datasets and benchmarks actually look like:
a short description, the key facts, and real samples for each one.

**→ [mralbu.github.io/ai-datasets](https://mralbu.github.io/ai-datasets/)**

[![The MNIST page: key facts, a digit at its real 28×28 size next to an 8× enlargement, and a grid of samples](docs/screenshot.png)](https://mralbu.github.io/ai-datasets/d/mnist)

Not a catalog. Hugging Face and Papers with Code already do that. The goal is a few dozen
historically important datasets, each shown well enough that you come away knowing the data.

## How it works

```
datasets/<id>.yaml      one file per dataset: facts, description, HF source, display kind
scripts/sample.py       (uv + polars + altair) fetches a seeded random sample of rows from the
                        Hugging Face dataset viewer API → src/generated/<id>.json + public/samples/<id>/
src/                    Astro site: catalog, timeline, one page per dataset
```

Samples are fetched at build time and are not committed (see `.gitignore`).

## Develop

```sh
npm install
npm run fetch              # uv run scripts/sample.py  (or: uv run scripts/sample.py mnist iris)
                           # a failed dataset only warns; add --strict to make it an error
BASE_PATH=/ npm run dev    # http://localhost:4321
```

## Add a dataset

1. Copy one of `datasets/*.yaml` and fill it in. `display.kind` is one of:
   - `image`: `image` field, optional `label`, `scale` (enlargement factor for the grid)
   - `text`: `text` field, optional `label`
   - `qa`: `question` and `answer` fields
   - `table`: optional `chart: {x, y, color}`; set `source.fetch_all: true` so the chart sees every row
2. `uv run scripts/sample.py <id>` and check the page.

The schema lives in `src/content.config.ts`; the build fails on a malformed entry.

## Deploy

`.github/workflows/deploy.yml` fetches samples, builds, and publishes to GitHub Pages on every push
to `main` (and weekly). In the repo settings, set **Pages → Source** to **GitHub Actions**.
