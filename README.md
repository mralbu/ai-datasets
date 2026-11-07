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

1. Copy a similar entry in `datasets/*.yaml` and fill it in. `display.kind` picks the renderer
   (field names are columns of the Hugging Face dataset):

   | kind | fields | used by |
   |---|---|---|
   | `image` | `image`, `label`, `scale` (enlarge tiny images), `thumb` (px height for photos), `class_grid`, `short_labels` | MNIST, CIFAR-10, ImageNet |
   | `boxes` | `image`, `objects` (COCO-style `bbox` `[x, y, w, h]` + `label`) | COCO |
   | `text` | `text`, `label` | IMDB |
   | `extractive` | `context`, `question`, `answers`, `title` | SQuAD |
   | `record` | `label`; shows every text field, per config | GLUE |
   | `qa` | `question`, `answer` | GSM8K |
   | `mcq` | `question`, `choices`, `answer`, `subject` | MMLU, HellaSwag |
   | `code` | `prompt`, `solution`, `tests`, `title` | HumanEval |
   | `audio` | `audio`, `text`, `speaker` | LibriSpeech |
   | `grid` | `train`, `test` (ARC-style pairs of integer grids) | ARC-AGI |
   | `table` | optional `chart: {type: scatter \| rate, x, y, color}` | Iris, Titanic |

   Useful `source` options: `n` (rows to sample), `seed`, `fetch_all: true` (small tables, needed for
   charts), `configs: [...]` (sample several configs, like GLUE's tasks), and `rows_per_page`
   (default 8; lower it when the split is sorted by a label with many values, as ImageNet's is).
2. `uv run scripts/sample.py <id>` and check the page with `BASE_PATH=/ npm run dev`.

The schema lives in `src/content.config.ts`; the build fails on a malformed entry.

## Deploy

`.github/workflows/deploy.yml` fetches samples, builds, and publishes to GitHub Pages on every push
to `main` (and weekly). In the repo settings, set **Pages → Source** to **GitHub Actions**.
