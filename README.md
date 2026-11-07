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
scripts/sample.py       (uv + polars + altair) samples rows from the Hugging Face dataset viewer
                        into a snapshot, then copies it to src/generated/ and public/samples/
src/                    Astro site: catalog, timeline, one page per dataset
samples branch          the snapshot: sampled rows, media and a manifest (generated, data only)
```

### Where samples come from

Samples are fetched once and kept on the **`samples` branch**, an orphan branch that holds only
data. Each dataset has a fingerprint (its `source` block and chart config). On every deploy the
workflow checks out `samples` into `.samples/`, re-fetches only datasets that are new or whose
fingerprint changed, prunes removed ones, commits the snapshot back to `samples`, and builds.
A normal push touches Hugging Face not at all.

- `--strict` in CI: if a fetch fails, the build fails and Pages keeps serving the last good site.
  The previous sample of that dataset is kept in the snapshot.
- `manifest.json` on the branch records, per dataset, the Hugging Face revision and fetch time;
  each page shows it under the samples.
- To re-fetch on purpose: run the deploy workflow manually with `refetch` set to dataset ids or
  `all`, or bump `SAMPLER_VERSION` in `scripts/sample.py`.
- Taking a sample down: delete or edit it on the `samples` branch (squash or recreate the branch
  if it must leave history); nothing on `main` needs rewriting.
- A monthly workflow (`check-sources.yml`) makes one tiny request per source and fails if a
  dataset was renamed, gated or deleted upstream.

## Develop

```sh
npm install
git fetch origin samples && git worktree add .samples samples   # reuse the published snapshot
npm run fetch              # uv run scripts/sample.py: fetches only what is missing or changed
BASE_PATH=/ npm run dev    # http://localhost:4321
```

Without the worktree, `.samples/` is just a local cache and the first `npm run fetch` fills it
(about 3 minutes). Other options: `uv run scripts/sample.py <id>` (only that dataset),
`--force` (re-fetch even if unchanged), `--check` (health check only).

## Add a dataset

1. Copy a similar entry in `datasets/*.yaml` and fill it in. `display.kind` picks the renderer
   (field names are columns of the Hugging Face dataset):

   | kind | fields | used by |
   |---|---|---|
   | `image` | `image`, `label`, `scale` (enlarge tiny images), `thumb` (px height for photos), `class_grid`, `short_labels` | MNIST, Fashion-MNIST, CIFAR-10, ImageNet |
   | `boxes` | `image`, `objects` (COCO-style `bbox` `[x, y, w, h]` + a label list), `label` (its name, default `label`) | COCO, Pascal VOC |
   | `text` | `text`, `label`, `summary` (shown on demand), `caption` (small line, e.g. a URL) | IMDB, 20 Newsgroups, CNN/DailyMail, C4 |
   | `extractive` | `context`, `question`, `answers`, `title` | SQuAD |
   | `record` | `label`; shows every text field, per config | GLUE, SuperGLUE, SNLI |
   | `qa` | `question`, `answer`, `tags` (fields shown as tags), `math` (render LaTeX with KaTeX), `input`, `wrong` (list of false answers), `answer_label` | GSM8K, MATH, TruthfulQA, Alpaca |
   | `mcq` | `question`, `choices` (a list, or `{text, label}`), `answer` (index or key), `subject` | MMLU, HellaSwag, AI2 ARC |
   | `code` | `prompt`, `solution`, `tests`, `title`; `*_lang` (Shiki language, `text` for prose) and `*_label` | HumanEval, SWE-bench |
   | `audio` | `audio`, `text` and/or `label` (class or list of classes), `speaker` | LibriSpeech, AudioSet, Speech Commands, LJSpeech |
   | `grid` | `train`, `test` (ARC-style pairs of integer grids) | ARC-AGI |
   | `table` | optional `chart: {type: scatter \| rate, x, y, color}`, `columns` | Iris, Titanic, Palmer Penguins, MovieLens |
| `annotated` | `image`, `captions` (list) or `attributes: true` (yes/no columns), `thumb` | Flickr30k, CelebA |
| `image_mcq` | `images` (list of image columns), `question`, `options`, `answer` (letter), `subject` | MMMU |
| `preference` | `chosen`, `rejected` (Human:/Assistant: transcripts) | Anthropic HH-RLHF |
| `translation` | `translation` (a Translation feature), `source`, `target` (language codes) | WMT14 |

   Useful `source` options: `n` (rows to sample), `seed`, `fetch_all: true` (small tables, needed for
   charts), `configs: [...]` (sample several configs, like GLUE's tasks), and `rows_per_page`
   (default 8; lower it when the split is sorted by a label with many values, as ImageNet's is),
   and `min_length: {field: n}` (skip rows whose field is shorter, like WikiText's blank lines).

   Each page has a "Get the data" box with copy-paste snippets for the `hf` CLI, 🤗 `datasets` and
   Polars, built from `source`. Override it with an optional `download` block: `hf` (the canonical
   repo when the sample comes from a mirror), `config`, `split`, `include` (glob for
   `hf download --include`, for one config of a big repo), `revision`, `gated: true` and `note`.
2. `uv run scripts/sample.py <id>` and check the page with `BASE_PATH=/ npm run dev`.
3. Related datasets (`lineage`) take `hf: <owner/name>` to link to Hugging Face, or `url:` when
   there is no HF copy, and `id:` when the dataset also has a page on this site.

The schema lives in `src/content.config.ts`; the build fails on a malformed entry.

## Deploy

`.github/workflows/deploy.yml` updates the snapshot, builds, and publishes to GitHub Pages on every
push to `main`. In the repo settings, set **Pages → Source** to **GitHub Actions**.
