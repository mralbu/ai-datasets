// Copy-paste snippets for the full dataset (Hub CLI, 🤗 datasets, Polars), built from `source`
// and the optional `download` block. Used by the "Get the data" box and by llms.txt.
import type { Dataset } from './data';

export interface Snippet { id: 'cli' | 'datasets' | 'polars'; label: string; lang: 'bash' | 'python'; code: string }

export function snippets(ds: Dataset): Snippet[] {
  const { source, download: dl } = ds.data;

  const repo = dl.hf ?? source.hf;
  const configs = (source.configs ?? []).map((c) => (typeof c === 'string' ? c : c.config));
  const config = dl.config ?? (dl.hf ? undefined : configs[0] ?? source.config);
  const first = source.configs?.[0];
  const split = dl.split ?? (dl.hf ? undefined : (typeof first === 'object' ? first.split : undefined) ?? source.split) ?? 'train';
  const named = config && config !== 'default';
  const q = (v: string) => JSON.stringify(v);

  const others = configs.length > 1 ? `  # or any of: ${configs.slice(1, 6).join(', ')}${configs.length > 6 ? ', …' : ''}` : '';
  const cli = [
    '# pip install -U huggingface_hub',
    ...(dl.gated ? ['hf auth login   # gated: accept the terms on the dataset page first'] : []),
    [
      `hf download ${repo} --repo-type dataset`,
      dl.include && `--include ${q(dl.include)}`,
      dl.revision && `--revision ${dl.revision}`,
      `--local-dir ${ds.id}`,
    ].filter(Boolean).join(' \\\n  '),
  ].join('\n');

  const py = [
    'from datasets import load_dataset   # pip install datasets',
    '',
    `ds = load_dataset(${[q(repo), named && q(config!), dl.revision && `revision=${q(dl.revision)}`].filter(Boolean).join(', ')})${others}`,
    'print(ds)',
  ].join('\n');

  // The Hub's Parquet conversion (refs/convert/parquet, "@~parquet") exists for every dataset the viewer can show.
  const polars = [
    'import polars as pl',
    '',
    `df = pl.read_parquet(${q(`hf://datasets/${repo}@~parquet/${config ?? 'default'}/${split}/*.parquet`)})`,
    ...(dl.gated ? ['# gated: set HF_TOKEN, or run `hf auth login` first'] : []),
    'print(df.shape)',
  ].join('\n');

  return [
    { id: 'cli', label: 'CLI', lang: 'bash', code: cli },
    { id: 'datasets', label: 'datasets', lang: 'python', code: py },
    { id: 'polars', label: 'Polars', lang: 'python', code: polars },
  ];
}

/** The repository the full dataset is downloaded from (a mirror's canonical repo when set). */
export const downloadRepo = (ds: Dataset) => ds.data.download.hf ?? ds.data.source.hf;
