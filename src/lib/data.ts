import { getCollection, type CollectionEntry } from 'astro:content';

export type Dataset = CollectionEntry<'datasets'>;

export interface Feature { name: string; type: { _type?: string; names?: string[]; dtype?: string } }
export interface SampleRow { row_idx: number; row: Record<string, any>; truncated: string[] }
export interface Samples {
  id: string;
  source: { hf: string; config?: string; split?: string };
  num_rows_total: number;
  fetched_at: string;
  features: Feature[];
  rows: SampleRow[];
  chart?: Record<string, unknown>;
}

// Written by scripts/sample.py. Missing files just mean "not fetched yet".
const files = import.meta.glob<Samples>('../generated/*.json', { eager: true, import: 'default' });

export function getSamples(id: string): Samples | undefined {
  return files[`../generated/${id}.json`];
}

export async function getDatasets(): Promise<Dataset[]> {
  return (await getCollection('datasets')).sort((a, b) => a.data.year - b.data.year);
}

/** Map a ClassLabel integer to its name, otherwise return the value unchanged. */
export function labelOf(samples: Samples, field: string, value: unknown): string {
  const f = samples.features.find((f) => f.name === field);
  if (f?.type._type === 'ClassLabel' && typeof value === 'number') return f.type.names?.[value] ?? String(value);
  return String(value);
}

/** Prefix a site-relative path with the configured base. */
export function url(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}/${path.replace(/^\//, '')}`;
}

export const MODALITIES: Record<string, string> = {
  vision: 'Vision',
  text: 'Text',
  tabular: 'Tabular',
  audio: 'Audio',
  benchmark: 'LLM benchmarks',
  other: 'Other',
};
