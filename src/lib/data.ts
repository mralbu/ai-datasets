import { getCollection, type CollectionEntry } from 'astro:content';

export type Dataset = CollectionEntry<'datasets'>;
export type Display = Dataset['data']['display'];

export interface FeatureType { _type?: string; names?: string[]; dtype?: string; [k: string]: unknown }
export interface Feature { name: string; type: FeatureType }
export interface SampleRow { row_idx: number; config?: string; row: Record<string, any>; truncated: string[] }
export interface Part { config: string; split: string; num_rows_total: number; features: Feature[] }
export interface Samples {
  id: string;
  source: { hf: string; config?: string; split?: string };
  num_rows_total: number;
  fetched_at: string;
  revision?: string | null; // Hub commit the sample was taken from
  features: Feature[];
  parts?: Part[];
  rows: SampleRow[];
  chart?: Record<string, unknown>;
}

// Written by scripts/sample.py. Missing files just mean "not fetched yet".
const files = import.meta.glob<Samples>('../generated/*.json', { eager: true, import: 'default' });

export function getSamples(id: string): Samples | undefined {
  return files[`../generated/${id}.json`];
}

export async function getDatasets(): Promise<Dataset[]> {
  return (await getCollection('datasets')).sort((a, b) => a.data.year - b.data.year || a.data.name.localeCompare(b.data.name));
}

/** Features for a row: per-config for suites like GLUE, else the dataset's. */
export function featuresFor(s: Samples, row?: SampleRow): Feature[] {
  return s.parts?.find((p) => p.config === row?.config)?.features ?? s.features;
}

/** Map a ClassLabel integer to its name, otherwise return the value unchanged. */
export function labelOf(s: Samples, field: string, value: unknown, row?: SampleRow): string {
  const f = featuresFor(s, row).find((f) => f.name === field);
  if (f?.type._type === 'ClassLabel' && typeof value === 'number') return f.type.names?.[value] ?? String(value);
  return String(value);
}

/** Class names of a ClassLabel feature (possibly nested, e.g. objects.label in COCO). */
export function classNames(s: Samples, field: string, sub = 'label'): string[] {
  const f = s.features.find((f) => f.name === field);
  const t: any = f?.type;
  return t?.names ?? t?.[sub]?.feature?.names ?? [];
}

/** Prefix a site-relative path with the configured base. */
export function url(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}/${path.replace(/^\//, '')}`;
}

// Display order of categories everywhere (catalog sections, filter chips).
export const MODALITIES: Record<string, string> = {
  vision: 'Vision',
  text: 'Text',
  benchmark: 'LLMs', // benchmarks and training data (instructions, preferences) for language models
  tabular: 'Tabular',
  audio: 'Audio',
  other: 'Other',
};

/** The standard ARC display palette for colors 0–9. */
export const ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'];
