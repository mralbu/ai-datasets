import { getCollection, type CollectionEntry } from 'astro:content';

export type Dataset = CollectionEntry<'datasets'>;
export type Display = Dataset['data']['display'];

export interface FeatureType { _type?: string; names?: string[]; dtype?: string; [k: string]: unknown }
export interface Feature { name: string; type: FeatureType }
export interface SampleRow {
  row_idx: number; config?: string; row: Record<string, any>; truncated: string[];
  points?: Record<string, { total: number; from: number }>; // list columns cut to source.max_points
}
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
  // A window of consecutive rows of a long-format time series (source.window), as columns.
  window?: { start: number; n: number; columns: Record<string, unknown[]> };
}

// Written by scripts/sample.py. Missing files just mean "not fetched yet".
const files = import.meta.glob<Samples>('../generated/*.json', { eager: true, import: 'default' });

export function getSamples(id: string): Samples | undefined {
  return files[`../generated/${id}.json`];
}

export async function getDatasets(): Promise<Dataset[]> {
  return (await getCollection('datasets')).sort((a, b) => a.data.year - b.data.year || a.data.name.localeCompare(b.data.name));
}

export type Paper = CollectionEntry<'papers'>;

export async function getPapers(): Promise<Paper[]> {
  return (await getCollection('papers')).sort((a, b) => a.data.year - b.data.year || a.data.short.localeCompare(b.data.short));
}

/** Where to read a paper: alphaXiv and arXiv when it has an arXiv id, else its own page. */
export function paperLinks(p: Paper['data']) {
  return {
    alphaxiv: p.arxiv ? `https://www.alphaxiv.org/abs/${p.arxiv}` : undefined,
    arxiv: p.arxiv ? `https://arxiv.org/abs/${p.arxiv}` : undefined,
    main: p.arxiv ? `https://arxiv.org/abs/${p.arxiv}` : p.url!,
  };
}

/** A comparable key for a paper link: the arXiv id when there is one, else the URL. */
export function paperKey(href: string): string {
  return href.match(/arxiv\.org\/(?:abs|pdf)\/(\d{4}\.\d{4,5})/)?.[1] ?? href.replace(/\/$/, '');
}

/** Papers on the site that link to a dataset. */
export async function papersFor(datasetId: string): Promise<Paper[]> {
  return (await getPapers()).filter((p) => p.data.datasets.some((d) => d.id === datasetId));
}

/** Papers on the site that build on a paper. */
export async function ledTo(paperId: string): Promise<Paper[]> {
  return (await getPapers()).filter((p) => p.data.builds_on.some((b) => b.id === paperId));
}

export type Track = CollectionEntry<'tracks'>;

export async function getTracks(): Promise<Track[]> {
  return (await getCollection('tracks')).sort((a, b) => a.data.title.localeCompare(b.data.title));
}

export interface PaperTrackContext {
  track: Track;
  index: number;
  total: number;
  prev?: Paper;
  next?: Paper;
}

/** Tracks that include a given paper, with its position and immediate neighbors in each track. */
export async function tracksFor(paperId: string): Promise<PaperTrackContext[]> {
  const [tracks, papers] = await Promise.all([getTracks(), getPapers()]);
  const pById = new Map(papers.map((p) => [p.id, p]));
  return tracks.flatMap((track) => {
    const steps = track.data.steps;
    const index = steps.findIndex((s) => s.paper.id === paperId);
    if (index < 0) return [];
    return [{
      track,
      index,
      total: steps.length,
      prev: index > 0 ? pById.get(steps[index - 1].paper.id) : undefined,
      next: index + 1 < steps.length ? pById.get(steps[index + 1].paper.id) : undefined,
    }];
  });
}

export type TrackItem =
  | { kind: 'dataset'; ds: Dataset; year: number }
  | { kind: 'paper'; p: Paper; year: number; step: number; note: string };

/**
 * Weave a track's datasets into its authored paper order without re-sorting papers by year.
 * Datasets are sorted by year and placed before the first step with paper.year >= dataset.year
 * (matching timeline.astro, where datasets precede same-year papers).
 */
export function trackTimeline(track: Track, dsById: Map<string, Dataset>, pById: Map<string, Paper>): TrackItem[] {
  const datasets = track.data.datasets
    .map((r) => dsById.get(r.id)!)
    .sort((a, b) => a.data.year - b.data.year || a.data.name.localeCompare(b.data.name));
  const steps = track.data.steps.map((s, i) => ({ p: pById.get(s.paper.id)!, step: i + 1, note: s.note }));
  const out: TrackItem[] = [];
  let di = 0;
  for (const s of steps) {
    while (di < datasets.length && datasets[di].data.year <= s.p.data.year) {
      out.push({ kind: 'dataset', ds: datasets[di], year: datasets[di].data.year });
      di++;
    }
    out.push({ kind: 'paper', p: s.p, year: s.p.data.year, step: s.step, note: s.note });
  }
  while (di < datasets.length) {
    out.push({ kind: 'dataset', ds: datasets[di], year: datasets[di].data.year });
    di++;
  }
  return out;
}

const warnedTracks = new Set<string>();

/** Soft build-time check: warn when a track step has no builds_on link to any earlier step. */
export function checkTrackBuildsOn(track: Track, pById: Map<string, Paper>): void {
  if (warnedTracks.has(track.id)) return;
  warnedTracks.add(track.id);
  const seen = new Set<string>();
  for (const [i, step] of track.data.steps.entries()) {
    const pid = step.paper.id;
    const p = pById.get(pid);
    if (i > 0 && p && !p.data.builds_on.some((b) => seen.has(b.id))) {
      console.warn(`[tracks/${track.id}] step ${i + 1} (${pid}) has no builds_on link to any earlier step in this track`);
    }
    seen.add(pid);
  }
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
  foundations: 'Foundations', // papers only: training techniques for any data and any model
  vision: 'Vision',
  text: 'Text',
  benchmark: 'LLMs', // benchmarks and training data (instructions, preferences) for language models
  tabular: 'Tabular',
  timeseries: 'Time series',
  audio: 'Audio',
  other: 'Other',
};

export { TAGS, type Tag } from './tags';

/** The standard ARC display palette for colors 0–9. */
export const ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'];
