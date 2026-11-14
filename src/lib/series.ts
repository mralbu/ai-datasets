// Turns sampled rows into time series to draw (display kind "series"), and draws them as SVG paths.
import type { Samples, SampleRow, Display } from './data';

type SeriesDisplay = Extract<Display, { kind: 'series' }>;

export interface Series {
  title?: string;
  tags: string[];
  channels: (number | null)[][]; // one or more lines on the same axes
  times?: string[]; // one per point, when the data has them
  start?: string; // otherwise the first timestamp, if known
  future?: (number | null)[]; // values to forecast, drawn after the history
  futureStart?: string;
  points?: { total: number; from: number }; // the series was cut to this stretch by the sampler
  unit?: string;
}

const num = (v: unknown): number | null => {
  const x = typeof v === 'string' ? Number(v) : v;
  return typeof x === 'number' && Number.isFinite(x) ? x : null;
};

/** A list cell as channels: [1, 2, 3] is one line; [[1], [2]] (UCR's layout) or [[a, b], …] is one per column. */
function channelsOf(cell: unknown): (number | null)[][] {
  if (!Array.isArray(cell) || !cell.length) return [];
  if (!Array.isArray(cell[0])) return [cell.map(num)];
  const k = (cell[0] as unknown[]).length;
  return Array.from({ length: k }, (_, j) => cell.map((p) => num((p as unknown[])[j])));
}

const valueCols = (d: SeriesDisplay) => (Array.isArray(d.value) ? d.value : [d.value]);

function tagsOf(d: SeriesDisplay, r: SampleRow, multiConfig: boolean): string[] {
  const t: string[] = [];
  if (multiConfig && r.config) t.push(r.config);
  if (d.label && r.row[d.label] != null) {
    const v = r.row[d.label];
    t.push(typeof v === 'number' ? `class ${v}` : String(v));
  }
  for (const f of d.tags) if (r.row[f] != null && r.row[f] !== '') t.push(String(r.row[f]));
  return t;
}

/** One series per sampled row (list columns), or one per value column of a window of rows. */
export function seriesOf(s: Samples, d: SeriesDisplay): Series[] {
  if (s.window) {
    const cols = s.window.columns;
    const times = d.time ? (cols[d.time] ?? []).map(String) : undefined;
    return valueCols(d).map((c) => ({
      title: d.names[c] ? `${c} · ${d.names[c]}` : c,
      tags: [],
      channels: [(cols[c] ?? []).map(num)],
      times,
      unit: d.unit,
    }));
  }
  const multi = (s.parts?.length ?? 0) > 1;
  return s.rows.map((r) => {
    const col = valueCols(d)[0];
    const pts = r.points?.[col];
    return {
      title: d.title ? String(r.row[d.title] ?? '') : undefined,
      tags: tagsOf(d, r, multi),
      channels: channelsOf(r.row[col]),
      times: d.time && Array.isArray(r.row[d.time]) ? r.row[d.time].map(String) : undefined,
      start: d.start && !pts ? String(r.row[d.start] ?? '') || undefined : undefined,
      future: d.future ? channelsOf(r.row[d.future])[0] : undefined,
      futureStart: d.future_start ? String(r.row[d.future_start] ?? '') || undefined : undefined,
      points: pts,
      unit: d.unit,
    };
  });
}

/** "2016-07-01T00:00:00" -> "2016-07-01"; keeps the time of day only for sub-daily data. */
export function timeFormatter(times: string[] | undefined): (t: string) => string {
  const clean = (t: string) => t.replace('T', ' ').replace(/(\d\d:\d\d):00(\.0+)?$/, '$1');
  if (!times || times.length < 2) return (t) => clean(t).replace(/ 00:00$/, '');
  const a = Date.parse(times[0]), b = Date.parse(times[1]);
  const subDaily = Number.isFinite(a) && Number.isFinite(b) && Math.abs(b - a) < 86_400_000;
  return subDaily ? clean : (t) => clean(t).replace(/ \d\d:\d\d$/, '');
}

export function fmtValue(v: number | null): string {
  if (v === null) return '∅';
  const a = Math.abs(v);
  return new Intl.NumberFormat('en-US', {
    maximumSignificantDigits: 4,
    notation: a >= 1e6 ? 'compact' : 'standard',
  }).format(v);
}

export interface Plot {
  lines: string[]; // one path per channel
  band?: string; // min–max envelope, when points were bucketed
  future?: string;
  split?: number; // x where the forecast starts
  zero?: number; // y of the zero line, when the range crosses it
  min: number;
  max: number;
  lo: number; // the y range as drawn (min and max, padded)
  hi: number;
  n: number; // points of history
  hover: { v: (number | null)[]; t?: string[]; i: number[] }; // a light copy for the readout
}

/**
 * Paths in a W×H box (stretched by the SVG, strokes kept thin with non-scaling-stroke).
 * Long series are bucketed: a mean line plus a faint min–max band, so peaks are not lost.
 */
export function plot(sr: Series, W = 1000, H = 100, maxBuckets = 500): Plot {
  const n = Math.max(0, ...sr.channels.map((c) => c.length));
  const fut = sr.future ?? [];
  const total = n + fut.length;
  const all = [...sr.channels.flat(), ...fut].filter((v): v is number => v !== null);
  let min = all.length ? Math.min(...all) : 0, max = all.length ? Math.max(...all) : 1;
  if (min === max) { min -= 1; max += 1; }
  const pad = (max - min) * 0.06;
  const lo = min - pad, hi = max + pad;
  const X = (i: number) => (total <= 1 ? W / 2 : (i / (total - 1)) * W);
  const Y = (v: number) => H - ((v - lo) / (hi - lo)) * H;
  const f = (x: number) => Math.round(x * 10) / 10;

  const linePath = (vals: (number | null)[], offset = 0) => {
    let d = '', pen = false;
    vals.forEach((v, i) => {
      if (v === null) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${f(X(i + offset))} ${f(Y(v))}`;
      pen = true;
    });
    return d;
  };

  const out: Plot = { lines: [], min, max, lo, hi, n, hover: { v: [], i: [] } };
  const buckets = Math.min(maxBuckets, n);
  if (n > maxBuckets * 1.5) {
    const size = n / buckets;
    const top: string[] = [], bot: string[] = [];
    out.lines = sr.channels.map((ch) => {
      let d = '', pen = false;
      for (let b = 0; b < buckets; b++) {
        const seg = ch.slice(Math.floor(b * size), Math.floor((b + 1) * size)).filter((v): v is number => v !== null);
        if (!seg.length) { pen = false; continue; }
        const mean = seg.reduce((a, c) => a + c, 0) / seg.length;
        const x = f(X((b + 0.5) * size));
        d += `${pen ? 'L' : 'M'}${x} ${f(Y(mean))}`;
        pen = true;
        if (ch === sr.channels[0]) {
          top.push(`${x} ${f(Y(Math.max(...seg)))}`);
          bot.unshift(`${x} ${f(Y(Math.min(...seg)))}`);
        }
      }
      return d;
    });
    if (top.length > 1) out.band = `M${top.join('L')}L${bot.join('L')}Z`;
  } else {
    out.lines = sr.channels.map((ch) => linePath(ch));
  }
  if (fut.length) {
    // Join the forecast to the last point of history so the dashes continue the line.
    const last = sr.channels[0]?.[n - 1] ?? null;
    out.future = linePath([last, ...fut], n - 1);
    out.split = f(X(n - 1));
  }
  if (lo < 0 && hi > 0) out.zero = f(Y(0));

  // Hover readout: at most ~400 evenly spaced points of history + forecast.
  const step = Math.max(1, Math.ceil(total / 400));
  for (let i = 0; i < total; i += step) {
    out.hover.i.push(i);
    out.hover.v.push(i < n ? sr.channels[0]?.[i] ?? null : fut[i - n] ?? null);
    if (sr.times && i < sr.times.length) (out.hover.t ??= []).push(sr.times[i]);
  }
  return out;
}
