import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const link = z.object({ title: z.string(), url: z.string().url() });

// How samples are rendered. Field names refer to columns in the Hugging Face dataset.
const display = z.discriminatedUnion('kind', [
  z.object({
    kind: z.literal('image'),
    image: z.string(),
    label: z.string().optional(),
    scale: z.number().default(4), // enlargement for small images
    thumb: z.number().optional(), // fixed thumbnail height (px) for large images
    class_grid: z.boolean().default(false), // also show one example per class
    short_labels: z.boolean().default(false), // "tench, Tinca tinca" -> "tench"
  }),
  z.object({ kind: z.literal('boxes'), image: z.string(), objects: z.string() }),
  z.object({ kind: z.literal('text'), text: z.string(), label: z.string().optional() }),
  z.object({
    kind: z.literal('table'),
    chart: z
      .object({
        type: z.enum(['scatter', 'rate']).default('scatter'),
        x: z.string(),
        y: z.string(),
        color: z.string().optional(),
        title: z.string().optional(),
        y_title: z.string().optional(),
      })
      .optional(),
  }),
  z.object({ kind: z.literal('qa'), question: z.string(), answer: z.string() }),
  z.object({ kind: z.literal('extractive'), context: z.string(), question: z.string(), answers: z.string(), title: z.string().optional() }),
  z.object({ kind: z.literal('mcq'), question: z.string(), choices: z.string(), answer: z.string(), subject: z.string().optional() }),
  z.object({ kind: z.literal('code'), prompt: z.string(), solution: z.string(), tests: z.string().optional(), title: z.string().optional() }),
  z.object({ kind: z.literal('audio'), audio: z.string(), text: z.string(), speaker: z.string().optional() }),
  z.object({ kind: z.literal('grid'), train: z.string(), test: z.string() }),
  z.object({ kind: z.literal('record'), label: z.string().optional() }),
]);

// One YAML file per dataset in /datasets. scripts/sample.py reads the same files.
const datasets = defineCollection({
  loader: glob({ pattern: '*.yaml', base: './datasets' }),
  schema: z.object({
    name: z.string(),
    year: z.number().int(),
    modality: z.enum(['vision', 'text', 'tabular', 'audio', 'benchmark', 'other']),
    task: z.string(),
    one_liner: z.string(),
    creators: z.string(),
    size: z.string(),
    splits: z.record(z.string(), z.number()),
    input: z.string(),
    labels: z.string(),
    license: z.string(),
    paper: link.optional(),
    homepage: z.string().url().optional(),
    why: z.string(),
    trivia: z.array(z.string()).default([]),
    config_notes: z.record(z.string(), z.string()).default({}),
    lineage: z.array(z.object({ name: z.string(), note: z.string().optional() })).default([]),
    landmark_papers: z
      .array(z.object({ title: z.string(), year: z.number().int(), url: z.string().url() }))
      .default([]),
    source: z.object({
      hf: z.string(),
      config: z.string().optional(),
      split: z.string().optional(),
      configs: z.array(z.union([z.string(), z.object({ config: z.string(), split: z.string().optional() })])).optional(),
      n: z.number().int().optional(),
      seed: z.number().int().optional(),
      fetch_all: z.boolean().optional(),
      rows_per_page: z.number().int().optional(),
    }),
    display,
  }),
});

export const collections = { datasets };
