import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const link = z.object({ title: z.string(), url: z.string().url() });

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
    paper: link,
    homepage: z.string().url().optional(),
    why: z.string(),
    trivia: z.array(z.string()).default([]),
    lineage: z.array(z.object({ name: z.string(), note: z.string().optional() })).default([]),
    landmark_papers: z
      .array(z.object({ title: z.string(), year: z.number().int(), url: z.string().url() }))
      .default([]),
    source: z.object({
      hf: z.string(),
      config: z.string().optional(),
      split: z.string().optional(),
      n: z.number().int().optional(),
      seed: z.number().int().optional(),
      fetch_all: z.boolean().optional(),
    }),
    display: z.discriminatedUnion('kind', [
      z.object({ kind: z.literal('image'), image: z.string(), label: z.string().optional(), scale: z.number().default(4) }),
      z.object({ kind: z.literal('text'), text: z.string(), label: z.string().optional() }),
      z.object({
        kind: z.literal('table'),
        chart: z.object({ x: z.string(), y: z.string(), color: z.string().optional(), title: z.string().optional() }).optional(),
      }),
      z.object({ kind: z.literal('qa'), question: z.string(), answer: z.string() }),
    ]),
  }),
});

export const collections = { datasets };
