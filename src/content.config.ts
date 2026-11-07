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
  z.object({
    kind: z.literal('boxes'),
    image: z.string(),
    objects: z.string(), // COCO-style struct of lists: bbox [x, y, w, h] in pixels, plus a label list
    label: z.string().default('label'), // name of the label list inside objects (Pascal VOC copies call it category)
  }),
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
    columns: z.array(z.string()).optional(), // columns to show (default: all)
  }),
  z.object({
    kind: z.literal('qa'),
    question: z.string(),
    answer: z.string(),
    tags: z.array(z.string()).default([]), // fields shown as small tags (e.g. MATH's subject and level)
    math: z.boolean().default(false), // render $…$ LaTeX with KaTeX
  }),
  z.object({ kind: z.literal('extractive'), context: z.string(), question: z.string(), answers: z.string(), title: z.string().optional() }),
  z.object({ kind: z.literal('mcq'), question: z.string(), choices: z.string(), answer: z.string(), subject: z.string().optional() }),
  z.object({
    kind: z.literal('code'),
    prompt: z.string(),
    solution: z.string(),
    tests: z.string().optional(),
    title: z.string().optional(),
    // Shiki language per field; 'text' renders the prompt as prose (a GitHub issue, say).
    prompt_lang: z.string().default('python'),
    solution_lang: z.string().default('python'),
    tests_lang: z.string().default('python'),
    prompt_label: z.string().default('The prompt the model sees:'),
    solution_label: z.string().default('Reference solution'),
    tests_label: z.string().default('Hidden unit tests'),
  }),
  z.object({ kind: z.literal('audio'), audio: z.string(), text: z.string(), speaker: z.string().optional() }),
  z.object({ kind: z.literal('grid'), train: z.string(), test: z.string() }),
  z.object({ kind: z.literal('record'), label: z.string().optional() }),
  // A Translation feature, e.g. {de: …, en: …}; source/target pick the order.
  z.object({ kind: z.literal('translation'), translation: z.string(), source: z.string(), target: z.string() }),
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
    // Related datasets: link to Hugging Face (hf), or a homepage (url) when there is no HF copy,
    // plus id when the dataset also has a page on this site.
    lineage: z
      .array(z.object({ name: z.string(), note: z.string().optional(), hf: z.string().optional(), url: z.string().url().optional(), id: z.string().optional() }))
      .default([]),
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
      min_length: z.record(z.string(), z.number().int()).optional(), // skip rows whose field is shorter (WikiText's blank lines)
    }),
    // How to get the full dataset (the download box). Defaults to the sampled source repo.
    download: z
      .object({
        hf: z.string().optional(), // when the sample comes from a mirror, the canonical repo
        config: z.string().optional(),
        split: z.string().optional(),
        include: z.string().optional(), // glob for `hf download --include`, for one config of a big repo
        revision: z.string().optional(), // e.g. refs/convert/parquet for loading-script repos
        gated: z.boolean().default(false), // needs `hf auth login` and accepting terms on the Hub
        note: z.string().optional(),
      })
      .default({}),
    display,
  }),
});

export const collections = { datasets };
