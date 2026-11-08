import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { TAGS, type Tag } from './lib/tags';

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
    names: z.array(z.string()).optional(), // class names by id, when the labels are plain integers (KITTI's copy)
    ignore: z.array(z.string()).default([]), // classes drawn as dashed "don't care" regions, without a label
  }),
  z.object({
    kind: z.literal('text'),
    text: z.string(),
    label: z.string().optional(),
    summary: z.string().optional(), // reference summary, shown on demand (CNN/DailyMail)
    caption: z.string().optional(), // small muted line under the text, e.g. the source URL (C4)
  }),
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
    input: z.string().optional(), // extra input under the question (Alpaca), or {contexts, labels} sections (PubMedQA)
    verdict: z.string().optional(), // a short final answer shown as a badge above the answer (PubMedQA's yes/no/maybe)
    wrong: z.string().optional(), // list of incorrect answers, shown with the solution (TruthfulQA)
    answer_label: z.string().default('Show solution'),
    math: z.boolean().default(false), // render $…$ LaTeX with KaTeX
  }),
  // answers: SQuAD's {text, answer_start}, or a list of {span_text, span_start} (Natural Questions);
  // question: a string, or a list of {input_text}.
  z.object({ kind: z.literal('extractive'), context: z.string(), question: z.string(), answers: z.string(), title: z.string().optional() }),
  // Multi-hop QA over several paragraphs (HotpotQA): context {title, sentences}, supporting_facts {title, sent_id}.
  z.object({
    kind: z.literal('multihop'), question: z.string(), answer: z.string(), context: z.string(), supporting: z.string(),
    tags: z.array(z.string()).default([]),
  }),
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
    tests_label: z.string().default('Unit tests (not shown to the model)'),
  }),
  z.object({
    kind: z.literal('audio'),
    audio: z.string(),
    text: z.string().optional(), // transcript (speech recognition, text-to-speech)
    label: z.string().optional(), // class name, or a list of them (sound classification)
    speaker: z.string().optional(),
  }),
  z.object({ kind: z.literal('grid'), train: z.string(), test: z.string() }),
  z.object({ kind: z.literal('record'), label: z.string().optional() }),
  // A photo with what annotators said about it: a list of captions, or true/false attributes.
  // labels: a list-of-strings field shown as tags (ChestX-ray14's findings), with `none` when empty;
  // details: {Label: field} shown as a small line (age, sex, view).
  z.object({
    kind: z.literal('annotated'), image: z.string(), captions: z.string().optional(), attributes: z.boolean().default(false), thumb: z.number().default(220),
    labels: z.string().optional(), none: z.string().default('none'), details: z.record(z.string(), z.string()).default({}),
  }),
  // Multiple choice over one or more images (MMMU); options may be a list or a Python-style list string.
  z.object({ kind: z.literal('image_mcq'), images: z.array(z.string()), question: z.string(), options: z.string(), answer: z.string(), subject: z.string().optional() }),
  // Two answers to the same conversation, one preferred by a human rater (RLHF data).
  z.object({ kind: z.literal('preference'), chosen: z.string(), rejected: z.string() }),
  // Polygon segmentation (ADE20K-style objects with name, polygon {x, y}, depth rank and part level).
  z.object({ kind: z.literal('segmentation'), image: z.string(), objects: z.string(), scene: z.string().optional() }),
  // Per-pixel annotations rasterized by the sampler (source.dense): label maps, depth, surface normals.
  z.object({
    kind: z.literal('dense'),
    image: z.string(),
    layers: z.array(z.object({
      field: z.string(),
      type: z.enum(['labels', 'depth', 'normals']),
      name: z.string(),
      outline: z.boolean().default(false), // labels: draw region borders only (a lesion outline), not filled masks
      // labels: the pixel value is the id; classes without a color (void, unlabeled) stay transparent
      classes: z.array(z.object({ id: z.number(), name: z.string(), color: z.string().optional() })).optional(),
    })).min(1),
  }),
  // A search query with candidate passages, some marked relevant (MS MARCO); answers shown on demand.
  z.object({
    kind: z.literal('retrieval'), query: z.string(), passages: z.string(), text: z.string().default('passage_text'),
    selected: z.string().default('is_selected'), url: z.string().optional(), answers: z.string().optional(), type: z.string().optional(),
  }),
  // Visual question answering: a photo, a question and several human answers (VQA v2 has ten).
  z.object({
    kind: z.literal('vqa'), image: z.string(), question: z.string(), answers: z.string(), answer_key: z.string().default('answer'),
    type: z.string().optional(),
  }),
  // Scene graphs (Visual Genome): objects with attributes, relationships between them, region phrases, Q&A.
  // Boxes are [x1, y1, x2, y2] as fractions of the image.
  z.object({
    kind: z.literal('scene_graph'), image: z.string(), objects: z.string(), relationships: z.string(),
    regions: z.string().optional(), qa: z.string().optional(),
  }),
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
    tags: z.array(z.enum(Object.keys(TAGS) as [Tag, ...Tag[]])).default([]), // see src/lib/tags.ts
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
      drop: z.array(z.string()).optional(), // columns to leave out of the sample
      max_image_px: z.number().int().optional(), // downscale larger images (annotations keep original coordinates)
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
