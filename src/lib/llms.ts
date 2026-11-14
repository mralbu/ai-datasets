// Plain-Markdown versions of the site for language models (https://llmstxt.org):
// /llms.txt (index), /llms-full.txt (every entry in full) and /d/<id>.md (one entry).
import { getDatasets, getSamples, MODALITIES, TAGS, url, type Dataset } from './data';
import { snippets, downloadRepo } from './snippets';

/** Absolute URL of a site path, e.g. https://mralbu.github.io/ai-datasets/d/mnist. */
export const abs = (site: URL | undefined, path: string) => (site ? new URL(url(path), site).href : url(path));

const hf = (repo: string) => `https://huggingface.co/datasets/${repo}`;
const oneLine = (s: string) => s.replace(/\s+/g, ' ').trim();

export const SUMMARY =
  'A small, curated museum of classic AI datasets and benchmarks across vision, language models, text, ' +
  'tabular data, time series and audio. Each entry explains why the dataset mattered, lists the key facts, shows real ' +
  'samples from the Hugging Face dataset viewer, and gives copy-paste snippets to download the full data.';

const ABOUT = `## About the site

- Entries are chosen because they changed the field, were the standard benchmark for years, or appear in every introductory course. It is not a catalog: Hugging Face already is one.
- "LLMs" is the category for language-model benchmarks and training data (instructions, preferences).
- Samples are a fixed random selection of rows taken once from the Hugging Face dataset viewer, so pages are stable between visits. Where the official copy cannot be read by the viewer a community copy is used; each entry names the exact repository.
- Samples stay under each dataset's own license. Gated benchmarks whose authors ask that questions not be published (GPQA, Humanity's Last Exam) are linked but not sampled.
- Download snippets use the \`hf\` CLI, the 🤗 \`datasets\` library and Polars (\`hf://datasets/<repo>@~parquet/<config>/<split>/*.parquet\`). Check license and size first: some datasets are research-only, some are hundreds of gigabytes.`;

/** Full Markdown description of one dataset. `level` is the heading level of its title. */
export function datasetMarkdown(ds: Dataset, site: URL | undefined, level = 1): string {
  const d = ds.data;
  const h = (n: number) => '#'.repeat(level + n);
  const byId = (id: string) => abs(site, `/d/${id}`);
  const splits = Object.entries(d.splits).map(([k, v]) => `${k} ${v.toLocaleString('en-US')}`).join(', ');
  const tags = [MODALITIES[d.modality], ...d.tags.map((t) => TAGS[t])].join(', ');
  const s = getSamples(ds.id);

  const out: string[] = [
    `${h(0)} ${d.name} (${d.year})`,
    '',
    `> ${oneLine(d.one_liner)}`,
    '',
    `Page: ${byId(ds.id)}`,
    '',
    `${h(1)} Key facts`,
    '',
    `- Category: ${tags}`,
    `- Task: ${d.task}`,
    `- Creators: ${d.creators}`,
    `- Size: ${d.size}`,
    `- Splits: ${splits}`,
    `- Input: ${d.input}`,
    `- Labels: ${d.labels}`,
    `- License: ${d.license}`,
    d.paper && `- Paper: [${d.paper.title}](${d.paper.url})`,
    d.homepage && `- Homepage: ${d.homepage}`,
    `- Hugging Face: [${downloadRepo(ds)}](${hf(downloadRepo(ds))})`,
    downloadRepo(ds) !== d.source.hf && `- Samples on the site come from: [${d.source.hf}](${hf(d.source.hf)})`,
    s?.revision && `- Sampled at Hub revision ${s.revision} (${s.fetched_at.slice(0, 10)})`,
    '',
    `${h(1)} Why it matters`,
    '',
    oneLine(d.why),
  ].filter((x): x is string => x !== undefined && x !== false && x !== null);

  if (d.trivia.length) out.push('', `${h(1)} Worth knowing`, '', ...d.trivia.map((t) => `- ${oneLine(t)}`));

  const notes = Object.entries(d.config_notes);
  if (notes.length) out.push('', `${h(1)} Configs`, '', ...notes.map(([k, v]) => `- \`${k}\`: ${oneLine(v)}`));

  out.push('', `${h(1)} Get the data`, '');
  if (d.download.gated) out.push(`Gated: log in and accept the terms on ${hf(downloadRepo(ds))} first.`, '');
  if (d.download.note) out.push(oneLine(d.download.note), '');
  for (const sn of snippets(ds)) out.push('```' + sn.lang, sn.code, '```', '');
  out.pop();

  if (d.lineage.length) {
    out.push('', `${h(1)} Related datasets`, '');
    for (const l of d.lineage) {
      const href = l.hf ? hf(l.hf) : l.url;
      const name = href ? `[${l.name}](${href})` : l.name;
      out.push(`- ${name}${l.note ? `: ${l.note}` : ''}${l.id ? ` (on this site: ${byId(l.id)})` : ''}`);
    }
  }
  if (d.landmark_papers.length) {
    out.push('', `${h(1)} Landmark papers`, '', ...d.landmark_papers.map((p) => `- [${p.title}](${p.url}) (${p.year})`));
  }
  return out.join('\n') + '\n';
}

async function grouped() {
  const all = await getDatasets();
  return Object.keys(MODALITIES)
    .map((m) => ({ m, items: all.filter((d) => d.data.modality === m) }))
    .filter((g) => g.items.length);
}

/** /llms.txt: what the site is, then one linked line per dataset, grouped by category. */
export async function llmsTxt(site: URL | undefined): Promise<string> {
  const groups = await grouped();
  const count = groups.reduce((n, g) => n + g.items.length, 0);
  const out = [
    '# AI Datasets',
    '',
    `> ${SUMMARY}`,
    '',
    `The site has ${count} datasets. Every dataset page has a plain-Markdown version at the same URL plus \`.md\` ` +
      `(for example ${abs(site, '/d/mnist.md')}); ${abs(site, '/llms-full.txt')} has all of them in one file. ` +
      `The source, one YAML file per dataset, is at https://github.com/mralbu/ai-datasets.`,
    '',
    ABOUT,
  ];
  for (const { m, items } of groups) {
    out.push('', `## ${MODALITIES[m]}`, '');
    for (const ds of items) {
      const tags = ds.data.tags.length ? ` [${ds.data.tags.map((t) => TAGS[t]).join(', ')}]` : '';
      out.push(`- [${ds.data.name} (${ds.data.year})](${abs(site, `/d/${ds.id}.md`)}): ${oneLine(ds.data.one_liner)} ${ds.data.task}.${tags}`);
    }
  }
  out.push(
    '',
    '## Optional',
    '',
    `- [All datasets in full](${abs(site, '/llms-full.txt')}): every dataset entry in one file`,
    `- [Catalog](${abs(site, '/')}): the HTML catalog, filterable by category and by tag (\`?tag=medical\`, \`remote-sensing\`, \`driving\`, \`retrieval\`)`,
    `- [Timeline](${abs(site, '/timeline')}): every dataset by year`,
    `- [About](${abs(site, '/about')}): how entries are chosen and where samples come from`,
    `- [Source on GitHub](https://github.com/mralbu/ai-datasets): the Astro site, the dataset YAML files and the sampling script`,
  );
  return out.join('\n') + '\n';
}

/** /llms-full.txt: the index header, then every dataset in full. */
export async function llmsFullTxt(site: URL | undefined): Promise<string> {
  const groups = await grouped();
  const out = ['# AI Datasets', '', `> ${SUMMARY}`, '', `Index: ${abs(site, '/llms.txt')}`, '', ABOUT];
  for (const { m, items } of groups) {
    out.push('', `## ${MODALITIES[m]}`);
    for (const ds of items) out.push('', datasetMarkdown(ds, site, 3).trimEnd());
  }
  return out.join('\n') + '\n';
}

export const markdownResponse = (body: string) =>
  new Response(body, { headers: { 'Content-Type': 'text/markdown; charset=utf-8' } });
