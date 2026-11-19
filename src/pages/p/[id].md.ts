// Plain-Markdown version of each paper page, linked from /llms.txt.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getDatasets, getPapers, getTracks, type Dataset, type Paper, type Track } from '../../lib/data';
import { markdownResponse, paperMarkdown } from '../../lib/llms';

export const getStaticPaths = (async () => {
  const [datasets, papers, tracks] = await Promise.all([getDatasets(), getPapers(), getTracks()]);
  return papers.map((p) => ({ params: { id: p.id }, props: { p, datasets, papers, tracks } }));
}) satisfies GetStaticPaths;

export const GET: APIRoute<{ p: Paper; datasets: Dataset[]; papers: Paper[]; tracks: Track[] }> = ({ props, site }) =>
  markdownResponse(paperMarkdown(props.p, site, props.datasets, props.papers, 1, props.tracks));
