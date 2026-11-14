// Plain-Markdown version of each paper page, linked from /llms.txt.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getDatasets, getPapers, type Dataset, type Paper } from '../../lib/data';
import { markdownResponse, paperMarkdown } from '../../lib/llms';

export const getStaticPaths = (async () => {
  const [datasets, papers] = [await getDatasets(), await getPapers()];
  return papers.map((p) => ({ params: { id: p.id }, props: { p, datasets, papers } }));
}) satisfies GetStaticPaths;

export const GET: APIRoute<{ p: Paper; datasets: Dataset[]; papers: Paper[] }> = ({ props, site }) =>
  markdownResponse(paperMarkdown(props.p, site, props.datasets, props.papers));
