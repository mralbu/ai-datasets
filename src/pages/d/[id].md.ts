// Plain-Markdown version of each dataset page, linked from /llms.txt.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getDatasets, getPapers, type Dataset, type Paper } from '../../lib/data';
import { datasetMarkdown, markdownResponse } from '../../lib/llms';

export const getStaticPaths = (async () => {
  const papers = await getPapers();
  return (await getDatasets()).map((ds) => ({ params: { id: ds.id }, props: { ds, papers } }));
}) satisfies GetStaticPaths;

export const GET: APIRoute<{ ds: Dataset; papers: Paper[] }> = ({ props, site }) =>
  markdownResponse(datasetMarkdown(props.ds, site, 1, props.papers));
