// Plain-Markdown version of each dataset page, linked from /llms.txt.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getDatasets, type Dataset } from '../../lib/data';
import { datasetMarkdown, markdownResponse } from '../../lib/llms';

export const getStaticPaths = (async () =>
  (await getDatasets()).map((ds) => ({ params: { id: ds.id }, props: { ds } }))) satisfies GetStaticPaths;

export const GET: APIRoute<{ ds: Dataset }> = ({ props, site }) => markdownResponse(datasetMarkdown(props.ds, site));
