import type { APIRoute } from 'astro';
import { llmsTxt, markdownResponse } from '../lib/llms';

export const GET: APIRoute = async ({ site }) => markdownResponse(await llmsTxt(site));
