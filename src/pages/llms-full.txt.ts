import type { APIRoute } from 'astro';
import { llmsFullTxt, markdownResponse } from '../lib/llms';

export const GET: APIRoute = async ({ site }) => markdownResponse(await llmsFullTxt(site));
