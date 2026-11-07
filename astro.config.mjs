// @ts-check
import { defineConfig } from 'astro/config';

// GitHub Pages serves project sites from /<repo>/. Override locally with BASE_PATH=/.
export default defineConfig({
  site: process.env.SITE_URL ?? 'https://mralbu.github.io',
  base: process.env.BASE_PATH ?? '/ai-datasets',
  trailingSlash: 'ignore',
  // vega-embed is one large chunk, loaded only on pages with a chart.
  vite: { build: { chunkSizeWarningLimit: 1500 } },
});
