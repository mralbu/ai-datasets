import katex from 'katex';

const escape = (t: string) => t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// $$…$$, \[…\], \(…\) and $…$ (an escaped \$ is a literal dollar sign).
const MATH = /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<!\\)\$((?:\\\$|[^$])+?)\$/g;

/** Text with LaTeX math (as in the MATH dataset) to HTML, rendered at build time with KaTeX. */
export function renderMath(text: string): string {
  // Geometry problems draw their figures with Asymptote code; it is not shown.
  const src = text.replace(/\[asy\][\s\S]*?\[\/asy\]/g, '\u0000');
  let out = '';
  let last = 0;
  for (const m of src.matchAll(MATH)) {
    out += prose(src.slice(last, m.index));
    const display = m[1] !== undefined || m[2] !== undefined;
    const tex = m[1] ?? m[2] ?? m[3] ?? m[4];
    out += katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: 'ignore' });
    last = m.index! + m[0].length;
  }
  return out + prose(src.slice(last));
}

const prose = (t: string) =>
  escape(t.replace(/\\\$/g, '$')).replace(/\u0000/g, '<span class="asy">[figure drawn with Asymptote code, not shown]</span>');
