// Cross-cutting tags (domains, tasks), shown next to the category. Categories say what the data is
// (vision, text…); tags say where it comes from or what it is for, so one entry can carry several.
export const TAGS = {
  medical: 'Medical',
  'remote-sensing': 'Remote sensing',
  driving: 'Driving',
  retrieval: 'Retrieval',
} as const;
export type Tag = keyof typeof TAGS;
