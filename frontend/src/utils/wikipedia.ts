/**
 * Utility functions for Wikipedia URL generation.
 */

export function getWikipediaUrl(title: string): string {
  // Replace spaces with underscores and URL encode
  const encoded = title.replace(/\s+/g, "_");
  return `https://en.wikipedia.org/wiki/${encodeURIComponent(encoded)}`;
}

