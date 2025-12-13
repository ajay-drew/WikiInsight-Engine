/**
 * Utility functions for Wikidata URL generation.
 */

export function getWikidataUrl(qid: string): string {
  return `https://www.wikidata.org/wiki/${qid}`;
}

