import React from "react";

/**
 * Highlights search terms in text by wrapping matches in <mark> tags.
 * Case-insensitive matching while preserving original case.
 */
export function highlightText(text: string, query: string): string {
  if (!query.trim() || !text) {
    return text;
  }

  // Escape special regex characters in query
  const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  
  // Split query into words and filter out empty strings
  const queryWords = escapedQuery
    .split(/\s+/)
    .filter(word => word.length > 0);

  if (queryWords.length === 0) {
    return text;
  }

  // Create regex pattern that matches any of the query words (case-insensitive)
  const pattern = new RegExp(`(${queryWords.join("|")})`, "gi");
  
  // Split text by matches and wrap matches in <mark> tags
  const parts = text.split(pattern);
  
  return parts
    .map((part, index) => {
      // Check if this part matches any query word (case-insensitive)
      const isMatch = queryWords.some(
        word => part.toLowerCase() === word.toLowerCase()
      );
      
      if (isMatch) {
        return `<mark class="bg-yellow-500/30 text-yellow-200 px-0.5 rounded">${part}</mark>`;
      }
      return part;
    })
    .join("");
}

/**
 * React component version that returns JSX with highlighted text.
 */
export function HighlightedText({ text, query }: { text: string; query: string }) {
  const highlighted = highlightText(text, query);
  return <span dangerouslySetInnerHTML={{ __html: highlighted }} />;
}

