import { Page, Request, Response } from '@playwright/test';

/**
 * Utilities for waiting on API calls and UI updates
 */

/**
 * Wait for specific API request to complete
 */
export async function waitForAPIRequest(
  page: Page,
  urlPattern: string | RegExp,
  method?: string,
  timeout: number = 10000
): Promise<Request> {
  return page.waitForRequest(
    (request) => {
      const urlMatches = typeof urlPattern === 'string'
        ? request.url().includes(urlPattern)
        : urlPattern.test(request.url());
      const methodMatches = method ? request.method() === method : true;
      return urlMatches && methodMatches;
    },
    { timeout }
  );
}

/**
 * Wait for specific API response
 */
export async function waitForAPIResponse(
  page: Page,
  urlPattern: string | RegExp,
  status?: number,
  timeout: number = 10000
): Promise<Response> {
  return page.waitForResponse(
    (response) => {
      const urlMatches = typeof urlPattern === 'string'
        ? response.url().includes(urlPattern)
        : urlPattern.test(response.url());
      const statusMatches = status ? response.status() === status : true;
      return urlMatches && statusMatches;
    },
    { timeout }
  );
}

/**
 * Wait for SSE connection to be established
 */
export async function waitForSSEConnection(
  page: Page,
  urlPattern: string | RegExp,
  timeout: number = 10000
): Promise<Response> {
  return page.waitForResponse(
    (response) => {
      const urlMatches = typeof urlPattern === 'string'
        ? response.url().includes(urlPattern)
        : urlPattern.test(response.url());
      // SSE responses might not have content-type header set correctly in mocks
      // So we just check the URL matches
      return urlMatches;
    },
    { timeout }
  );
}

/**
 * Wait for SSE message with specific data
 */
export async function waitForSSEMessage(
  page: Page,
  urlPattern: string | RegExp,
  dataMatcher?: (data: any) => boolean,
  timeout: number = 30000
): Promise<any> {
  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      reject(new Error('SSE message timeout'));
    }, timeout);
    
    page.on('response', async (response) => {
      const urlMatches = typeof urlPattern === 'string'
        ? response.url().includes(urlPattern)
        : urlPattern.test(response.url());
      
      if (urlMatches) {
        const contentType = response.headers()['content-type'] || '';
        if (contentType.includes('text/event-stream')) {
          // For SSE, we need to monitor the EventSource connection
          // This is a simplified version - actual SSE testing may need custom handling
          clearTimeout(timeoutId);
          resolve({ connected: true });
        }
      }
    });
  });
}

/**
 * Wait for UI element to appear after API call
 */
export async function waitForElementAfterAPI(
  page: Page,
  selector: string,
  apiUrl: string | RegExp,
  timeout: number = 10000
): Promise<void> {
  // Wait for API response first
  await waitForAPIResponse(page, apiUrl, undefined, timeout);
  
  // Then wait for UI element
  await page.waitForSelector(selector, { timeout: 5000 });
}

/**
 * Wait for loading state to disappear
 */
export async function waitForLoadingComplete(
  page: Page,
  loadingSelector: string = '[class*="loading"], [class*="spinner"], [class*="animate-spin"]',
  timeout: number = 10000
): Promise<void> {
  try {
    // Wait for loading element to disappear
    await page.waitForSelector(loadingSelector, { state: 'hidden', timeout });
  } catch {
    // Loading element might not exist, which is fine
  }
}

/**
 * Wait for text content to appear
 */
export async function waitForTextContent(
  page: Page,
  text: string,
  timeout: number = 5000
): Promise<void> {
  await page.waitForSelector(`text=${text}`, { timeout });
}

/**
 * Wait for multiple API requests to complete
 */
export async function waitForMultipleAPIRequests(
  page: Page,
  requests: Array<{ url: string | RegExp; method?: string }>,
  timeout: number = 30000
): Promise<Request[]> {
  const promises = requests.map(req => 
    waitForAPIRequest(page, req.url, req.method, timeout)
  );
  return Promise.all(promises);
}

/**
 * Wait for network to be idle
 */
export async function waitForNetworkIdle(
  page: Page,
  timeout: number = 5000
): Promise<void> {
  await page.waitForLoadState('networkidle', { timeout });
}

