import { Page, Route, Request, Response } from '@playwright/test';

/**
 * Helper functions for API request interception and verification
 */

export interface RequestVerification {
  url: string | RegExp;
  method?: string;
  body?: any;
  headers?: Record<string, string>;
}

/**
 * Intercept and verify an API request
 */
export async function interceptAndVerify(
  page: Page,
  verification: RequestVerification,
  mockResponse?: any
): Promise<Request> {
  return new Promise((resolve, reject) => {
    page.route(verification.url, async (route: Route) => {
      const request = route.request();
      
      // Verify method
      if (verification.method && request.method() !== verification.method) {
        reject(new Error(`Expected method ${verification.method}, got ${request.method()}`));
        return;
      }
      
      // Verify body for POST requests
      if (verification.body && request.method() === 'POST') {
        const postData = request.postDataJSON();
        for (const [key, value] of Object.entries(verification.body)) {
          if (postData[key] !== value) {
            reject(new Error(`Expected body.${key} to be ${value}, got ${postData[key]}`));
            return;
          }
        }
      }
      
      // Verify headers
      if (verification.headers) {
        const headers = request.headers();
        for (const [key, value] of Object.entries(verification.headers)) {
          if (headers[key] !== value) {
            reject(new Error(`Expected header ${key} to be ${value}, got ${headers[key]}`));
            return;
          }
        }
      }
      
      // Mock response or continue
      if (mockResponse) {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(mockResponse),
        });
      } else {
        await route.continue();
      }
      
      resolve(request);
    });
  });
}

/**
 * Wait for a specific API request to complete
 */
export async function waitForRequest(
  page: Page,
  url: string | RegExp,
  method?: string
): Promise<Request> {
  return page.waitForRequest((request) => {
    const urlMatches = typeof url === 'string' 
      ? request.url().includes(url)
      : url.test(request.url());
    const methodMatches = method ? request.method() === method : true;
    return urlMatches && methodMatches;
  });
}

/**
 * Wait for a specific API response
 */
export async function waitForResponse(
  page: Page,
  url: string | RegExp,
  status?: number
): Promise<Response> {
  return page.waitForResponse((response) => {
    const urlMatches = typeof url === 'string'
      ? response.url().includes(url)
      : url.test(response.url());
    const statusMatches = status ? response.status() === status : true;
    return urlMatches && statusMatches;
  });
}

/**
 * Verify request payload matches expected format
 */
export function verifyRequestPayload(request: Request, expected: any): void {
  const postData = request.postDataJSON();
  
  for (const [key, value] of Object.entries(expected)) {
    if (postData[key] !== value) {
      throw new Error(`Expected ${key} to be ${value}, got ${postData[key]}`);
    }
  }
}

/**
 * Create a mock response for an API endpoint
 */
export function createMockResponse(data: any, status: number = 200): any {
  return {
    status,
    contentType: 'application/json',
    body: JSON.stringify(data),
  };
}

/**
 * Create a mock error response
 */
export function createMockErrorResponse(
  message: string,
  status: number = 500
): any {
  return {
    status,
    contentType: 'application/json',
    body: JSON.stringify({ detail: message }),
  };
}

/**
 * Wait for UI element to appear after API call
 */
export async function waitForUIUpdate(
  page: Page,
  selector: string,
  timeout: number = 5000
): Promise<void> {
  await page.waitForSelector(selector, { timeout });
}

/**
 * Check if element is visible
 */
export async function isVisible(
  page: Page,
  selector: string
): Promise<boolean> {
  try {
    const element = await page.locator(selector);
    return await element.isVisible();
  } catch {
    return false;
  }
}

/**
 * Get text content of element
 */
export async function getText(
  page: Page,
  selector: string
): Promise<string | null> {
  try {
    const element = await page.locator(selector);
    return await element.textContent();
  } catch {
    return null;
  }
}

