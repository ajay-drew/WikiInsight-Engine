import { Request, Response } from '@playwright/test';

/**
 * Utilities to verify API requests and responses
 */

export interface RequestVerificationOptions {
  url?: string | RegExp;
  method?: string;
  body?: any;
  headers?: Record<string, string>;
}

export interface ResponseVerificationOptions {
  status?: number;
  body?: any;
  contentType?: string;
}

/**
 * Verify request URL matches expected endpoint
 */
export function verifyRequestURL(request: Request, expected: string | RegExp): void {
  const url = request.url();
  if (typeof expected === 'string') {
    if (!url.includes(expected)) {
      throw new Error(`Expected URL to contain ${expected}, got ${url}`);
    }
  } else {
    if (!expected.test(url)) {
      throw new Error(`Expected URL to match ${expected}, got ${url}`);
    }
  }
}

/**
 * Verify request method is correct
 */
export function verifyRequestMethod(request: Request, expected: string): void {
  const method = request.method();
  if (method !== expected) {
    throw new Error(`Expected method ${expected}, got ${method}`);
  }
}

/**
 * Deep equality check for arrays and objects
 */
function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    return a.every((val, idx) => deepEqual(val, b[idx]));
  }
  if (typeof a === 'object' && typeof b === 'object') {
    const keysA = Object.keys(a);
    const keysB = Object.keys(b);
    if (keysA.length !== keysB.length) return false;
    return keysA.every(key => deepEqual(a[key], b[key]));
  }
  return false;
}

/**
 * Verify request body matches expected format
 */
export function verifyRequestBody(request: Request, expected: any): void {
  const postData = request.postDataJSON();
  
  if (!postData) {
    throw new Error('Request has no body');
  }
  
  for (const [key, value] of Object.entries(expected)) {
    if (!deepEqual(postData[key], value)) {
      throw new Error(`Expected body.${key} to be ${JSON.stringify(value)}, got ${JSON.stringify(postData[key])}`);
    }
  }
}

/**
 * Verify request headers are correct
 */
export function verifyRequestHeaders(request: Request, expected: Record<string, string>): void {
  const headers = request.headers();
  
  for (const [key, value] of Object.entries(expected)) {
    const actualValue = headers[key.toLowerCase()];
    if (actualValue !== value) {
      throw new Error(`Expected header ${key} to be ${value}, got ${actualValue}`);
    }
  }
}

/**
 * Verify response status code
 */
export function verifyResponseStatus(response: Response, expected: number): void {
  const status = response.status();
  if (status !== expected) {
    throw new Error(`Expected status ${expected}, got ${status}`);
  }
}

/**
 * Verify response body
 */
export async function verifyResponseBody(response: Response, expected: any): Promise<void> {
  const body = await response.json();
  
  for (const [key, value] of Object.entries(expected)) {
    if (body[key] !== value) {
      throw new Error(`Expected response.${key} to be ${JSON.stringify(value)}, got ${JSON.stringify(body[key])}`);
    }
  }
}

/**
 * Comprehensive request verification
 */
export function verifyRequest(
  request: Request,
  options: RequestVerificationOptions
): void {
  if (options.url) {
    verifyRequestURL(request, options.url);
  }
  
  if (options.method) {
    verifyRequestMethod(request, options.method);
  }
  
  if (options.body) {
    verifyRequestBody(request, options.body);
  }
  
  if (options.headers) {
    verifyRequestHeaders(request, options.headers);
  }
}

/**
 * Comprehensive response verification
 */
export async function verifyResponse(
  response: Response,
  options: ResponseVerificationOptions
): Promise<void> {
  if (options.status !== undefined) {
    verifyResponseStatus(response, options.status);
  }
  
  if (options.body) {
    await verifyResponseBody(response, options.body);
  }
  
  if (options.contentType) {
    const contentType = response.headers()['content-type'];
    if (!contentType?.includes(options.contentType)) {
      throw new Error(`Expected content-type to include ${options.contentType}, got ${contentType}`);
    }
  }
}

