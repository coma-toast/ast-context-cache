export interface RequestOptions {
  method?: string;
  body?: unknown;
  retries?: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** ApiClient calls the shop API with the session's bearer token. */
export class ApiClient {
  constructor(private baseUrl: string, private token: () => string | null) {}

  /** fetchJson sends a request and decodes the JSON body, throwing ApiError on failure. */
  async fetchJson<T>(path: string, opts: RequestOptions = {}): Promise<T> {
    const res = await retryRequest(() =>
      fetch(this.baseUrl + path, {
        method: opts.method ?? "GET",
        headers: { Authorization: `Bearer ${this.token() ?? ""}`, "Content-Type": "application/json" },
        body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      }), opts.retries ?? 2);
    if (!res.ok) {
      throw new ApiError(res.status, await res.text());
    }
    return (await res.json()) as T;
  }
}

/** retryRequest retries a request on 5xx responses with exponential backoff. */
export async function retryRequest(send: () => Promise<Response>, retries: number): Promise<Response> {
  let res = await send();
  for (let attempt = 0; attempt < retries && res.status >= 500; attempt++) {
    await new Promise((r) => setTimeout(r, 100 * 2 ** attempt));
    res = await send();
  }
  return res;
}
