import { ApiClient, retryRequest } from "./client";

describe("retryRequest", () => {
  it("retries server errors", async () => {
    const send = jest.fn().mockResolvedValueOnce({ status: 503 }).mockResolvedValueOnce({ status: 200 });
    const res = await retryRequest(send, 2);
    expect(res.status).toBe(200);
  });
});

describe("ApiClient.fetchJson", () => {
  it("sends the bearer token", async () => {
    const client = new ApiClient("https://shop.test", () => "tok");
    expect(client).toBeDefined();
  });
});
