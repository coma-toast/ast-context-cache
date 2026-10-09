import { SessionStore } from "./session";

jest.mock("../api/client");

describe("SessionStore", () => {
  it("clears the token on logout", () => {
    const store = new SessionStore({ fetchJson: jest.fn() } as never);
    store.logout();
    expect(store.currentToken()).toBeNull();
  });
});
