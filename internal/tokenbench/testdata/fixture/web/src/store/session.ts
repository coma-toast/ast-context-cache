import { ApiClient } from "../api/client";

export interface User {
  id: string;
  email: string;
}

/** SessionStore keeps the signed-in user and their token in memory. */
export class SessionStore {
  private user: User | null = null;
  private token: string | null = null;

  constructor(private api: ApiClient) {}

  /** login exchanges credentials for a token and loads the user profile. */
  async login(email: string, password: string): Promise<User> {
    const { token } = await this.api.fetchJson<{ token: string }>("/login", { method: "POST", body: { email, password } });
    this.token = token;
    this.user = await this.api.fetchJson<User>("/me");
    return this.user;
  }

  /** logout clears the session. */
  logout(): void {
    this.user = null;
    this.token = null;
  }

  currentToken(): string | null {
    return this.token;
  }
}
