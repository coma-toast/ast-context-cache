import { CartService } from "./cart";

describe("CartService", () => {
  it("merges quantities for the same sku", () => {
    const cart = new CartService();
    cart.addItem({ sku: "a", quantity: 1, priceCents: 100 });
    cart.addItem({ sku: "a", quantity: 2, priceCents: 100 });
    expect(cart.totalCents()).toBe(300);
  });
});
