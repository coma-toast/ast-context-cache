import { formatCurrency } from "../utils/format";

export interface CartItem {
  sku: string;
  quantity: number;
  priceCents: number;
}

/** CartService holds the shopping cart and computes its total. */
export class CartService {
  private items = new Map<string, CartItem>();

  /** addItem adds quantity of sku, merging with an existing line. */
  addItem(item: CartItem): void {
    const existing = this.items.get(item.sku);
    this.items.set(item.sku, existing ? { ...existing, quantity: existing.quantity + item.quantity } : item);
  }

  /** removeItem drops the line for sku. */
  removeItem(sku: string): void {
    this.items.delete(sku);
  }

  totalCents(): number {
    let total = 0;
    this.items.forEach((i) => (total += i.priceCents * i.quantity));
    return total;
  }

  /** displayTotal is the cart total formatted for the checkout button. */
  displayTotal(): string {
    return formatCurrency(this.totalCents());
  }
}
