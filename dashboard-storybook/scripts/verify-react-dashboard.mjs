import { chromium } from "playwright";

const base = process.env.DASHBOARD_URL || "http://localhost:7830/dashboard/";

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.goto(base, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(2000);
const hasRoot = (await page.locator("#root").count()) > 0;
const title = await page.title();
await browser.close();
if (!hasRoot) {
  console.error("FAIL: #root not found");
  process.exit(1);
}
console.log(`PASS: React dashboard title=${title}`);
