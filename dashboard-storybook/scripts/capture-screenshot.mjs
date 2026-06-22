import { spawn } from "node:child_process";
import { mkdir, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
const staticDir = path.join(repoRoot, "docs/storybook-static");
const outDir = path.join(repoRoot, "docs/images");
const outFile = path.join(outDir, "dashboard-overview.png");

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

function serveStatic(root, port) {
  return spawn("python3", ["-m", "http.server", String(port), "--bind", "127.0.0.1"], {
    cwd: root,
    stdio: "ignore",
  });
}

async function waitForServer(url, attempts = 30) {
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch {
      /* retry */
    }
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`server not ready: ${url}`);
}

async function main() {
  if (!(await exists(staticDir))) {
    console.error("Run npm run build-storybook first (docs/storybook-static missing).");
    process.exit(1);
  }
  await mkdir(outDir, { recursive: true });
  const port = 6018;
  const server = serveStatic(staticDir, port);
  const base = `http://127.0.0.1:${port}`;
  await waitForServer(`${base}/iframe.html`);
  const url = `${base}/iframe.html?id=dashboard-overview--overview&viewMode=story`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(800);
  await page.screenshot({ path: outFile, fullPage: false });
  await browser.close();
  server.kill("SIGTERM");
  console.log("Wrote", outFile);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
