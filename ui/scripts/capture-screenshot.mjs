import { spawn } from "node:child_process";
import { mkdir, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
const staticDir = path.join(repoRoot, "docs/storybook-static");
const outDir = path.join(repoRoot, "docs/images");
const port = 6018;

// Story ids come from ui/src/storybook/STORY_IDS.md — keep both in sync.
const shots = [
  { id: "dashboard-overview--hero", file: "dashboard-overview.png" },
  { id: "dashboard-overview--index-runtime", file: "dashboard-overview-index-runtime.png" },
  { id: "dashboard-memory--healthy", file: "dashboard-memory.png" },
  { id: "dashboard-memory--empty-docs", file: "dashboard-memory-docs.png" },
  { id: "dashboard-embeddings--healthy", file: "dashboard-embeddings.png" },
  { id: "dashboard-embeddings--degraded", file: "dashboard-embeddings-degraded.png" },
  { id: "dashboard-settings--embedding-and-virtual", file: "dashboard-settings.png" },
];

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

function serveStatic(root, servePort) {
  return spawn("python3", ["-m", "http.server", String(servePort), "--bind", "127.0.0.1"], {
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

async function captureOne(browser, base, id, outFile) {
  const url = `${base}/iframe.html?id=${id}&viewMode=story`;
  const page = await browser.newPage({ viewport: { width: 1280, height: 1800 } });
  await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
  const frame = page.locator('[data-testid="story-frame"], .story-frame').first();
  await frame.waitFor({ state: "visible", timeout: 30000 });
  await page.waitForTimeout(500);
  await frame.screenshot({ path: outFile });
  await page.close();
  console.log("Wrote", outFile);
}

async function main() {
  if (!(await exists(staticDir))) {
    console.error("Run npm run build-storybook first (docs/storybook-static missing).");
    process.exit(1);
  }
  await mkdir(outDir, { recursive: true });
  const server = serveStatic(staticDir, port);
  const base = `http://127.0.0.1:${port}`;
  try {
    await waitForServer(`${base}/iframe.html`);
    const browser = await chromium.launch({ headless: true });
    try {
      for (const { id, file } of shots) {
        await captureOne(browser, base, id, path.join(outDir, file));
      }
    } finally {
      await browser.close();
    }
  } finally {
    server.kill("SIGTERM");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
