import { spawn } from "node:child_process";
import { appendFile, mkdir, writeFile, access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
const staticDir = path.join(repoRoot, "docs/storybook-static");
const runDir = path.join(repoRoot, "outputs/dashboard-overview-story/final_runs/run_1");
const shotsDir = path.join(runDir, "screenshots");
const logFile = path.join(runDir, "final_script_log.txt");
const port = 6020;

const checks = [
  ["CP1 top bar chrome", ".dashboard-story-root .top-bar", null],
  ["CP1 health bar inline", ".dashboard-story-root .health-bar-inline", null],
  ["CP1 project filter", ".dashboard-story-root .project-select", null],
  ["CP1 panel heading", ".dashboard-story-root .panel-heading h2", "Overview"],
  ["CP2 query activity section", ".dashboard-story-root .section-title", "QUERY ACTIVITY"],
  ["CP2 stat meters", ".dashboard-story-root .metric-stat-card", null],
  ["CP2 stat meter bars", ".dashboard-story-root .stat-meter-stack", null],
  ["CP3 index section", ".dashboard-story-root .section-title", "INDEX & RUNTIME"],
  ["CP3 resource util", ".dashboard-story-root .resource-util-card", null],
  ["CP3 embed panel", ".dashboard-story-root .embed-panel", null],
  ["CP3 corpus bars", ".dashboard-story-root .corpus-bars", null],
  ["CP4 watchers", ".dashboard-story-root .watcher-card", null],
  ["CP5 healthy embed sync badge", ".dashboard-story-root .embed-sync-badge", "in sync"],
  ["CP6 embed recent", ".dashboard-story-root .embed-activity", null],
];

async function log(line) {
  console.log(line);
  await appendFile(logFile, line + "\n");
}

function serveStatic(root, p) {
  return spawn("python3", ["-m", "http.server", String(p), "--bind", "127.0.0.1"], {
    cwd: root,
    stdio: "ignore",
  });
}

async function waitForServer(url) {
  for (let i = 0; i < 30; i++) {
    try {
      if ((await fetch(url)).ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`server not ready: ${url}`);
}

async function main() {
  await writeFile(logFile, "");
  await log("step 0 params: story=dashboard-overview--overview");
  await access(staticDir);
  await mkdir(shotsDir, { recursive: true });
  const server = serveStatic(staticDir, port);
  const base = `http://127.0.0.1:${port}`;
  await waitForServer(`${base}/iframe.html`);
  const url = `${base}/iframe.html?id=dashboard-overview--overview&viewMode=story`;
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1800 } });
  await log(`step 1 action: goto ${url}`);
  await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(1000);
  const missing = [];
  for (const [cp, sel, text] of checks) {
    const loc = page.locator(sel);
    const count = await loc.count();
    if (count === 0) {
      missing.push(cp);
      await log(`FAIL ${cp}: ${sel} not found`);
      continue;
    }
    if (text) {
      let ok = false;
      const want = text.toLowerCase();
      for (let i = 0; i < Math.min(count, 5); i++) {
        if ((await loc.nth(i).innerText()).toLowerCase().includes(want)) {
          ok = true;
          break;
        }
      }
      if (!ok) {
        missing.push(cp);
        await log(`FAIL ${cp}: expected text ${text}`);
        continue;
      }
    }
    await log(`PASS ${cp}`);
  }
  const shot = path.join(shotsDir, "final_execution_1_overview_full.png");
  await page.locator(".dashboard-story-root").screenshot({ path: shot });
  await log(`step 2 action: screenshot ${shot}`);
  await browser.close();
  server.kill("SIGTERM");
  if (missing.length) {
    await log("RESULT: missing: " + missing.join(", "));
    process.exit(1);
  }
  await log("RESULT: all critical points verified");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
