// Optional manual visual-confirm helper.
//
// Compares the built Storybook "Overview / Hero" story against the live
// dashboard (if running) so a maintainer can eyeball parity after changing
// theme/layout. This is a convenience checklist printer, not an automated
// pixel-diff — Storybook fixtures are the source of truth for CI/local
// verification (see verify-stories.mjs).
//
// Alternate maintainer path: Webwright (see ~/.claude/skills/webwright) can
// drive a real browser interactively to eyeball the same two pages if you'd
// rather do this outside of a script.
import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
const staticDir = path.join(repoRoot, "docs/storybook-static");
const port = 6019;

const LIVE_URL = process.env.DASHBOARD_URL || "http://localhost:7830/dashboard/";
const OVERVIEW_STORY_ID = "dashboard-overview--hero";

const CHECKLIST = [
  "Dark theme background matches (near #0d1117, not white/light)",
  '"Index & runtime" section heading present',
  '"Query activity" section heading present',
];

async function exists(p) {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
}

async function isLiveUp(url) {
  try {
    const res = await fetch(url, { method: "GET" });
    return res.ok;
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

async function main() {
  const liveUp = await isLiveUp(LIVE_URL);
  if (!liveUp) {
    console.log(`live dashboard not running; skipped (Storybook-only OK) — tried ${LIVE_URL}`);
    process.exit(0);
  }

  if (!(await exists(staticDir))) {
    console.error("FAIL: run `npm run build-storybook` first (docs/storybook-static missing).");
    process.exit(1);
  }

  const server = serveStatic(staticDir, port);
  const base = `http://127.0.0.1:${port}`;
  const browser = await chromium.launch({ headless: true });
  try {
    await waitForServer(`${base}/iframe.html`);

    console.log(`Opening live dashboard: ${LIVE_URL}`);
    const livePage = await browser.newPage({ viewport: { width: 1280, height: 1800 } });
    await livePage.goto(LIVE_URL, { waitUntil: "networkidle", timeout: 60000 });
    await livePage.waitForTimeout(1000);
    await livePage.close();

    const storyUrl = `${base}/iframe.html?id=${OVERVIEW_STORY_ID}&viewMode=story`;
    console.log(`Opening Storybook Overview: ${storyUrl}`);
    const storyPage = await browser.newPage({ viewport: { width: 1280, height: 1800 } });
    await storyPage.goto(storyUrl, { waitUntil: "networkidle", timeout: 60000 });
    await storyPage.locator('[data-testid="story-frame"], .story-frame').first().waitFor({ state: "visible" });
    await storyPage.waitForTimeout(500);
    await storyPage.close();
  } finally {
    await browser.close();
    server.kill("SIGTERM");
  }

  console.log("\nManual visual confirm checklist (compare the two pages above):");
  for (const item of CHECKLIST) console.log(`  [ ] ${item}`);
  console.log("\nThis script does not fail on mismatch — it's a maintainer aid, not a gate.");
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
