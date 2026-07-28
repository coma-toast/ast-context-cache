import { spawn } from "node:child_process";
import { access } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../..");
const staticDir = path.join(repoRoot, "docs/storybook-static");
const port = 6020;

const storiesToCheck = [
  {
    id: "dashboard-overview--hero",
    label: "Overview / Hero",
    textSignals: ["Index & runtime", "Query activity"],
  },
  {
    id: "dashboard-memory--healthy",
    label: "Memory / Healthy",
    textSignals: ["Virtual context", "Memory"],
  },
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

// Rough luminance check so we don't hardcode a specific hex; catches an
// accidental light-theme regression without depending on templ class names.
function isDarkColor(cssColor) {
  const m = cssColor.match(/rgba?\(\s*(\d+),\s*(\d+),\s*(\d+)/);
  if (!m) return null;
  const [r, g, b] = [m[1], m[2], m[3]].map(Number);
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  return luminance < 0.5;
}

async function checkStory(browser, base, story) {
  const failures = [];
  const url = `${base}/iframe.html?id=${story.id}&viewMode=story`;
  const page = await browser.newPage({ viewport: { width: 1280, height: 1800 } });
  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 60000 });

    const frame = page.locator('[data-testid="story-frame"], .story-frame').first();
    const frameCount = await frame.count();
    if (frameCount === 0) {
      failures.push(`${story.label}: [data-testid="story-frame"] / .story-frame not found`);
      return failures;
    }
    await frame.waitFor({ state: "visible", timeout: 30000 });

    const bodyBg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
    const dark = isDarkColor(bodyBg);
    const bodyText = await page.evaluate(() => document.body.innerText || "");
    const hasTextSignal = story.textSignals.some((t) => bodyText.includes(t));

    if (dark === false && !hasTextSignal) {
      failures.push(
        `${story.label}: body background not dark-ish (${bodyBg}) and none of [${story.textSignals.join(", ")}] found in text`,
      );
    } else if (dark === null && !hasTextSignal) {
      failures.push(
        `${story.label}: could not parse body background (${bodyBg}) and none of [${story.textSignals.join(", ")}] found in text`,
      );
    }
  } finally {
    await page.close();
  }
  return failures;
}

async function main() {
  if (!(await exists(staticDir))) {
    console.error("FAIL: run `npm run build-storybook` first (docs/storybook-static missing).");
    process.exit(1);
  }

  const server = serveStatic(staticDir, port);
  const base = `http://127.0.0.1:${port}`;
  const allFailures = [];
  try {
    await waitForServer(`${base}/iframe.html`);
    const browser = await chromium.launch({ headless: true });
    try {
      for (const story of storiesToCheck) {
        const failures = await checkStory(browser, base, story);
        if (failures.length === 0) {
          console.log(`PASS: ${story.label} (${story.id})`);
        } else {
          allFailures.push(...failures);
        }
      }
    } finally {
      await browser.close();
    }
  } finally {
    server.kill("SIGTERM");
  }

  if (allFailures.length > 0) {
    for (const f of allFailures) console.error(`FAIL: ${f}`);
    process.exit(1);
  }
  console.log("PASS: all stories verified");
}

main().catch((err) => {
  console.error("FAIL:", err);
  process.exit(1);
});
