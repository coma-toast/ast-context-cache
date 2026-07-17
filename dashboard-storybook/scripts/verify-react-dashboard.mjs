import { chromium } from 'playwright'

const base = process.env.DASHBOARD_URL || 'http://localhost:7830/dashboard/'
const tabs = ['Overview', 'Memory', 'Activity', 'Analytics', 'Recent', 'Settings']

const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } })
const pageErrors = []
page.on('pageerror', (err) => pageErrors.push(String(err)))

await page.goto(base, { waitUntil: 'networkidle', timeout: 120000 })
await page.waitForTimeout(2000)

const hasRoot = (await page.locator('#root').count()) > 0
if (!hasRoot) {
  console.error('FAIL: #root not found')
  process.exit(1)
}

const failures = []
for (const label of tabs) {
  await page.getByRole('button', { name: label, exact: true }).click()
  await page.waitForTimeout(1500)
  const errBoundary = await page.getByText(/failed to render/i).count()
  if (errBoundary > 0) failures.push(`${label}: error boundary`)
  const main = await page.locator('main').innerText()
  if (/^Loading…$/m.test(main.trim()) || main.length < 40) failures.push(`${label}: stuck loading`)
}

const title = await page.title()
await browser.close()

if (pageErrors.length) {
  console.error('FAIL: page errors:', [...new Set(pageErrors)].join('; '))
  process.exit(1)
}
if (failures.length) {
  console.error('FAIL:', failures.join(', '))
  process.exit(1)
}

console.log(`PASS: React dashboard (${tabs.length} tabs) title=${title}`)
