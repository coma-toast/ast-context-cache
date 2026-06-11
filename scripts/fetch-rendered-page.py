#!/usr/bin/env python3
"""Fetch fully rendered HTML from a URL using Playwright Firefox."""

from __future__ import annotations

import argparse
import asyncio
import sys

from playwright.async_api import async_playwright

VIEWPORT = {"width": 1280, "height": 1800}


async def fetch_rendered_page(url: str, timeout_ms: int, wait_ms: int) -> str:
    """Load a URL in headless Firefox and return the post-JS DOM HTML.

    Args:
        url: Page URL to open. Must be http or https.
            Default: none (required).
        timeout_ms: Maximum time to wait for navigation (domcontentloaded).
            Default: 60000.
        wait_ms: Extra milliseconds after load for client-side rendering.
            Default: 2000.

    Returns:
        Full page HTML string from page.content() after JS execution.
    """
    async with async_playwright() as playwright:
        browser = await playwright.firefox.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        if wait_ms > 0:
            await page.wait_for_timeout(wait_ms)
        html = await page.content()
        await browser.close()
        return html


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch JS-rendered page HTML via Playwright Firefox")
    parser.add_argument("--url", required=True, help="URL to fetch")
    parser.add_argument("--timeout-ms", type=int, default=60000, help="Navigation timeout in ms")
    parser.add_argument("--wait-ms", type=int, default=2000, help="Post-load wait for SPA hydration in ms")
    args = parser.parse_args()
    try:
        html = asyncio.run(fetch_rendered_page(args.url, args.timeout_ms, args.wait_ms))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
