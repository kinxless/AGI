"""BrowserManager: single persistent Chromium instance via Playwright sync API.

A module-level singleton is used so every tool call shares one browser
session rather than launching a new one each time.

On headless RunPod servers, headless=False requires a virtual display.
Set BROWSER_HEADLESS=true in .env or prepend:
    Xvfb :99 -screen 0 1280x900x24 &
    export DISPLAY=:99
before running the agent. browser_tools.py does this automatically.
"""
from __future__ import annotations

from typing import Optional

from config import BROWSER_HEADLESS

# Playwright is an optional dependency — fail gracefully so the rest of the
# agent still works if it isn't installed.
try:
    from playwright.sync_api import sync_playwright, Browser, Page, Playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


class BrowserManager:
    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    def is_running(self) -> bool:
        return self._browser is not None and self._page is not None

    def start(self) -> None:
        if self.is_running():
            return
        if not _PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "playwright is not installed. Run: pip install playwright && "
                "playwright install chromium"
            )
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=BROWSER_HEADLESS,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1280,900",
            ],
        )
        self._page = self._browser.new_page(viewport={"width": 1280, "height": 900})

    def stop(self) -> None:
        try:
            if self._page:
                self._page.close()
        except Exception:
            pass
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._page = None
        self._browser = None
        self._playwright = None

    def get_page(self) -> Page:
        if not self.is_running():
            self.start()
        return self._page  # type: ignore[return-value]

    def navigate(self, url: str) -> None:
        page = self.get_page()
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)

    def screenshot(self) -> bytes:
        return self.get_page().screenshot(type="png", timeout=30_000)

    def get_page_text(self) -> str:
        try:
            text = self.get_page().inner_text("body", timeout=10_000)
            return text[:3000]
        except Exception as e:
            return f"Could not extract page text: {e}"

    def click_at(self, x: int, y: int) -> None:
        self.get_page().mouse.click(x, y, timeout=30_000)

    def type_text(self, text: str) -> None:
        self.get_page().keyboard.type(text)

    def press_key(self, key: str) -> None:
        self.get_page().keyboard.press(key)

    def scroll(self, direction: str) -> None:
        delta = -400 if direction.lower() == "up" else 400
        self.get_page().mouse.wheel(0, delta)


_browser_manager: Optional[BrowserManager] = None


def get_browser_manager() -> BrowserManager:
    global _browser_manager
    if _browser_manager is None:
        _browser_manager = BrowserManager()
    return _browser_manager
