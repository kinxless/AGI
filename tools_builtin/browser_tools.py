"""Browser tools: Playwright-based web interaction + Qwen2.5-VL vision analysis.

All tools catch every exception and return it as a string so the agent loop
never crashes due to a browser or vision error.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime

from pydantic import BaseModel, Field

from agent.tools import register_tool
from agent.browser import get_browser_manager
from agent.vision import get_vision_analyzer
from config import SCREENSHOT_DIR

os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_screenshot(png: bytes) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(SCREENSHOT_DIR, f"screenshot_{ts}.png")
    with open(path, "wb") as f:
        f.write(png)
    return path


def _screenshot_and_ask(question: str) -> tuple[bytes, str, str]:
    """Take screenshot, save it, ask vision model. Returns (bytes, path, answer)."""
    bm = get_browser_manager()
    png = bm.screenshot()
    path = _save_screenshot(png)
    answer = get_vision_analyzer().analyze(png, question)
    return png, path, answer


def _parse_coords(text: str) -> tuple[int, int]:
    """Extract {x, y} from vision model output, even if it adds extra prose."""
    match = re.search(r'\{[^}]*"x"\s*:\s*(\d+)[^}]*"y"\s*:\s*(\d+)[^}]*\}', text)
    if not match:
        match = re.search(r'\{[^}]*"y"\s*:\s*(\d+)[^}]*"x"\s*:\s*(\d+)[^}]*\}', text)
        if match:
            return int(match.group(2)), int(match.group(1))
        raise ValueError(f"No coordinate JSON found in: {text[:200]}")
    return int(match.group(1)), int(match.group(2))


def _ensure_display() -> None:
    """Start Xvfb on :99 if DISPLAY is not set (headless RunPod server)."""
    if os.environ.get("DISPLAY"):
        return
    try:
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1280x900x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)
        os.environ["DISPLAY"] = ":99"
    except FileNotFoundError:
        pass  # Xvfb not available; browser will use its own headless mode


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class OpenBrowserArgs(BaseModel):
    url: str = Field(..., description="Full URL to open, including https://")


@register_tool(
    "open_browser",
    "Open the browser and navigate to a URL. Must be called before any other browser tools.",
    OpenBrowserArgs,
)
def open_browser(url: str) -> str:
    try:
        _ensure_display()
        bm = get_browser_manager()
        bm.start()
        bm.navigate(url)
        return f"Opened {url}"
    except Exception as e:
        return f"ERROR open_browser: {e}"


class TakeScreenshotArgs(BaseModel):
    question: str = Field(
        default="What is on this screen? Describe it in detail.",
        description="Question to ask the vision model about the screenshot.",
    )


@register_tool(
    "take_screenshot",
    "Take a screenshot of the current browser page and analyze it with the vision model.",
    TakeScreenshotArgs,
)
def take_screenshot(question: str = "What is on this screen? Describe it in detail.") -> str:
    try:
        bm = get_browser_manager()
        if not bm.is_running():
            return "Browser not open. Call open_browser first."
        _, path, answer = _screenshot_and_ask(question)
        return f"Screenshot saved to {path}\nVision: {answer}"
    except Exception as e:
        return f"ERROR take_screenshot: {e}"


class ClickElementArgs(BaseModel):
    description: str = Field(
        ..., description="Plain-English description of the element to click."
    )


@register_tool(
    "click_element",
    "Click a visible element on the page, described in plain English. Uses vision model to find coordinates.",
    ClickElementArgs,
)
def click_element(description: str) -> str:
    try:
        bm = get_browser_manager()
        if not bm.is_running():
            return "Browser not open. Call open_browser first."

        png = bm.screenshot()
        _save_screenshot(png)
        coord_response = get_vision_analyzer().analyze(
            png,
            f'I need to click on: {description}. '
            f'What are the x,y pixel coordinates of that element? '
            f'Reply ONLY with JSON: {{"x": int, "y": int}}',
        )
        x, y = _parse_coords(coord_response)
        bm.click_at(x, y)
        time.sleep(0.5)

        _, path, after = _screenshot_and_ask("Describe what is now on the screen.")
        return f"Clicked {description} at ({x},{y}). Screenshot: {path}\nNew screen: {after}"
    except Exception as e:
        return f"ERROR click_element: {e}"


class TypeTextArgs(BaseModel):
    text: str = Field(..., description="Text to type at the current cursor position.")


@register_tool(
    "type_text",
    "Type text into the currently focused input field in the browser.",
    TypeTextArgs,
)
def type_text(text: str) -> str:
    try:
        bm = get_browser_manager()
        if not bm.is_running():
            return "Browser not open. Call open_browser first."
        bm.type_text(text)
        return f"Typed: {text}"
    except Exception as e:
        return f"ERROR type_text: {e}"


class PressKeyArgs(BaseModel):
    key: str = Field(..., description="Key to press, e.g. Enter, Tab, Escape, ArrowDown.")


@register_tool(
    "press_key",
    "Press a keyboard key in the browser (Enter, Tab, Escape, ArrowDown, etc.).",
    PressKeyArgs,
)
def press_key(key: str) -> str:
    try:
        bm = get_browser_manager()
        if not bm.is_running():
            return "Browser not open. Call open_browser first."
        bm.press_key(key)
        return f"Pressed: {key}"
    except Exception as e:
        return f"ERROR press_key: {e}"


class ScrollPageArgs(BaseModel):
    direction: str = Field(
        default="down", description="Scroll direction: 'up' or 'down'."
    )


@register_tool(
    "scroll_page",
    "Scroll the current browser page up or down.",
    ScrollPageArgs,
)
def scroll_page(direction: str = "down") -> str:
    try:
        bm = get_browser_manager()
        if not bm.is_running():
            return "Browser not open. Call open_browser first."
        bm.scroll(direction)
        return f"Scrolled {direction}"
    except Exception as e:
        return f"ERROR scroll_page: {e}"


class GetPageTextArgs(BaseModel):
    pass


@register_tool(
    "get_page_text",
    "Get the visible text content of the current browser page (truncated to 3000 chars).",
    GetPageTextArgs,
)
def get_page_text() -> str:
    try:
        bm = get_browser_manager()
        if not bm.is_running():
            return "Browser not open. Call open_browser first."
        text = bm.get_page_text()
        return f"Page text (truncated to 3000 chars):\n{text}"
    except Exception as e:
        return f"ERROR get_page_text: {e}"


class SearchWebArgs(BaseModel):
    query: str = Field(..., description="Search query to enter into Google.")


@register_tool(
    "search_web",
    "Search the web for a query and return a vision-model summary of the results.",
    SearchWebArgs,
)
def search_web(query: str) -> str:
    try:
        from urllib.parse import quote_plus
        _ensure_display()
        bm = get_browser_manager()
        bm.start()
        # Navigate directly to results — bypasses GDPR/cookie consent dialogs
        url = f"https://duckduckgo.com/?q={quote_plus(query)}&kl=us-en"
        bm.navigate(url)
        time.sleep(2)

        _, path, summary = _screenshot_and_ask(
            "Summarize the search results shown on this page. "
            "List the top results with their titles and brief descriptions."
        )
        return f"Search results for '{query}':\n{summary}\n(Screenshot: {path})"
    except Exception as e:
        return f"ERROR search_web: {e}"


class CloseBrowserArgs(BaseModel):
    pass


@register_tool(
    "close_browser",
    "Close the browser and release all resources.",
    CloseBrowserArgs,
)
def close_browser() -> str:
    try:
        get_browser_manager().stop()
        return "Browser closed."
    except Exception as e:
        return f"ERROR close_browser: {e}"
