"""Regression tests for public documentation MathJax LaTeX rendering."""

from __future__ import annotations

import functools
import http.server
import shutil
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS_SITE = ROOT / "docs_site"
DOCS = DOCS_SITE / "docs"
JAVASCRIPTS = DOCS / "javascripts"
MATHJAX_DIR = JAVASCRIPTS / "mathjax"
SITE_DIR = ROOT / "site"


# -----------------------------------------------------------------------------
# Static Asset and Configuration Contract Tests
# -----------------------------------------------------------------------------


def test_mathjax_vendored_assets_present() -> None:
    """Verify that all required vendored MathJax assets exist locally."""
    assert (JAVASCRIPTS / "mathjax-config.js").is_file()
    assert (MATHJAX_DIR / "tex-mml-chtml.js").is_file()
    assert (MATHJAX_DIR / "LICENSE.txt").is_file()

    # boldsymbol extension is loaded dynamically by MathJax; must be present locally
    boldsymbol_ext = MATHJAX_DIR / "input" / "tex" / "extensions" / "boldsymbol.js"
    assert boldsymbol_ext.is_file(), "Vendored boldsymbol.js extension missing"
    assert boldsymbol_ext.stat().st_size > 1000, "boldsymbol.js appears truncated"

    # Web fonts must be present under output/chtml/fonts/woff-v2/
    font_dir = MATHJAX_DIR / "output" / "chtml" / "fonts" / "woff-v2"
    assert font_dir.is_dir(), "MathJax woff-v2 font directory missing"
    required_fonts = [
        "MathJax_Zero.woff",
        "MathJax_Main-Regular.woff",
        "MathJax_Main-Bold.woff",
        "MathJax_Math-Italic.woff",
    ]
    for font_name in required_fonts:
        font_path = font_dir / font_name
        assert font_path.is_file(), f"Font {font_name} missing from vendored assets"
        assert font_path.stat().st_size > 0, f"Font {font_name} is empty"


def test_mathjax_config_contract() -> None:
    """Verify mathjax-config.js contains the contract required for gragra docs."""
    content = (JAVASCRIPTS / "mathjax-config.js").read_text(encoding="utf-8")

    # Contract 1: window.MathJax must declare [tex]/boldsymbol loader and package
    assert "[tex]/boldsymbol" in content, "loader.load must specify [tex]/boldsymbol"
    assert "boldsymbol" in content, "tex.packages must include boldsymbol"

    # Contract 2: inline and display delimiters matching arithmatex generic output
    assert r'["\\(", "\\)"]' in content or r'[["\\(", "\\)"]]' in content
    assert r'["\\[", "\\]"]' in content or r'[["\\[", "\\]"]]' in content

    # Contract 3: processHtmlClass must target arithmatex
    assert '"arithmatex"' in content or "'arithmatex'" in content

    # Contract 4: Safe typeset handling preventing duplicate rendering
    # Must only typeset unrendered elements (those without mjx-container)
    assert "!el.querySelector" in content or "mjx-container" in content, (
        "mathjax-config.js must guard against re-typesetting existing mjx-container"
    )
    assert "document$.subscribe" in content, "Must subscribe to Material's document$"


def test_mkdocs_mathjax_configuration() -> None:
    """Verify mkdocs.yml configures arithmatex and JavaScript assets correctly."""
    mkdocs_content = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")

    # Arithmatex extension in generic mode
    assert "pymdownx.arithmatex" in mkdocs_content
    assert "generic: true" in mkdocs_content

    # Script tags order: mathjax-config.js BEFORE tex-mml-chtml.js
    cfg_pos = mkdocs_content.find("javascripts/mathjax-config.js")
    bundle_pos = mkdocs_content.find("javascripts/mathjax/tex-mml-chtml.js")
    assert cfg_pos != -1, "mathjax-config.js missing from extra_javascript"
    assert bundle_pos != -1, "tex-mml-chtml.js missing from extra_javascript"
    assert cfg_pos < bundle_pos, "mathjax-config.js must load before tex-mml-chtml.js"


# -----------------------------------------------------------------------------
# Browser Automation Tests (Run when playwright and chrome are available)
# -----------------------------------------------------------------------------


def _find_chrome_executable() -> str | None:
    candidates = (
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return shutil.which("google-chrome") or shutil.which("chromium")


@pytest.fixture(scope="module")
def local_site_server():
    """Serve the built site directory on an ephemeral localhost port."""
    if not SITE_DIR.is_dir() or not (SITE_DIR / "index.html").is_file():
        pytest.skip("Site not built; run `mkdocs build` first")

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(SITE_DIR),
    )
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def test_mathjax_browser_renders_inline_and_display(local_site_server: str) -> None:
    """Verify inline and display math rendering in browser without raw delimiters."""
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = _find_chrome_executable()
    if not chrome:
        pytest.skip("Google Chrome or Chromium binary not found")

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True)
        page = browser.new_page()

        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        url = f"{local_site_server}/models/stochastic_atmospheric_infrasound/"
        page.goto(url)
        page.wait_for_timeout(500)

        data = page.evaluate(
            """() => {
            const ariths = Array.from(document.querySelectorAll('.arithmatex'));
            const unrendered = ariths.filter(el => !el.querySelector('mjx-container'));
            const textContent = document.body.innerText;
            const hasRawInline = /\\\\\\([a-zA-Z0-9]/.test(textContent);
            const hasRawDisplay = /\\\\\\[[a-zA-Z0-9]/.test(textContent);
            return {
                totalArith: ariths.length,
                totalMjx: document.querySelectorAll('mjx-container').length,
                unrenderedCount: unrendered.length,
                hasRawInline,
                hasRawDisplay
            };
        }"""
        )

        assert not errors, f"JavaScript errors occurred: {errors}"
        assert data["totalArith"] > 0, "No arithmatex elements found"
        assert data["unrenderedCount"] == 0, (
            f"{data['unrenderedCount']} math elements unrendered"
        )
        assert data["totalMjx"] == data["totalArith"], (
            "Mismatch between arithmatex and mjx-container counts"
        )
        assert not data["hasRawInline"], "Raw inline delimiter \\( found in page text"
        assert not data["hasRawDisplay"], "Raw display delimiter \\[ found in page text"

        browser.close()


def test_mathjax_browser_navigation_no_duplicates(local_site_server: str) -> None:
    """Verify navigation re-renders math without duplication or nesting."""
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = _find_chrome_executable()
    if not chrome:
        pytest.skip("Google Chrome or Chromium binary not found")

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True)
        page = browser.new_page()

        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))

        # Initial load
        page.goto(f"{local_site_server}/models/stochastic_atmospheric_infrasound/")
        page.wait_for_timeout(500)

        # Navigate to isotropic_stochastic_rayleigh_field
        page.goto(f"{local_site_server}/models/isotropic_stochastic_rayleigh_field/")
        page.wait_for_timeout(500)

        data = page.evaluate(
            """() => {
            const ariths = Array.from(document.querySelectorAll('.arithmatex'));
            const unrendered = ariths.filter(el => !el.querySelector('mjx-container'));
            const isDup = el => el.querySelectorAll('mjx-container').length > 1;
            const dup = ariths.filter(isDup);
            return {
                totalArith: ariths.length,
                totalMjx: document.querySelectorAll('mjx-container').length,
                unrenderedCount: unrendered.length,
                duplicatedCount: dup.length
            };
        }"""
        )

        assert not errors, f"JavaScript errors on navigation: {errors}"
        assert data["totalArith"] > 0
        assert data["unrenderedCount"] == 0
        assert data["duplicatedCount"] == 0, (
            "Nested/duplicate mjx-containers detected after navigation"
        )
        assert data["totalMjx"] == data["totalArith"]

        browser.close()


def test_mathjax_key_pages_render_without_errors(local_site_server: str) -> None:
    """Verify that key reference and model pages render 100% of math expressions."""
    playwright = pytest.importorskip("playwright.sync_api")
    chrome = _find_chrome_executable()
    if not chrome:
        pytest.skip("Google Chrome or Chromium binary not found")

    key_pages = [
        "/guide/physics-reference/",
        "/models/isotropic_stochastic_rayleigh_field/",
        "/models/stochastic_atmospheric_infrasound/",
    ]

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome, headless=True)
        page = browser.new_page()

        for path in key_pages:
            errors: list[str] = []

            def _on_error(err: object, err_list: list[str] = errors) -> None:
                err_list.append(str(err))

            page.on("pageerror", _on_error)

            page.goto(f"{local_site_server}{path}")
            page.wait_for_timeout(500)

            data = page.evaluate(
                """() => {
                const ariths = Array.from(document.querySelectorAll('.arithmatex'));
                const notRendered = el => !el.querySelector('mjx-container');
                const unrendered = ariths.filter(notRendered);
                return {
                    totalArith: ariths.length,
                    unrenderedCount: unrendered.length
                };
            }"""
            )

            assert not errors, f"Errors on {path}: {errors}"
            assert data["totalArith"] > 0, f"Expected math on {path}"
            assert data["unrenderedCount"] == 0, (
                f"{data['unrenderedCount']} unrendered math on {path}"
            )

        browser.close()
