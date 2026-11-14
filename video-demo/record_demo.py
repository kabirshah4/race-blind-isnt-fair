"""Screen-record the live FairView demo (no voice) for the demo video.

Run:  ../.venv/bin/python record_demo.py
Writes assets/demo_flip.webm and assets/demo_random.webm.
A fake mouse cursor is drawn so viewers can follow the clicks.
"""
import shutil
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

import os
URL = os.environ.get("DEMO_URL", "https://fairview-ai.pages.dev/")
ZOOM = 1.7
OUT = Path(__file__).parent / "assets"
OUT.mkdir(exist_ok=True)

CURSOR = """
const c = document.createElement('div');
c.id = '__cur';
c.innerHTML = '<svg width="28" height="28" viewBox="0 0 24 24"><path d="M3 2l7 19 2.6-7.6L20 11z" fill="#0B1A33" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
Object.assign(c.style, {position:'fixed', left:'640px', top:'380px', zIndex:99999, pointerEvents:'none',
  transition:'left .7s cubic-bezier(.3,.7,.2,1), top .7s cubic-bezier(.3,.7,.2,1), transform .12s'});
document.body.appendChild(c);
"""


def move(page, selector, click=False, wait=0.9):
    box = page.locator(selector).first.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.evaluate(f"(() => {{ const c = document.getElementById('__cur'); c.style.left = '{(x - 4) / ZOOM}px'; c.style.top = '{(y - 3) / ZOOM}px'; }})()")
    time.sleep(wait)
    if click:
        page.evaluate("document.getElementById('__cur').style.transform='scale(.82)'")
        page.locator(selector).first.click()
        time.sleep(0.12)
        page.evaluate("document.getElementById('__cur').style.transform='none'")


def scroll_to(page, selector, offset=24, wait=1.4, smooth=True):
    page.evaluate(f"window.scrollTo({{top: document.querySelector('{selector}').getBoundingClientRect().top + scrollY - {offset}, behavior: '{'smooth' if smooth else 'instant'}'}})")
    time.sleep(wait)


def record(name, steps):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080},
                                  record_video_dir=str(OUT / "_raw"), record_video_size={"width": 1920, "height": 1080})
        page = ctx.new_page()
        page.goto(URL + "?rec=1")
        page.wait_for_function("document.querySelectorAll('td em').length === 6")
        page.evaluate(f"document.documentElement.style.zoom = '{ZOOM}'")   # big, readable text at 1080p
        page.evaluate(CURSOR)
        steps(page)
        video = page.video.path()
        ctx.close()
        browser.close()
    shutil.move(video, OUT / f"{name}.webm")


def flip(page):
    time.sleep(3.5)                                   # hero card cycling through neighbourhoods
    scroll_to(page, ".step", offset=8, wait=2.0)       # step 1 + both cards in view
    move(page, '#nb button[data-nb="Mostly minority"]', wait=2.5)
    move(page, '#nb button[data-nb="Mostly white"]', click=True); time.sleep(3.2)
    move(page, '#nb button[data-nb="Mixed"]', click=True); time.sleep(2.6)
    move(page, '#nb button[data-nb="Mostly minority"]', click=True); time.sleep(3.2)
    scroll_to(page, "#thead", offset=120, wait=1.0, smooth=False)   # the every-address table (jump, no smear)
    move(page, "#tbody tr:first-child th", wait=4.0)


def random_people(page):
    # Two real 2023 applicants from the 500 random ones: #13 (same answer everywhere, like most people)
    # and #370 ($100K, 24% down, bank approved; the typical AI's answer flips with the address).
    page.evaluate("(() => { const picks = [12, 369]; let i = 0; Math.random = () => (picks[Math.min(i++, picks.length - 1)] + 0.5) / 500; })()")
    scroll_to(page, ".step", offset=8, wait=1.2)
    move(page, "#random", wait=1.6)
    move(page, "#random", click=True, wait=0.3); time.sleep(3.0)
    move(page, "#random", click=True, wait=0.3); time.sleep(3.5)
    move(page, '#nb button[data-nb="Mostly white"]', click=True); time.sleep(3.0)
    move(page, "#randstat", wait=3.5)


if __name__ == "__main__":
    import sys
    if "random" not in sys.argv: record("demo_flip", flip)
    record("demo_random", random_people)
    shutil.rmtree(OUT / "_raw", ignore_errors=True)
    print("done")
