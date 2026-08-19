#!/usr/bin/env python3
"""生产站功能截图：用于 bgmlikes 功能文章插图。

对 https://bgmhiway.asia 截屏，输出到 <repo>/pics/article_*.png。
- 推荐页 /likes 用站长账号 hiway 拉真实推荐（走公开端点，无需登录）
- 简介浮窗 / 每日放送：真实数据
- 浮窗打分 / 打分页：登录态功能示意（页内注入登录态 + 演示数据，不写任何真实账号）
"""
import asyncio
import json
import pathlib
import sys

from playwright.async_api import async_playwright

BASE = "https://bgmhiway.asia"
USER = "hiway"
OUT = pathlib.Path(__file__).resolve().parent.parent / "pics"
VIEWPORT = {"width": 1280, "height": 900}

# 打分页演示数据（真实 subject_id，封面走生产 /img/ 代理真实加载）
RATE_POPULAR_MOCK = {
    "data": [
        {"subject_id": 535669, "name": "冰之城墙", "name_cn": "冰之城墙", "score": 8.3, "date": "2026-07"},
        {"subject_id": 51, "name": "CLANNAD", "name_cn": "CLANNAD", "score": 8.9, "date": "2007-10"},
        {"subject_id": 551918, "name": "飙马野郎 JOJO的奇妙冒险", "name_cn": "飙马野郎 JOJO的奇妙冒险", "score": 8.7, "date": "2026-04"},
        {"subject_id": 509986, "name": "末日后酒店", "name_cn": "末日后酒店", "score": 8.4, "date": "2025-04"},
        {"subject_id": 393037, "name": "義妹生活", "name_cn": "义妹生活", "score": 7.6, "date": "2024-07"},
        {"subject_id": 513345, "name": "薫香花的凛然绽放", "name_cn": "薰香花朵凛然绽放", "score": 8.0, "date": "2026-04"},
        {"subject_id": 524707, "name": "我们不可能成为恋人！绝对不行", "name_cn": "我们不可能成为恋人！绝对不行", "score": 7.9, "date": "2026-04"},
        {"subject_id": 493016, "name": "异国日记", "name_cn": "异国日记", "score": 8.5, "date": "2025-07"},
    ],
    "profile_nsfw": False,
}

# 暂停所有 CSS 动画（花瓣定格，避免截到半透明飞动残影）
PAUSE_ANIM = """() => {
  const s = document.createElement('style');
  s.textContent = '*{animation-play-state:paused!important}';
  document.head.appendChild(s);
}"""

# index /likes：推荐页等推荐渲染
def wait_cover(page):
    async def f():
        await page.wait_for_function(
            "() => { const im = document.querySelector('#list img'); return !im || im.complete; }",
            timeout=30000,
        )
        await page.wait_for_timeout(2200)  # 让首屏其余封面慢慢落
    return f()


async def shot(page, name):
    page.screenshot(path=str(OUT / name))
    print(f"saved {name}")


async def run():
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True)
        page = await browser.new_page(viewport=VIEWPORT, device_scale_factor=2)
        page.set_default_timeout(60000)

        # ---- 1. 首页 ----
        print(">>> home")
        await page.goto(f"{BASE}/", wait_until="load")
        await page.wait_for_timeout(1500)
        await page.evaluate(PAUSE_ANIM)
        await page.screenshot(path=str(OUT / "article_home.png"), full_page=True)
        print("saved article_home.png")

        # ---- 2. 推荐页顶部 ----
        print(">>> likes top")
        await page.goto(f"{BASE}/likes", wait_until="load")
        await page.fill("#q", USER)
        await page.click("#go")
        await page.wait_for_selector("#list .card", timeout=90000)
        await wait_cover(page)
        await page.evaluate(PAUSE_ANIM)
        await page.screenshot(path=str(OUT / "article_likes.png"))
        print("saved article_likes.png")

        # ---- 3. 推荐页下滑（更多卡片）----
        print(">>> likes scroll")
        await page.evaluate("() => window.scrollTo(0, 1350)")
        await page.wait_for_timeout(2500)
        await page.evaluate(PAUSE_ANIM)
        await page.screenshot(path=str(OUT / "article_likes_scroll.png"))
        print("saved article_likes_scroll.png")

        # ---- 4. 简介浮窗（点击第一张卡片）----
        print(">>> modal intro")
        await page.evaluate("() => window.scrollTo(0, 0)")
        await page.click("#list .card >> nth=0")
        # 用 .modal-title 作为 renderModal 完成信号（.modal-empty 会被"加载中…"占位命中，不可靠）
        await page.wait_for_selector("#modal:not(.hidden) .modal-title", timeout=30000)
        await page.wait_for_function(
            "() => { const im = document.querySelector('#modal img'); return !im || im.complete; }",
            timeout=30000,
        )
        await page.wait_for_timeout(2500)  # 相似动画图加载
        await page.evaluate(PAUSE_ANIM)
        await page.screenshot(path=str(OUT / "article_modal.png"))
        print("saved article_modal.png")

        # ---- 5. 浮窗 10 星打分示意（页内注入登录态 + rateMap，不写真实账号）----
        print(">>> modal rated")
        await page.evaluate("""() => {
          loggedIn = true;
          me = 'hiway';
          const sid = modalOpenSid;
          rateMap.set(sid, {state: '看过', rate: 8});
          renderModal(modalCache.get(sid));
        }""")
        await page.wait_for_selector("#modal .stars", timeout=10000)
        await page.wait_for_timeout(1200)
        await page.evaluate(PAUSE_ANIM)
        await page.screenshot(path=str(OUT / "article_modal_rated.png"))
        print("saved article_modal_rated.png")

        # ---- 6. 每日放送 ----
        print(">>> daily")
        await page.goto(f"{BASE}/daily", wait_until="load")
        await page.wait_for_selector(".day-sec, .card[data-sid]", timeout=60000)
        await page.wait_for_function(
            "() => { const im = document.querySelector('.day-sec img, .card img'); return !im || im.complete; }",
            timeout=30000,
        )
        await page.wait_for_timeout(2200)
        await page.evaluate(PAUSE_ANIM)
        await page.screenshot(path=str(OUT / "article_daily.png"))
        print("saved article_daily.png")

        # ---- 7. 打分页（登录态功能示意：拦截 popular 返回演示数据）----
        print(">>> rate")
        async def handle_popular(route):
            await route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(RATE_POPULAR_MOCK),
            )
        await page.route("**/api/rate/popular*", handle_popular)
        await page.goto(f"{BASE}/rate", wait_until="load")
        await page.wait_for_selector("#loginHint:not(.hidden), #content .empty", timeout=15000)
        await page.evaluate("""() => {
          loggedIn = true;
          rateMap = new Map();
          userLabel.textContent = 'hiway';
          userLabel.classList.remove('hidden');
          loginBtn.classList.add('hidden');
          logoutBtn.classList.remove('hidden');
          loginHint.classList.add('hidden');
          searchBox.classList.remove('hidden');
          showPopular();
        }""")
        await page.wait_for_selector("#content .card", timeout=20000)
        await page.wait_for_function(
            "() => { const im = document.querySelector('#content img'); return !im || im.complete; }",
            timeout=30000,
        )
        await page.wait_for_timeout(2200)
        await page.evaluate(PAUSE_ANIM)
        await page.screenshot(path=str(OUT / "article_rate.png"))
        print("saved article_rate.png")

        await browser.close()
        print("ALL DONE")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as e:
        print("FAILED:", repr(e))
        sys.exit(1)
