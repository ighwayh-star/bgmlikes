# -*- coding: utf-8 -*-
"""离线 UI 验证：用 Playwright 拦截所有 API，点击验证两页卡片/浮窗/打分/隐藏。
不碰真实 BGM，不加载语料库。运行：python _verify_ui.py
"""
import asyncio
import base64
import datetime
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from playwright.async_api import async_playwright

WEB_DIR = r"D:\PROJECTS\bgmlikes\web"
BASE = "http://127.0.0.1:8765"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

td = datetime.date.today().isoweekday()  # 1=周一..7=周日 == BGM weekday.id
yd = td - 1 if td > 1 else 7
WD = {1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六", 7: "周日"}


def anime(sid, cn, ja, score):
    return {
        "id": sid, "name": ja, "name_cn": cn,
        "images": {"grid": f"https://lain.bgm.tv/grid/{sid}.jpg", "small": f"https://lain.bgm.tv/small/{sid}.jpg"},
        "rating": {"score": score, "total": 10},
        "url": f"https://bgm.tv/subject/{sid}",
    }


A1001 = anime(1001, "日常一", "Anime One", 7.5)
A1002 = anime(1002, "日常二", "Anime Two", 8.0)
A1003 = anime(1003, "昨日番", "Yesterday Show", 6.8)
A1004 = anime(1004, "季番三", "Anime Three", 9.1)

calendar = [
    {"weekday": {"id": yd, "cn": WD[yd]}, "items": [A1003]},
    {"weekday": {"id": td, "cn": WD[td]}, "items": [A1001, A1002]},
    {"weekday": {"id": (td % 7) + 1, "cn": "X"}, "items": [A1004]},
]
subjects = {}
for a in (A1001, A1002, A1003, A1004):
    subjects[a["id"]] = {
        "subject_id": a["id"], "name": a["name"], "name_cn": a["name_cn"],
        "summary": f"{a['name_cn']} 的剧情简介。这是一段用于验证简介浮窗的纯文本。",
        "rating": a["rating"]["score"], "date": "2026-04-01",
        "tags": ["日常", "搞笑"],
        "similar": [
            {"subject_id": 1002, "name": "Anime Two", "rating": 8.0},
            {"subject_id": 1004, "name": "Anime Three", "rating": 9.1},
        ],
    }
collections = {"items": [{"subject_id": 1001, "state": "看过", "rate": 8},
                         {"subject_id": 1003, "state": "在看", "rate": 0}],
               "profile_nsfw": False}
recs = [{"subject_id": 1001, "name": "日常一", "rating": 7.5, "popularity_rank": 100}]
# 无限滚动测试用长池（40 部，模拟推荐池）
recs_long = [{"subject_id": 2000 + i, "name": f"滚动动画 {i}", "rating": 7.5, "popularity_rank": 100}
             for i in range(40)]
RECS_LIST = recs  # /v1/recommend 返回的列表（滚动测试时换成 recs_long）
# 打分页热门列表（1001 在 collections 里 rate 8）
popular = [
    {"subject_id": 1001, "name": "Anime One", "name_cn": "日常一", "score": 7.5, "date": "2026-04-01"},
    {"subject_id": 1004, "name": "Anime Three", "name_cn": "季番三", "score": 9.1, "date": "2026-04-05"},
]

patched = []   # (sid, rate)
deleted = []   # [sid]


class Logged:
    state = True


async def handler(route):
    req = route.request
    method = req.method
    p = req.url.split("8765", 1)[-1].split("?")[0]
    try:
        if p.startswith("/img/"):
            return await route.fulfill(status=200, content_type="image/png", body=PNG)
        if p == "/api/calendar":
            return await route.fulfill(json=calendar)
        if p == "/api/rate/collections":
            return await route.fulfill(json=collections)
        if p == "/api/rate/popular":
            return await route.fulfill(json={"data": popular})
        if p == "/api/rate/search":
            return await route.fulfill(json={"data": []})
        if p.startswith("/api/subject/"):
            sid = int(p.rsplit("/", 1)[-1])
            return await route.fulfill(json=subjects.get(sid, {}))
        m = re.fullmatch(r"/api/rate/(\d+)", p)
        if m:
            sid = int(m.group(1))
            if method == "PATCH":
                body = json.loads(req.post_data or "{}")
                patched.append((sid, body.get("rate")))
                return await route.fulfill(status=200, json={})
            if method == "DELETE":
                deleted.append(sid)
                return await route.fulfill(status=200, json={})
            return await route.fulfill(status=405, json={"detail": "method"})
        if p == "/auth/me":
            return await route.fulfill(json={"logged_in": Logged.state, "username": "test"})
        if p == "/auth/logout":
            return await route.fulfill(status=200, json={})
        if p == "/preferences/hidden":
            return await route.fulfill(json={"hidden": []} if method == "GET" else {})
        if p.startswith("/preferences/hidden/"):
            return await route.fulfill(status=200, json={})
        if p == "/daily/hidden":
            return await route.fulfill(json={"hidden": []})
        if p.startswith("/daily/hidden/"):
            return await route.fulfill(status=200, json={})
        if p == "/daily/collected":
            return await route.fulfill(json={"items": [{"id": 1001, "state": "看过"}]})
        if p == "/v1/recommend":
            return await route.fulfill(json={"username": "test", "normal": RECS_LIST, "source": "cache"})
        rel = p.lstrip("/")
        f = os.path.join(WEB_DIR, rel)
        if os.path.isfile(f):
            return await route.fulfill(
                status=200,
                content_type="text/html" if rel.endswith(".html") else "text/plain",
                body=open(f, "rb").read(),
            )
        return await route.fulfill(status=404, body="nf")
    except Exception as e:  # noqa: BLE001
        print("ROUTE ERR", p, e)
        return await route.fulfill(status=500, body=str(e))


results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS" if cond else "FAIL"), name, detail)


async def run_index(browser, logged_in):
    Logged.state = logged_in
    patched.clear(); deleted.clear()
    ctx = await browser.new_context(viewport={"width": 1000, "height": 800})
    await ctx.route(re.compile(r".*"), handler)
    page = await ctx.new_page()
    # 覆盖 window.alert：alert 是同步阻塞对话框，Playwright 驱动下可能卡死；改成记录消息
    await page.add_init_script("window.__alerts=[];window.alert=function(m){window.__alerts.push(String(m));};")
    await page.goto(BASE + "/index.html")
    if not logged_in:
        # 未登录不会自动推荐：手动输入用户名并点推荐
        await page.fill("#q", "testuser")
        await page.click("#go")
    await page.wait_for_selector(".card", timeout=10000)
    card = await page.query_selector(".card")
    check("index card has BGM link", await card.query_selector(".link-bgm") is not None)
    check("index card has Bili link", await card.query_selector(".link-bili") is not None)
    if logged_in:
        check("index card has hide btn (self)", await card.query_selector(".hide") is not None)
    else:
        check("index logged-out card has NO hide btn", await card.query_selector(".hide") is None)

    await card.click()
    await page.wait_for_selector(".modal .modal-title", timeout=10000)
    title = await page.text_content(".modal .modal-title")
    check("index modal opens with title", title and "Anime One" in title)
    summary = await page.text_content(".modal .modal-summary")
    check("index modal summary shown", summary and "剧情简介" in summary)
    # 浮窗 BGM/B站 跳转按钮
    mbgm = await page.query_selector(".modal .modal-link-bgm")
    mbili = await page.query_selector(".modal .modal-link-bili")
    check("index modal has BGM jump btn", mbgm is not None)
    check("index modal has Bili jump btn", mbili is not None)
    if mbgm:
        href = await mbgm.get_attribute("href")
        check("index modal BGM href is subject url",
              re.match(r"^https://bgm\.tv/subject/\d+$", href or "") is not None, href or "")
    if mbili:
        href = await mbili.get_attribute("href")
        check("index modal Bili href has keyword", href and "keyword=" in href, href or "")

    if logged_in:
        stars = await page.query_selector_all(".modal .stars .star")
        check("index modal shows 10 stars", len(stars) == 10)
        await page.wait_for_selector('.modal .stars .star.on[data-v="8"]', timeout=3000)
        badge = await page.text_content(".modal .rated-badge")
        check("index stars prefilled from collections (rate 8)", badge and "看过 8分" in badge)
        # 点第 5 颗星
        await page.click('.modal .star[data-v="5"]')
        await page.wait_for_timeout(300)
        check("index PATCH rate=5 sent", (1001, 5) in patched)
        badge = await page.text_content(".modal .rated-badge")
        check("index badge updates to 5", badge and "看过 5分" in badge)
        check("index clear button appears", await page.query_selector(".modal .clear-btn:not(.hidden)") is not None)
        # 再点第 5 颗星 → 清除
        await page.click('.modal .star[data-v="5"]')
        await page.wait_for_timeout(400)
        check("index DELETE sent", 1001 in deleted)
        # 移开鼠标结束悬浮预览，看真实 rateMap 值（悬浮预览会临时点亮 1..k，属预期）
        await page.mouse.move(10, 10)
        await page.wait_for_timeout(200)
        cls = await page.evaluate(
            "[...document.querySelectorAll('.modal .stars .star')].map(s => s.className + '@' + s.dataset.v)"
        )
        check("index stars cleared after clear", all("on" not in c for c in cls), str(cls))
    else:
        check("index modal shows login hint", await page.query_selector(".modal .modal-rate-hint a") is not None)
        check("index modal no stars when logged out",
              await page.query_selector(".modal .stars") is None)

    # 相似项跳转
    await page.click(".modal .similar-item")
    await page.wait_for_timeout(300)
    title = await page.text_content(".modal .modal-title")
    check("index similar navigates", title and "Anime Two" in title)

    # 关闭三途径：✕
    await page.click(".modal-close")
    await page.wait_for_timeout(150)
    vis = await page.evaluate("getComputedStyle(document.getElementById('modal')).display")
    check("index modal closes via ✕ (double-class)", vis == "none")

    # 重开 → 点遮罩关闭
    await page.click(".card")
    await page.wait_for_selector(".modal .modal-title")
    await page.mouse.click(20, 20)
    await page.wait_for_timeout(150)
    vis = await page.evaluate("getComputedStyle(document.getElementById('modal')).display")
    check("index modal closes via overlay", vis == "none")

    # 重开 → Esc 关闭
    await page.click(".card")
    await page.wait_for_selector(".modal .modal-title")
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(150)
    vis = await page.evaluate("getComputedStyle(document.getElementById('modal')).display")
    check("index modal closes via Esc", vis == "none")

    await ctx.close()


async def run_index_scroll(browser):
    """推荐页新 UI：三列网格 + 无限滚动渐进加载（池上限 50）+ 标题左上角。"""
    global RECS_LIST
    RECS_LIST = recs_long
    ctx = await browser.new_context(viewport={"width": 1100, "height": 800})
    await ctx.route(re.compile(r".*"), handler)
    page = await ctx.new_page()
    await page.add_init_script("window.__alerts=[];window.alert=function(m){window.__alerts.push(String(m));};")
    await page.goto(BASE + "/index.html")
    await page.fill("#q", "testuser")
    await page.click("#go")
    await page.wait_for_selector(".card", timeout=10000)

    grid = await page.evaluate("getComputedStyle(document.getElementById('list')).gridTemplateColumns")
    check("index list is 3-column grid", len(grid.split()) == 3, grid)
    n0 = len(await page.query_selector_all(".card"))
    check("index initial batch = 15 cards", n0 == 15, str(n0))

    # 逐次滚到底触发哨兵 → 池内渐进加载，直到全部（40）展示
    for _ in range(8):
        await page.evaluate(
            "document.getElementById('list-sentinel').scrollIntoView({behavior:'instant',block:'end'})")
        await page.wait_for_timeout(250)
    n = len(await page.query_selector_all(".card"))
    check("index scroll loads full pool (40)", n == 40, str(n))

    # 排序编号已移除：卡片上无 rank 徽标
    rank = await page.query_selector(".card .rank")
    check("index no rank badge", rank is None)

    # 刷新（换一批）按钮：存在、文案"刷新"、点击后随机重排池并回到初始 15 条
    rb = await page.query_selector("#refresh")
    check("index refresh button present", rb is not None)
    rbtxt = await page.text_content("#refresh")
    check("index refresh button text=刷新", rbtxt is not None and "刷新" in rbtxt, rbtxt or "")
    await page.click("#refresh")
    await page.wait_for_timeout(250)
    n2 = len(await page.query_selector_all(".card"))
    check("index refresh resets to 15 cards", n2 == 15, str(n2))
    # 刷新后继续滚动：仍从池中渐进加载到全部（重排后池内无重复）
    for _ in range(8):
        await page.evaluate(
            "document.getElementById('list-sentinel').scrollIntoView({behavior:'instant',block:'end'})")
        await page.wait_for_timeout(250)
    n3 = len(await page.query_selector_all(".card"))
    sids = await page.evaluate("[...document.querySelectorAll('.card')].map(c => c.dataset.sid).join(',')")
    check("index refresh keeps scroll loading to pool (40)", n3 == 40, str(n3))
    check("index refresh pool no dup sids", len(set(sids.split(','))) == 40, sids)

    # 标题在页面左上角：topbar 通栏，左缘贴近视口
    box = await page.evaluate("document.querySelector('.topbar').getBoundingClientRect().left")
    check("index topbar title at top-left", box < 8, str(box))
    await ctx.close()
    RECS_LIST = recs


async def run_daily(browser, logged_in):
    Logged.state = logged_in
    patched.clear(); deleted.clear()
    ctx = await browser.new_context(viewport={"width": 1000, "height": 800})
    await ctx.route(re.compile(r".*"), handler)
    page = await ctx.new_page()
    await page.add_init_script("window.__alerts=[];window.alert=function(m){window.__alerts.push(String(m));};")
    await page.goto(BASE + "/daily.html")
    await page.wait_for_selector(".card", timeout=10000)
    card = await page.query_selector(".card")
    check("daily card has BGM link", await card.query_selector(".link-bgm") is not None)
    check("daily card has Bili link", await card.query_selector(".link-bili") is not None)
    check("daily card has cover", await card.query_selector(".cover") is not None)
    check("daily card has hide btn", await card.query_selector(".hide") is not None)

    # 点卡片 → 浮窗（不是新标签）
    await card.click()
    await page.wait_for_selector(".modal .modal-title", timeout=10000)
    title = await page.text_content(".modal .modal-title")
    check("daily card opens modal (not new tab)", title is not None)
    mbgm = await page.query_selector(".modal .modal-link-bgm")
    mbili = await page.query_selector(".modal .modal-link-bili")
    check("daily modal has BGM jump btn", mbgm is not None)
    check("daily modal has Bili jump btn", mbili is not None)
    if mbgm:
        href = await mbgm.get_attribute("href")
        check("daily modal BGM href is subject url",
              re.match(r"^https://bgm\.tv/subject/\d+$", href or "") is not None, href or "")
    if mbili:
        href = await mbili.get_attribute("href")
        check("daily modal Bili href has keyword", href and "keyword=" in href, href or "")
    if logged_in:
        await page.wait_for_selector(".modal .stars", timeout=3000)
        check("daily modal stars row present", True)
    else:
        check("daily modal login hint", await page.query_selector(".modal .modal-rate-hint a") is not None)
    await page.click(".modal-close")
    await page.wait_for_timeout(150)

    # 隐藏 → 灰化归入已隐藏 → 再点恢复
    if logged_in:
        hide_btn = card_query = None
        # 找一张未隐藏卡的 ✕（普通卡片区第一张）
        cards = await page.query_selector_all('.card:not(.hidden-card)')
        target = cards[0] if cards else card
        hb = await target.query_selector(".hide")
        await hb.click()
        await page.wait_for_timeout(250)
        hid_card = await page.query_selector(".card.hidden-card")
        check("daily hide grays card + moves to hidden", hid_card is not None)
        hid_sec = await page.query_selector(".hidden-sec")
        check("daily hidden section appears", hid_sec is not None)
        # 已隐藏卡片上的 ✕ = 恢复
        hb2 = await hid_card.query_selector(".hide")
        title_attr = await hb2.get_attribute("title")
        check("daily hidden hide btn title=恢复显示", title_attr == "恢复显示")
        await hb2.click()
        await page.wait_for_timeout(250)
        check("daily unhide restores card",
              await page.query_selector(".card.hidden-card") is None)
    else:
        # 未登录点 ✕ → alert（window.alert 已被覆盖记录，不会卡死）
        await page.eval_on_selector(".card .hide", "el => el.click()")
        await page.wait_for_timeout(400)
        al = await page.evaluate("window.__alerts")
        check("daily logged-out hide shows alert", any("登录" in m for m in al), str(al))

    # tab 切换 + 选择模式
    await page.click("#tabSeason")
    await page.wait_for_timeout(200)
    check("daily season tab renders cards", await page.query_selector("#content .card") is not None)
    await page.click('[data-sel="enter"]')
    await page.wait_for_timeout(200)
    c = await page.query_selector("#content .card:not(.hidden-card)")
    await c.click()
    await page.wait_for_timeout(200)
    check("daily select-mode toggles checked",
          await page.query_selector(".card.card--checked") is not None)
    # 选择模式下点卡不弹浮窗
    check("daily select-mode does not open modal",
          await page.evaluate("document.getElementById('modal').classList.contains('hidden')"))
    await page.click('[data-sel="exit"]')
    await page.wait_for_timeout(200)

    # 未登录浮窗显示登录链接（本页另一个 tab 场景已在上面覆盖）
    await ctx.close()


async def run_rate(browser):
    """打分页：卡片 BGM/B站 外链 + 点击卡片 → 简介浮窗（含跳转按钮 + 星形预填）。"""
    Logged.state = True
    patched.clear(); deleted.clear()
    ctx = await browser.new_context(viewport={"width": 900, "height": 800})
    await ctx.route(re.compile(r".*"), handler)
    page = await ctx.new_page()
    await page.add_init_script("window.__alerts=[];window.alert=function(m){window.__alerts.push(String(m));};")
    await page.goto(BASE + "/rate.html")
    await page.wait_for_selector(".card", timeout=10000)

    card = await page.query_selector(".card")
    bgm = await card.query_selector(".link-bgm")
    bili = await card.query_selector(".link-bili")
    check("rate card has BGM link", bgm is not None)
    check("rate card has Bili link", bili is not None)
    if bgm:
        href = await bgm.get_attribute("href")
        check("rate card BGM href is subject url",
              re.match(r"^https://bgm\.tv/subject/\d+$", href or "") is not None, href or "")
    if bili:
        href = await bili.get_attribute("href")
        check("rate card Bili href has keyword", href and "keyword=" in href, href or "")

    # 点击卡片（非星/外链）→ 简介浮窗
    await page.click(".card .name-cn")
    await page.wait_for_selector(".modal .modal-title", timeout=10000)
    title = await page.text_content(".modal .modal-title")
    check("rate card opens modal with title", title and "Anime One" in title, title or "")
    mbgm = await page.query_selector(".modal .modal-link-bgm")
    mbili = await page.query_selector(".modal .modal-link-bili")
    check("rate modal has BGM jump btn", mbgm is not None)
    check("rate modal has Bili jump btn", mbili is not None)
    if mbgm:
        href = await mbgm.get_attribute("href")
        check("rate modal BGM href is subject url",
              re.match(r"^https://bgm\.tv/subject/\d+$", href or "") is not None, href or "")
    if mbili:
        href = await mbili.get_attribute("href")
        check("rate modal Bili href has keyword", href and "keyword=" in href, href or "")
    # 浮窗星形从 rateMap 预填（1001 在 collections rate 8）
    await page.wait_for_selector(".modal .stars .star.on[data-v='8']", timeout=3000)
    badge = await page.text_content(".modal .rated-badge")
    check("rate modal stars prefilled (rate 8)", badge and "看过 8分" in badge, badge or "")

    # 关闭浮窗
    await page.click(".modal-close")
    await page.wait_for_timeout(150)
    vis = await page.evaluate("getComputedStyle(document.getElementById('modal')).display")
    check("rate modal closes via ✕", vis == "none")

    # 点星打分不打浮窗（打分仍走卡片行内）
    await page.click('.card .star[data-v="5"]')
    await page.wait_for_timeout(300)
    check("rate star click rates (no modal)", (1001, 5) in patched)
    vis = await page.evaluate("getComputedStyle(document.getElementById('modal')).display")
    check("rate star click does not open modal", vis == "none")

    await ctx.close()


async def main():
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True, channel="msedge")
        except Exception:
            browser = await p.chromium.launch(headless=True, channel="chrome")
        try:
            await run_index(browser, True)
            await run_index(browser, False)
            await run_index_scroll(browser)
            await run_daily(browser, True)
            await run_daily(browser, False)
            await run_rate(browser)
        finally:
            await browser.close()
    ok = all(results)
    print("\n=== %d/%d checks passed ===" % (sum(1 for r in results if r), len(results)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
