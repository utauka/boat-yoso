"""画面のひな形 site/template.html から、公開用のページを作る。

  site/index.html     … GitHub Pages 用（data/*.json を読み込む）
  dist/standalone.html … データを全部埋め込んだ1ファイル版（ネットなしでも開ける）
"""
from __future__ import annotations

import json
import sys

from common import ROOT, SITE_DATA

LIVE_URL = "https://utauka.github.io/boat-yoso/"
NAMES = ["stats", "racers", "predictions", "backtest", "pred_log"]
HEAD = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="robots" content="noindex">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}[hidden]{display:none!important}img{max-width:100%}</style>
"""


def main(standalone_only=False):
    tpl = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    title_end = tpl.index("</style>") + len("</style>")
    head_part, body_part = tpl[:title_end], tpl[title_end:]
    if not standalone_only:
        page = HEAD + head_part + "\n</head>\n<body>\n" + body_part.replace("<!--DATA-->", "") + "\n</body>\n</html>\n"
        (ROOT / "site" / "index.html").write_text(page, encoding="utf-8")
    blobs = []
    for n in NAMES:
        p = SITE_DATA / f"{n}.json"
        if not p.exists():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if n == "predictions":
            for z in obj.get("races", []):
                z.pop("_live", None)
        txt = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
        blobs.append(f'<script type="application/json" id="d-{n}">{txt}</script>')
    # 1ファイル版（チャット内の保存版）だけに、自動更新される本番サイトへの案内を付ける
    gen = ""
    pp = SITE_DATA / "predictions.json"
    if pp.exists():
        g = json.loads(pp.read_text(encoding="utf-8")).get("generated", "")
        if g:
            gen = f"{int(g[5:7])}月{int(g[8:10])}日 {g[11:16]}"
    banner = f"""<div class="live-banner" role="note"><div class="wrap">
  <span>この画面は保存版です（データは{gen}時点で止まっています）。毎日自動で更新される最新版はこちら</span>
  <a href="{LIVE_URL}" target="_blank" rel="noopener">最新版を開く</a>
</div></div>
<style>
.live-banner{{background:color-mix(in srgb,var(--accent) 14%,var(--paper));border-bottom:1px solid var(--line)}}
.live-banner .wrap{{display:flex;flex-wrap:wrap;align-items:center;gap:8px 14px;padding-block:10px;font-size:13.5px}}
.live-banner span{{flex:1 1 240px;min-width:0}}
.live-banner a{{background:var(--accent);color:var(--accent-ink);font-weight:700;text-decoration:none;padding:7px 14px;border-radius:999px;white-space:nowrap}}
.live-banner a:focus-visible{{outline:2px solid var(--ink);outline-offset:2px}}
</style>
"""
    out = ROOT / "dist" / "standalone.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tpl.replace("<!--DATA-->", "\n".join(blobs)).replace('<main class="wrap">', banner + '<main class="wrap">', 1),
                   encoding="utf-8")
    print(f"作成: site/index.html, dist/standalone.html（{out.stat().st_size/1e6:.1f}MB）")


if __name__ == "__main__":
    main("--standalone-only" in sys.argv)
