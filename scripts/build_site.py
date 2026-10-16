"""画面のひな形 site/template.html から、公開用のページを作る。

  site/index.html     … GitHub Pages 用（data/*.json を読み込む）
  dist/standalone.html … データを全部埋め込んだ1ファイル版（ネットなしでも開ける）
"""
from __future__ import annotations

import json
import sys

from common import ROOT, SITE_DATA

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
    out = ROOT / "dist" / "standalone.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(tpl.replace("<!--DATA-->", "\n".join(blobs)), encoding="utf-8")
    print(f"作成: site/index.html, dist/standalone.html（{out.stat().st_size/1e6:.1f}MB）")


if __name__ == "__main__":
    main("--standalone-only" in sys.argv)
