"""締切前の3連単オッズを公式サイトから取得する（GitHub上の自動実行で使用）。

公式ページ: https://www.boatrace.jp/owpc/pc/race/odds3t?rno=R&jcd=場コード2桁&hd=YYYYMMDD
ページ内の「td class=oddsPoint」が120個、次の順番で並んでいます。
  20行 × 6列。列 = 1着の艇（1〜6）、行 = 2着・3着の組み合わせ（若い番号順、2着ごとに4行）
マナー: 1回ごとに3秒以上あけ、必要なレース（締切が近いもの）だけ取りに行きます。
"""
from __future__ import annotations

import re
import time
import urllib.request

ODDS_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t?rno={r}&jcd={j:02d}&hd={ymd}"
_CELL = re.compile(r'<td[^>]*class="[^"]*oddsPoint[^"]*"[^>]*>\s*([^<]*?)\s*</td>', re.S)


def parse_odds3t(html: str) -> dict[str, float] | None:
    vals = _CELL.findall(html)
    if len(vals) != 120:
        return None
    out = {}
    for row in range(20):
        for first in range(1, 7):
            rest = [b for b in range(1, 7) if b != first]
            second = rest[row // 4]
            third = [b for b in rest if b != second][row % 4]
            v = vals[row * 6 + (first - 1)].replace(",", "")
            try:
                out[f"{first}-{second}-{third}"] = float(v)
            except ValueError:  # 欠場などで数字がない
                pass
    return out or None


def fetch_odds3t(j: int, r: int, ymd: str, wait: float = 3.0) -> dict[str, float] | None:
    url = ODDS_URL.format(r=r, j=j, ymd=ymd)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (kyotei-yoso-site; personal use)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as res:
            html = res.read().decode("utf-8", "ignore")
    except Exception as e:
        print(f"  オッズ取得失敗 {j}場{r}R: {e}")
        return None
    finally:
        time.sleep(wait)
    return parse_odds3t(html)
