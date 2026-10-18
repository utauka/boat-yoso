"""買い目の選び方（イン逃げの2点買い・1レース200円・配分は画面で切り替え）。

  対象: 戸田・徳山・大村
  1) AIが3連単120通りそれぞれの「当たる確率」と、イン（1コースの艇）が1着になる確率＝イン逃げ率を出す
  2) 買い目はすべて「インが1着」の組番
       本命: オッズ5〜20倍、中穴: オッズ20倍より上〜60倍。それぞれ当たる確率が高い順に2点ずつ候補にする
  3) 配分（どれも100円×2点＝200円）
       本命2点 / 本命1＋中穴1 / 中穴2点   … 画面のボタンで切り替え
  4) 展示が出ていて、イン逃げ率が IN_MIN 以上なら「買い」、それ未満は見送り
"""
from __future__ import annotations

import numpy as np

from features import PERMS, combo_name

TARGET_STADIUMS = {2: "戸田", 18: "徳山", 24: "大村"}
HONMEI_MIN = 5.0
HONMEI_MAX = 20.0
CHUANA_MAX = 60.0
IN_MIN = 0.60     # イン逃げ率がこれ以上なら「買い」
UNIT = 100
MIXES = {"2-0": "本命2点", "1-1": "本命1＋中穴1", "0-2": "中穴2点"}
DEFAULT_MIX = "2-0"
SKIP_EV = 0.8
_FIRST = np.array([p[0] for p in PERMS])


def in_pick(prob: np.ndarray, odds: np.ndarray, in_boat: int, live: bool = False) -> dict:
    order = np.argsort(-prob)
    head = _FIRST == (in_boat - 1)

    def pack(i, kind):
        return {"combo": combo_name(int(i)), "kind": kind, "p": round(float(prob[i]), 4),
                "odds": round(float(odds[i]), 1), "low": bool(live and prob[i] * odds[i] < SKIP_EV)}

    hon = [pack(i, "honmei") for i in order if head[i] and HONMEI_MIN <= odds[i] <= HONMEI_MAX][:2]
    chu = [pack(i, "chuana") for i in order if head[i] and HONMEI_MAX < odds[i] <= CHUANA_MAX][:2]
    mixes = {}
    for key in MIXES:
        h, c = map(int, key.split("-"))
        pk = hon[:h] + chu[:c]
        if len(pk) == 2:
            mixes[key] = {"combos": [x["combo"] for x in pk], "hitp": round(sum(x["p"] for x in pk), 4)}
    return {"honmei": hon, "chuana": chu, "mixes": mixes}
