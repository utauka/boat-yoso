"""買い目の選び方（3点買い・本命／中穴／大穴の配分を画面で切り替え）。

  1) AIが3連単120通りそれぞれの「当たる確率」を出す
  2) オッズで3つに分ける
       本命: 5倍以上〜20倍以下
       中穴: 20倍より上〜60倍以下
       大穴: 60倍より上
  3) それぞれの区分で、当たる確率が高い順に3点ずつ候補を出す
  4) 画面で選んだ配分（例：本命1・中穴1・大穴1）に合わせて、各区分の上位から合計3点を買う

配分は全10通り（MIXES）。画面の初期設定は DEFAULT_MIX です。
"""
from __future__ import annotations

import numpy as np

from features import combo_name

HONMEI_MIN = 5.0     # 本命の下限（これ未満の組番は買い目にしない）
HONMEI_MAX = 20.0
CHUANA_MAX = 60.0
TOP_N = 3
POINTS = 3
SKIP_EV = 0.8        # 本物のオッズのとき、期待値がこれ未満の買い目に「期待値低め」の印
KINDS = ("honmei", "chuana", "oana")
MIXES = [(h, c, POINTS - h - c) for h in range(POINTS, -1, -1) for c in range(POINTS - h, -1, -1)]
DEFAULT_MIX = (1, 1, 1)


def odds_kind(o: float) -> str | None:
    if o < HONMEI_MIN:
        return None
    return "honmei" if o <= HONMEI_MAX else "chuana" if o <= CHUANA_MAX else "oana"


def pick_candidates(prob: np.ndarray, odds: np.ndarray, live: bool = False) -> dict:
    order = np.argsort(-prob)
    rank = np.empty(120, int); rank[order] = np.arange(120)
    ev = prob * odds
    out = {k: [] for k in KINDS}
    for i in order:
        k = odds_kind(float(odds[i]))
        if k and len(out[k]) < TOP_N:
            out[k].append({"idx": int(i), "combo": combo_name(int(i)), "p": round(float(prob[i]), 4),
                           "odds": round(float(odds[i]), 1), "ev": round(float(ev[i]), 2), "rank": int(rank[i]) + 1,
                           "low": bool(live and ev[i] < SKIP_EV)})
        if all(len(v) >= TOP_N for v in out.values()):
            break
    return out


def mix_key(mix) -> str:
    return "-".join(map(str, mix))


def mix_combos(cand: dict, mix) -> list[str]:
    out = []
    for k, n in zip(KINDS, mix):
        out += [c["combo"] for c in cand[k][:n]]
    return out
