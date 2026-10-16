"""2点予想の選び方（ここを変えれば予想のクセを調整できます）。

  本命1点: オッズ5倍以上の中で、AIが出した「当たる確率」が一番高い組番
  穴1点  : オッズ30倍以上の中で、当たる確率が一番高い組番
  期待値  = 当たる確率 × オッズ（1.0を超えると「理屈の上ではプラス」）

過去6か月の検証では、期待値が一番高いものを選ぶより、確率が一番高いものを選ぶ方が
的中率・回収率ともに良かったため、この方式にしています（site/data/backtest.json 参照）。
期待値が SKIP_EV 未満のときは「見送り推奨」の印をつけます（本物のオッズが取れたときだけ）。
"""
from __future__ import annotations

import numpy as np

from features import combo_name

HONMEI_MIN_ODDS = 5.0
ANA_MIN_ODDS = 30.0
SKIP_EV = 0.8


def pick_two(prob: np.ndarray, odds: np.ndarray, live: bool = False) -> dict:
    ev = prob * odds
    order = np.argsort(-prob)
    rank = np.empty(120, int); rank[order] = np.arange(120)
    out = {}
    for kind, th in (("honmei", HONMEI_MIN_ODDS), ("ana", ANA_MIN_ODDS)):
        cand = np.where(odds >= th)[0]
        if not len(cand):
            out[kind] = None
            continue
        i = int(cand[np.argmax(prob[cand])])
        out[kind] = {"idx": i, "combo": combo_name(i), "p": round(float(prob[i]), 4),
                     "odds": round(float(odds[i]), 1), "ev": round(float(ev[i]), 2), "rank": int(rank[i]) + 1,
                     "skip": bool(live and ev[i] < SKIP_EV)}
    return out
