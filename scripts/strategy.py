"""買い目の選び方（ここを変えれば予想のクセを調整できます）。

  本命候補: オッズ5倍以上の中で、AIの「当たる確率」が高い順に4点
  穴候補  : オッズ30倍以上の中で、当たる確率が高い順に2点（本命候補と重なるものは除く）
  この候補から、2つの5点買いプランを作ります。
    プランA（バランス型）  : 本命の上位3点 ＋ 穴の上位2点
    プランB（回収率重視型）: 本命の上位4点 ＋ 穴の上位1点
  期待値 = 当たる確率 × オッズ（1.0を超えると「理屈の上ではプラス」）

過去6か月の検証では、期待値が一番高いものより、確率が高いものから選ぶ方が
的中率・回収率ともに良かったため、この方式にしています（site/data/backtest.json 参照）。
"""
from __future__ import annotations

import numpy as np

from features import combo_name

HONMEI_MIN_ODDS = 5.0
ANA_MIN_ODDS = 30.0
SKIP_EV = 0.8   # 本物のオッズのとき、期待値がこれ未満の買い目に「期待値低め」の印をつける

PLANS = {
    "A": {"name": "プランA", "label": "本命3＋穴2", "type": "バランス型", "honmei": 3, "ana": 2},
    "B": {"name": "プランB", "label": "本命4＋穴1", "type": "回収率重視型", "honmei": 4, "ana": 1},
}
MAX_HONMEI = max(p["honmei"] for p in PLANS.values())
MAX_ANA = max(p["ana"] for p in PLANS.values())


def pick_candidates(prob: np.ndarray, odds: np.ndarray, live: bool = False) -> dict:
    order = np.argsort(-prob)
    rank = np.empty(120, int); rank[order] = np.arange(120)
    ev = prob * odds
    hon = [int(k) for k in order if odds[k] >= HONMEI_MIN_ODDS][:MAX_HONMEI]
    ana = [int(k) for k in order if odds[k] >= ANA_MIN_ODDS and k not in hon][:MAX_ANA]

    def pack(i):
        return {"idx": i, "combo": combo_name(i), "p": round(float(prob[i]), 4), "odds": round(float(odds[i]), 1),
                "ev": round(float(ev[i]), 2), "rank": int(rank[i]) + 1, "low": bool(live and ev[i] < SKIP_EV)}

    return {"honmei": [pack(i) for i in hon], "ana": [pack(i) for i in ana]}


def plan_combos(cand: dict, plan: str) -> list[str]:
    p = PLANS[plan]
    return [c["combo"] for c in cand["honmei"][:p["honmei"]]] + [c["combo"] for c in cand["ana"][:p["ana"]]]
