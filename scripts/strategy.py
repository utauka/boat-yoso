"""買い目の選び方（ここを変えれば予想のクセを調整できます）。

イン逃げベース:
  1) AIが「イン（1コースの艇）が1着になる確率＝イン逃げ率」を出す
     材料: インの選手の逃げ率・まくられ率・まくり差され率・差され率、コース別成績、平均ST、展示ST、
           外の選手との展示STの差、展示タイム、モーター、外の選手のまくり率・まくり差し率、
           場のイン1着率（場全体・その風向・その風速）、風向、風速、波 など
  2) 買い目は「インが1着」の組番だけから選ぶ
       本命候補: オッズ5倍以上で、当たる確率が高い順に4点
       穴候補  : オッズ30倍以上で、当たる確率が高い順に2点
  3) 5点買いのプラン
       プランA（バランス型）  : 本命の上位3点 ＋ 穴の上位2点
       プランB（回収率重視型）: 本命の上位4点 ＋ 穴の上位1点
  4) イン逃げ率が IN_SKIP 未満のレースは「見送り推奨」
"""
from __future__ import annotations

import numpy as np

from features import PERMS, combo_name

HONMEI_MIN_ODDS = 5.0
ANA_MIN_ODDS = 30.0
IN_SKIP = 0.50   # イン逃げ率がこれ未満なら見送り推奨
IN_STRONG = 0.70  # これ以上なら「イン逃げ濃厚」
SKIP_EV = 0.8    # 本物のオッズのとき、期待値がこれ未満の買い目に「期待値低め」の印

PLANS = {
    "A": {"name": "プランA", "label": "本命3＋穴2", "type": "バランス型", "honmei": 3, "ana": 2},
    "B": {"name": "プランB", "label": "本命4＋穴1", "type": "回収率重視型", "honmei": 4, "ana": 1},
}
MAX_HONMEI = max(p["honmei"] for p in PLANS.values())
MAX_ANA = max(p["ana"] for p in PLANS.values())
_FIRST = np.array([p[0] + 1 for p in PERMS])


def in_level(p_in: float) -> str:
    return "strong" if p_in >= IN_STRONG else "normal" if p_in >= IN_SKIP else "skip"


def pick_candidates(prob: np.ndarray, odds: np.ndarray, in_boat: int, live: bool = False) -> dict:
    order = np.argsort(-prob)
    rank = np.empty(120, int); rank[order] = np.arange(120)
    ev = prob * odds
    head = _FIRST == in_boat
    hon = [int(k) for k in order if head[k] and odds[k] >= HONMEI_MIN_ODDS][:MAX_HONMEI]
    ana = [int(k) for k in order if head[k] and odds[k] >= ANA_MIN_ODDS and k not in hon][:MAX_ANA]

    def pack(i):
        return {"idx": i, "combo": combo_name(i), "p": round(float(prob[i]), 4), "odds": round(float(odds[i]), 1),
                "ev": round(float(ev[i]), 2), "rank": int(rank[i]) + 1, "low": bool(live and ev[i] < SKIP_EV)}

    return {"honmei": [pack(i) for i in hon], "ana": [pack(i) for i in ana]}


def plan_combos(cand: dict, plan: str) -> list[str]:
    p = PLANS[plan]
    return [c["combo"] for c in cand["honmei"][:p["honmei"]]] + [c["combo"] for c in cand["ana"][:p["ana"]]]
