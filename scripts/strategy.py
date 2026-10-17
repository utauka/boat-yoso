"""買い目の選び方（1艇軸の3連単流し・合成オッズ5〜12倍・点数は自由・買うか見送るかの判断つき）。

  対象: 戸田・徳山・大村
  1) AIが3連単120通りそれぞれの「当たる確率」を出す
  2) 6艇それぞれを「軸」（1着固定）にして、2着・3着の組み合わせを当たる確率が高い順に1点ずつ足していく
  3) そのたびに 合成オッズ = 1 ÷（各買い目の 1/オッズ の合計）を計算し、5〜12倍に入る買い方だけを候補にする
  4) 候補の中で「当たる確率の合計」がいちばん高い買い方を選ぶ（軸と点数が決まる）
  5) 展示が出ていて、当たる確率の合計（的中確率）が HITP_MIN 以上なら「買う」、それ未満は見送り
     （期待値 = 的中確率 × 合成オッズ は参考表示。期待値で選ぶと検証のたびに成績が大きくぶれたため）
  金額は「どれが当たっても払戻がほぼ同じ」になるよう、オッズの逆数に比例して配分する。
"""
from __future__ import annotations

import numpy as np

from features import PERMS, combo_name

TARGET_STADIUMS = {2: "戸田", 18: "徳山", 24: "大村"}
SYN_MIN = 5.0     # 合成オッズの下限
SYN_MAX = 12.0    # 合成オッズの上限
HITP_MIN = 0.18   # 的中確率がこれ以上なら「買う」
SKIP_EV = 0.8     # （表示用）個別の買い目の期待値がこれ未満なら「期待値低め」
_FIRST = np.array([p[0] for p in PERMS])


def axis_pick(prob: np.ndarray, odds: np.ndarray, live: bool = False) -> dict:
    best = None
    for a in range(6):
        idx = np.where(_FIRST == a)[0]
        idx = idx[np.argsort(-prob[idx])]
        inv = np.cumsum(1.0 / np.maximum(odds[idx], 1.0))
        hp = np.cumsum(prob[idx])
        syn = 1.0 / inv
        for k in range(len(idx)):
            if SYN_MIN <= syn[k] <= SYN_MAX and (best is None or hp[k] > best[2]):
                best = (a, k + 1, float(hp[k]), float(syn[k]), idx[:k + 1])
    if best is None:
        return {"axis": None, "picks": [], "n": 0, "syn": None, "hitp": 0.0, "ev": 0.0, "buy": False}
    a, k, hp, syn, sel = best
    inv = 1.0 / np.maximum(odds[sel], 1.0)
    w = inv / inv.sum()
    picks = [{"idx": int(i), "combo": combo_name(int(i)), "p": round(float(prob[i]), 4), "odds": round(float(odds[i]), 1),
              "w": round(float(wi), 4), "low": bool(live and prob[i] * odds[i] < SKIP_EV)} for i, wi in zip(sel, w)]
    ev = hp * syn
    return {"axis": a + 1, "picks": picks, "n": k, "syn": round(syn, 2), "hitp": round(hp, 4), "ev": round(ev, 3),
            "buy": hp >= HITP_MIN}


def return_per_100(pk: dict, win_combo: str, payout: int) -> float:
    """100円を配分どおりに買ったときの払戻（円）。"""
    for c in pk["picks"]:
        if c["combo"] == win_combo:
            return c["w"] * payout
    return 0.0
