"""買い目の選び方（1艇軸の3連単流し・合成オッズ5〜12倍・点数は自由・買うか見送るかの判断つき）。

  対象: 戸田・徳山・大村
  1) AIが3連単120通りそれぞれの「当たる確率」を出す
  2) 6艇それぞれを「軸」（1着固定）にして、2着・3着の組み合わせを当たる確率が高い順に1点ずつ足していく
  3) 1レースは500〜1,000円（100円×5〜10口）。点数は最大10点で、各買い目に100円単位で配分する
     （当たったときの払戻ができるだけそろうよう、払戻が少ない買い目に口を足していく）
     合成オッズ = どれが当たっても戻る最低倍率（最低払戻 ÷ 購入額）が5〜12倍に入る買い方だけを候補にする
  4) 候補の中で「当たる確率の合計」がいちばん高い買い方を選ぶ（軸と点数が決まる）
     金額は、合成オッズが5〜12倍に入る一番少ない金額にする（同じ的中確率なら安い方）
  5) 展示が出ていて、当たる確率の合計（的中確率）が HITP_MIN 以上なら「買う」、それ未満は見送り
     （期待値 = 的中確率 × 合成オッズ は参考表示。期待値で選ぶと検証のたびに成績が大きくぶれたため）
"""
from __future__ import annotations

import numpy as np

from features import PERMS, combo_name

TARGET_STADIUMS = {2: "戸田", 18: "徳山", 24: "大村"}
SYN_MIN = 5.0     # 合成オッズ（どれが当たっても戻る最低倍率）の下限
SYN_MAX = 12.0    # 同 上限
UNIT = 100        # 1口の金額
MIN_UNITS = 5     # 1レースの下限 500円 = 5口
MAX_UNITS = 10    # 1レースの上限 1,000円 = 10口
HITP_MIN = 0.18   # 的中確率がこれ以上なら「買う」
SKIP_EV = 0.8     # （表示用）個別の買い目の期待値がこれ未満なら「期待値低め」
_FIRST = np.array([p[0] for p in PERMS])


def allocate(odds: np.ndarray, units: int = MAX_UNITS) -> np.ndarray:
    """各買い目に1口ずつ置き、残りの口を「当たったときの払戻が一番少ない買い目」に足していく。"""
    u = np.ones(len(odds), int)
    for _ in range(units - len(odds)):
        u[int(np.argmin(u * odds))] += 1
    return u


def axis_pick(prob: np.ndarray, odds: np.ndarray, live: bool = False) -> dict:
    best = None
    for a in range(6):
        idx = np.where(_FIRST == a)[0]
        idx = idx[np.argsort(-prob[idx])]
        for k in range(1, MAX_UNITS + 1):
            sel = idx[:k]
            o = np.maximum(odds[sel], 1.0)
            hp = float(prob[sel].sum())
            if best is not None and hp < best[2] - 1e-12:
                continue
            for U in range(max(MIN_UNITS, k), MAX_UNITS + 1):
                u = allocate(o, U)
                syn = float((u * o).min() / U)
                if SYN_MIN <= syn <= SYN_MAX:
                    if best is None or hp > best[2] + 1e-12 or (abs(hp - best[2]) <= 1e-12 and U < int(best[5].sum())):
                        best = (a, k, hp, syn, sel, u)
                    break
    if best is None:
        return {"axis": None, "picks": [], "n": 0, "syn": None, "hitp": 0.0, "ev": 0.0, "buy": False, "stake": 0}
    a, k, hp, syn, sel, u = best
    U = int(u.sum())
    picks = [{"idx": int(i), "combo": combo_name(int(i)), "p": round(float(prob[i]), 4), "odds": round(float(odds[i]), 1),
              "u": int(ui), "yen": int(ui) * UNIT, "w": round(int(ui) / U, 4),
              "low": bool(live and prob[i] * odds[i] < SKIP_EV)} for i, ui in zip(sel, u)]
    ev = float(sum(prob[i] * odds[i] * ui for i, ui in zip(sel, u)) / U)
    return {"axis": a + 1, "picks": picks, "n": k, "syn": round(syn, 2), "hitp": round(hp, 4), "ev": round(ev, 3),
            "buy": hp >= HITP_MIN, "stake": U * UNIT}


def return_per_100(pk: dict, win_combo: str, payout: int) -> float:
    """100円を配分どおりに買ったときの払戻（円）。"""
    for c in pk["picks"]:
        if c["combo"] == win_combo:
            return c["w"] * payout
    return 0.0
