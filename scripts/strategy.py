"""買い目の選び方（3点買い・本命＋中穴・買うか見送るかの判断つき）。

  1) AIが3連単120通りそれぞれの「当たる確率」を出す
  2) オッズで分ける: 本命＝5〜20倍、中穴＝20倍より上〜60倍（大穴は使わない）
  3) 3点は「本命2点＋中穴1点」か「本命1点＋中穴2点」のうち、当たる確率の合計が高い方
     （各区分の中では、当たる確率が高い順）
  4) 3点の当たる確率の合計を「的中の自信」とし、CONF_MIN 以上のレースだけ「買う」
"""
from __future__ import annotations

import numpy as np

from features import combo_name

HONMEI_MIN = 5.0
HONMEI_MAX = 20.0
CHUANA_MAX = 60.0
CONF_MIN = 0.25   # 的中の自信がこれ以上なら「買う」
SKIP_EV = 0.8     # 本物のオッズのとき、期待値がこれ未満の買い目に「期待値低め」の印


def odds_kind(o: float) -> str | None:
    if o < HONMEI_MIN or o > CHUANA_MAX:
        return None
    return "honmei" if o <= HONMEI_MAX else "chuana"


def pick3(prob: np.ndarray, odds: np.ndarray, live: bool = False) -> dict:
    order = np.argsort(-prob)
    rank = np.empty(120, int); rank[order] = np.arange(120)
    hon = [int(k) for k in order if odds_kind(float(odds[k])) == "honmei"][:2]
    chu = [int(k) for k in order if odds_kind(float(odds[k])) == "chuana"][:2]
    opts = [x for x in (hon[:2] + chu[:1], hon[:1] + chu[:2]) if len(x) == 3] or [(hon + chu)[:3]]
    pk = max(opts, key=lambda x: float(prob[x].sum()) if x else 0.0)
    pk = sorted(pk, key=lambda i: -prob[i])
    conf = float(prob[pk].sum()) if pk else 0.0
    ev = prob * odds
    picks = [{"idx": i, "combo": combo_name(i), "kind": odds_kind(float(odds[i])), "p": round(float(prob[i]), 4),
              "odds": round(float(odds[i]), 1), "ev": round(float(ev[i]), 2), "rank": int(rank[i]) + 1,
              "low": bool(live and ev[i] < SKIP_EV)} for i in pk]
    return {"picks": picks, "conf": round(conf, 4), "buy": conf >= CONF_MIN}
