"""当日の6場の全レースについて2点予想を作り、site/data/predictions.json に書き出す。

使い方:
  python scripts/predict.py                # 今日の予想（推定オッズ）
  python scripts/predict.py --live-odds    # 締切が近いレースは公式の実オッズを取得して使う
  python scripts/predict.py --date 2026-10-16
一度締切を過ぎたレースの予想は、締切前に作ったものをそのまま残します（あと出しを防ぐため）。
結果が出たレースは、的中したかどうかを書き込み、実績（site/data/pred_log.json）に記録します。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (CLASS_NAMES, DATA, JST, MODELS, SITE_DATA, env_local_root, load_day_raw, normalize,
                    payout_class, read_store, write_json)
from features import COMBO_INDEX, CourseHistory, combo_name, prep, race_rows
from odds import fetch_odds3t
from strategy import HITP_MIN, SYN_MAX, SYN_MIN, TARGET_STADIUMS, axis_pick, return_per_100
from train import est_odds, predict_probs

PRED = SITE_DATA / "predictions.json"
LIVE = DATA / "live_odds.json"   # 取得した実オッズ（公開フォルダには置かない）
LOG = SITE_DATA / "pred_log.json"


def load_models():
    m = {t: lgb.Booster(model_file=str(MODELS / f"place_{t}.txt")) for t in ("y1", "y2", "y3")}
    om = lgb.Booster(model_file=str(MODELS / "odds.txt"))
    meta = json.loads((MODELS / "meta.json").read_text())
    return m, om, meta


def reasons(x: dict, p1: np.ndarray, rows: list[dict], pk: dict) -> list[str]:
    """画面に出す根拠（展示を中心に）。"""
    boats = x["boats"]
    out = []
    if pk.get("axis"):
        a = pk["axis"] - 1
        b = boats[a]
        out.append(f'軸は{b["b"]}号艇 {b.get("name","")}（{CLASS_NAMES.get(b.get("cls"),"")}・勝率{b.get("nwr")}）、1着率{p1[a]*100:.0f}%')
    ets = [(bb.get("ex") or {}).get("et") for bb in boats]
    if not all(ets):
        out.append("展示がまだ出ていません（展示後に判断します）")
        return out
    k = int(np.argmin(ets))
    srt = sorted(ets)
    out.append(f'展示タイム1位は{boats[k]["b"]}号艇（{ets[k]:.2f}、2位と{srt[1]-srt[0]:.2f}差）')
    selfs = [r.get("et_self") for r in rows]
    ok = [i for i, v in enumerate(selfs) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if ok:
        k = min(ok, key=lambda i: selfs[i])
        if selfs[k] < 0:
            out.append(f'ふだんより展示が一番良いのは{boats[k]["b"]}号艇（この場の直近平均より{-selfs[k]:.2f}秒速い）')
        k2 = max(ok, key=lambda i: selfs[i])
        if selfs[k2] > 0.05:
            out.append(f'{boats[k2]["b"]}号艇はふだんより展示が{selfs[k2]:.2f}秒遅い')
    exst = [(bb.get("ex") or {}).get("st") for bb in boats]
    if all(v is not None for v in exst):
        k = int(np.argmin(exst))
        out.append(f'展示スタート最速は{boats[k]["b"]}号艇（{exst[k]:.2f}、平均ST {boats[k].get("ast")}）')
    moves = [bb["b"] for bb in boats if (bb.get("ex") or {}).get("c") and bb["ex"]["c"] != bb["b"]]
    if moves:
        out.append("展示で進入変化あり（" + "・".join(f'{m}号艇' for m in moves) + "）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--live-odds", action="store_true")
    ap.add_argument("--window", type=int, default=25, help="締切まで何分以内のレースの実オッズを取るか")
    a = ap.parse_args()
    now = dt.datetime.now(JST)
    day = dt.date.fromisoformat(a.date) if a.date else now.date()
    raw = load_day_raw(day, env_local_root())
    races = [x for x in normalize(raw) if x["j"] in TARGET_STADIUMS] if raw else []
    races.sort(key=lambda x: (x.get("ca") or "", x["j"], x["r"]))

    prev = json.loads(PRED.read_text()) if PRED.exists() else {}
    prev_races = {(z["j"], z["r"]): z for z in prev.get("races", [])} if prev.get("date") == day.isoformat() else {}

    live_store = json.loads(LIVE.read_text()) if LIVE.exists() else {}
    if live_store.get("date") != day.isoformat():
        live_store = {"date": day.isoformat(), "odds": {}}
    models, om, meta = load_models()
    hist = CourseHistory(read_store("results_all"), read_store("races6"))
    hist.advance(day.isoformat())

    out_races = []
    for x in races:
        rows = race_rows(x, hist)
        if not rows:
            continue
        df = prep(pd.DataFrame(rows))
        p1, p2, p3, P = predict_probs(models, df)
        O = est_odds(om, df, meta["odds_correction"])[0]
        ca = dt.datetime.fromisoformat(x["ca"]).replace(tzinfo=JST) if x.get("ca") else None
        closed = bool(ca and now >= ca)
        old = prev_races.get((x["j"], x["r"]))

        if old and old.get("frozen") and "axis" in old:
            item = old  # 締切前に確定した予想を残す
        else:
            odds_src, live = "est", None
            mins = (ca - now).total_seconds() / 60 if ca else None
            key = f'{x["j"]}_{x["r"]}'
            if a.live_odds and mins is not None and 0 < mins <= a.window:
                live = fetch_odds3t(x["j"], x["r"], day.strftime("%Y%m%d"))
                if live:
                    live_store["odds"][key] = live
            if not live:
                live = live_store["odds"].get(key)
            O_use = O.copy()
            if live and len(live) >= 100:
                for k, v in live.items():
                    if k in COMBO_INDEX:
                        O_use[COMBO_INDEX[k]] = v
                odds_src = "live"
            pk = axis_pick(P[0], O_use, live=odds_src == "live")
            has_ex = all((b.get("ex") or {}).get("et") for b in x["boats"])
            state = ("buy" if pk["buy"] else "skip") if has_ex else "wait"
            top = np.argsort(-P[0])[:6]
            item = {
                "j": x["j"], "r": x["r"], "ca": x.get("ca"), "title": x.get("title"), "sub": x.get("sub"),
                "has_preview": has_ex,
                "boats": [{"b": b["b"], "num": b.get("num"), "name": b.get("name"), "cls": CLASS_NAMES.get(b.get("cls"), "-"),
                           "wr": b.get("nwr"), "m2": b.get("m2"), "et": (b.get("ex") or {}).get("et"),
                           "c": (b.get("ex") or {}).get("c"), "exst": (b.get("ex") or {}).get("st"), "ast": b.get("ast"),
                           "p1": round(float(p1[0][i]) * 100, 1), "p3": round(float(p3[0][i]) * 100, 1)}
                          for i, b in enumerate(x["boats"])],
                "axis": pk["axis"], "picks": pk["picks"], "n": pk["n"], "syn": pk["syn"], "hitp": pk["hitp"], "ev": pk["ev"],
                "state": state, "buy": state == "buy", "odds_source": odds_src,
                "exs": [{"self": None if r.get("et_self") is None or np.isnan(r["et_self"]) else round(float(r["et_self"]), 2)}
                        for r in rows],
                "top": [{"c": combo_name(int(i)), "p": round(float(P[0][i]) * 100, 1), "odds": round(float(O_use[i]), 1)}
                        for i in top],
                "reasons": reasons(x, p1[0], rows, pk),
                "made_at": now.isoformat(timespec="minutes"),
                "made_after_close": closed,
                "frozen": closed,
            }
        # 結果
        t = (x.get("pay") or {}).get("trifecta") or []
        if t:
            item["result"] = {"combo": t[0][0], "pay": t[0][1], "cls": payout_class(t[0][1])}
            for c in item["picks"]:
                c["hit"] = c["combo"] == t[0][0]
            item["hit"] = any(c["hit"] for c in item["picks"])
            item["ret100"] = round(return_per_100(item, t[0][0], t[0][1]), 1)
            item["frozen"] = True
        out_races.append(item)

    write_json(LIVE, live_store)
    write_json(PRED, {"date": day.isoformat(), "generated": now.isoformat(timespec="minutes"), "hitp_min": HITP_MIN, "syn": [SYN_MIN, SYN_MAX],
                      "stadiums": {str(j): n for j, n in TARGET_STADIUMS.items()}, "races": out_races})

    # 実績ログ（結果が出たレースだけ）
    COLS = ["date", "j", "r", "odds_source", "combos", "n", "syn", "hitp", "state", "result", "pay", "hit", "ret100", "after_close"]
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    keep = [r for r in log.get("rows", []) if r[0] != day.isoformat()] if log.get("cols") == COLS else []
    for z in out_races:
        if not z.get("result") or not z.get("picks"):
            continue
        keep.append([day.isoformat(), z["j"], z["r"], z["odds_source"], "/".join(c["combo"] for c in z["picks"]), z["n"],
                     z["syn"], z["hitp"], z["state"], z["result"]["combo"], z["result"]["pay"], int(bool(z.get("hit"))),
                     z.get("ret100", 0), int(bool(z.get("made_after_close")))])
    keep.sort()
    write_json(LOG, {"cols": COLS, "rows": keep})
    done = [z for z in out_races if z.get("result")]
    buys = [z for z in out_races if z.get("buy")]
    print(f"{day}: {len(out_races)}レースの予想（買う {len(buys)}／結果確定 {len(done)}）")


if __name__ == "__main__":
    main()
