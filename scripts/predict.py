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

from common import (CLASS_NAMES, DATA, JST, MODELS, SITE_DATA, STADIUMS, env_local_root, load_day_raw, normalize,
                    payout_class, read_store, write_json)
from features import COMBO_INDEX, CourseHistory, combo_name, in_diagnosis, prep, race_rows
from odds import fetch_odds3t
from strategy import DEFAULT_MIX, HONMEI_MIN, KINDS, MIXES, mix_key, pick_candidates
from train import est_odds, predict_probs

PRED = SITE_DATA / "predictions.json"
LIVE = DATA / "live_odds.json"   # 取得した実オッズ（公開フォルダには置かない）
LOG = SITE_DATA / "pred_log.json"


def load_models():
    m = {t: lgb.Booster(model_file=str(MODELS / f"place_{t}.txt")) for t in ("y1", "y2", "y3")}
    om = lgb.Booster(model_file=str(MODELS / "odds.txt"))
    meta = json.loads((MODELS / "meta.json").read_text())
    return m, om, meta


def reasons(x: dict, p1: np.ndarray, diag: dict) -> list[str]:
    boats = x["boats"]
    ib = diag["b"]
    b = boats[ib - 1]
    out = [f'イン{ib}号艇 {b.get("name","")}（{CLASS_NAMES.get(b.get("cls"),"")}）のイン逃げ率 {p1[ib-1]*100:.0f}%']
    if diag.get("nige") is not None:
        out.append(f'この選手のイン成績（直近1年{diag["in_starts"]}走）: 逃げ{diag["nige"]:.0f}%・まくられ{diag["makurare"]:.0f}%・まくり差され{diag["mzasare"]:.0f}%')
    if diag.get("gap") is not None:
        g = diag["gap"]
        out.append(f'展示ST: イン{diag["exst"]:.2f}／外の最速との差 {g:+.2f}' + ("（外の方が速い）" if g < 0 else ""))
    others = [k for k in range(6) if k != ib - 1]
    best = max(others, key=lambda k: p1[k])
    out.append(f'逆転の一番手は{boats[best]["b"]}号艇（1着率{p1[best]*100:.0f}%）')
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
    races = [x for x in normalize(raw) if x["j"] in STADIUMS] if raw else []
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
        diag = in_diagnosis(rows)
        ib = diag["b"]
        p_in = float(p1[0][ib - 1])
        diag.update(p=round(p_in * 100, 1), level="strong" if p_in >= 0.7 else "normal" if p_in >= 0.5 else "weak")
        O = est_odds(om, df, meta["odds_correction"])[0]
        ca = dt.datetime.fromisoformat(x["ca"]).replace(tzinfo=JST) if x.get("ca") else None
        closed = bool(ca and now >= ca)
        old = prev_races.get((x["j"], x["r"]))

        if old and old.get("frozen") and "cands" in old:
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
            pk = pick_candidates(P[0], O_use, live=odds_src == "live")
            top = np.argsort(-P[0])[:6]
            item = {
                "j": x["j"], "r": x["r"], "ca": x.get("ca"), "title": x.get("title"), "sub": x.get("sub"),
                "has_preview": any(b.get("ex") for b in x["boats"]),
                "boats": [{"b": b["b"], "num": b.get("num"), "name": b.get("name"), "cls": CLASS_NAMES.get(b.get("cls"), "-"),
                           "wr": b.get("nwr"), "m2": b.get("m2"), "et": (b.get("ex") or {}).get("et"),
                           "c": (b.get("ex") or {}).get("c"), "exst": (b.get("ex") or {}).get("st"),
                           "p1": round(float(p1[0][i]) * 100, 1), "p3": round(float(p3[0][i]) * 100, 1)}
                          for i, b in enumerate(x["boats"])],
                "cands": pk, "odds_source": odds_src, "in": diag,
                "top": [{"c": combo_name(int(i)), "p": round(float(P[0][i]) * 100, 1), "odds": round(float(O_use[i]), 1)}
                        for i in top],
                "reasons": reasons(x, p1[0], diag),
                "made_at": now.isoformat(timespec="minutes"),
                "made_after_close": closed,
                "frozen": closed,
            }
        # 結果
        t = (x.get("pay") or {}).get("trifecta") or []
        if t:
            item["result"] = {"combo": t[0][0], "pay": t[0][1], "cls": payout_class(t[0][1])}
            for k in KINDS:
                for c in item["cands"].get(k) or []:
                    c["hit"] = c["combo"] == t[0][0]
            item["frozen"] = True
        out_races.append(item)

    write_json(LIVE, live_store)
    write_json(PRED, {"date": day.isoformat(), "generated": now.isoformat(timespec="minutes"), "mixes": [mix_key(m) for m in MIXES], "default_mix": mix_key(DEFAULT_MIX), "honmei_min": HONMEI_MIN,
                      "stadiums": {str(j): n for j, n in STADIUMS.items()}, "races": out_races})

    # 実績ログ（結果が出たレースだけ）: 候補を区分ごとに残し、画面でどの配分でも計算できるようにする
    COLS = ["date", "j", "r", "odds_source", "honmei", "chuana", "oana", "result", "pay", "after_close"]
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    keep = [r for r in log.get("rows", []) if r[0] != day.isoformat()] if log.get("cols") == COLS else []
    for z in out_races:
        if not z.get("result"):
            continue
        keep.append([day.isoformat(), z["j"], z["r"], z["odds_source"]]
                    + ["/".join(c["combo"] for c in z["cands"][k]) for k in KINDS]
                    + [z["result"]["combo"], z["result"]["pay"], int(bool(z.get("made_after_close")))])
    keep.sort()
    write_json(LOG, {"cols": COLS, "rows": keep})
    done = [z for z in out_races if z.get("result")]
    print(f"{day}: {len(out_races)}レースの予想（結果確定 {len(done)}）")


if __name__ == "__main__":
    main()
