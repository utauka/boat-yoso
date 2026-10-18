"""予想モデルを学習し、過去データで成績を確かめる（バックテスト）。

  1) 着順モデル  : 各艇が「1着」「2着以内」「3着以内」に入る確率（LightGBM）
  2) 推定オッズモデル: 3連単の各組番が「市場でどのくらいの倍率になるか」の見込み
     （公式に過去オッズの配布がないため、的中した組番の払戻金から学習）
  3) 2点予想のルールを、学習に使っていない直近6か月で試し、的中率と回収率を記録

出力: models/*.txt（学習済みモデル）, site/data/backtest.json（検証結果）
"""
from __future__ import annotations

import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import MODELS, SITE_DATA, STADIUMS, payout_class, read_store, write_json
from features import (COMBO_INDEX, FEATURES, ODDS_FEATS, PERMS, build_frame, combo_frame, combo_name,
                      trifecta_probs)
from strategy import DEFAULT_MIX, IN_MIN, MIXES, TARGET_STADIUMS, in_pick

PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=31, min_data_in_leaf=60,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
ODDS_PARAMS = dict(objective="regression", learning_rate=0.04, num_leaves=63, min_data_in_leaf=40,
                   feature_fraction=0.9, verbose=-1)
ROUNDS = 500
ODDS_ROUNDS = 600


def fit_place_models(tr: pd.DataFrame):
    return {t: lgb.train(PARAMS, lgb.Dataset(tr[FEATURES], tr[t]), ROUNDS) for t in ("y1", "y2", "y3")}


def predict_probs(models, df: pd.DataFrame):
    n = len(df) // 6
    p = [models[t].predict(df[FEATURES]).reshape(n, 6) for t in ("y1", "y2", "y3")]
    # 1着確率はレース内で合計1、2連対は合計2、3連対は合計3になるよう整える
    p1 = p[0] / p[0].sum(1, keepdims=True)
    p2 = np.minimum(p[1] / p[1].sum(1, keepdims=True) * 2, 0.995)
    p3 = np.minimum(p[2] / p[2].sum(1, keepdims=True) * 3, 0.998)
    p2 = np.maximum(p2, p1); p3 = np.maximum(p3, p2)
    return p1, p2, p3, trifecta_probs(p1, p2, p3)


def winners(races_by_key, keys):
    combo, pay = [], []
    for k in keys:
        t = (races_by_key[k].get("pay") or {}).get("trifecta") or []
        if t and t[0][1] >= 100 and t[0][0] in COMBO_INDEX:
            combo.append(COMBO_INDEX[t[0][0]]); pay.append(t[0][1])
        else:
            combo.append(-1); pay.append(0)
    return np.array(combo), np.array(pay)


def fit_odds_model(df: pd.DataFrame, win_combo):
    ok = win_combo >= 0
    sub = df.iloc[np.repeat(np.where(ok)[0] * 6, 6) + np.tile(np.arange(6), ok.sum())]
    X = combo_frame(sub, only=win_combo[ok])
    return X, ok


def est_odds(odds_model, df: pd.DataFrame, correction: float = 1.0):
    """120通りの推定倍率。控除率25%に合わせて全体を調整（Σ 0.75/倍率 = 1）したうえで、
    過去の検証で分かった「実際より高めに出るクセ」(correction) で割って補正する。"""
    n = len(df) // 6
    X = combo_frame(df)
    o = np.exp(odds_model.predict(X[ODDS_FEATS])).reshape(n, 120)
    o = np.maximum(o, 1.0)
    scale = (1 / o).sum(1, keepdims=True) * 0.75
    return np.maximum(o * scale / correction, 1.0)


def main():
    races = read_store("races6")
    res_all = read_store("results_all")
    print("特徴量を作成中…")
    df = build_frame(races, res_all)
    df = df.sort_values(["key", "b"]).reset_index(drop=True)
    g = df.groupby("key")["y1"].transform("sum")
    df = df[g == 1].reset_index(drop=True)          # 結果が確定したレースだけ
    df = df[df.groupby("key")["b"].transform("count") == 6].reset_index(drop=True)
    by_key = {f'{x["d"]}_{x["j"]}_{x["r"]}': x for x in races}

    last = df["d"].max()
    cut = (dt.date.fromisoformat(last) - dt.timedelta(days=182)).isoformat()
    tr, te = df[df["d"] < cut].reset_index(drop=True), df[df["d"] >= cut].reset_index(drop=True)
    print(f"学習 {len(tr)//6}レース（〜{cut}）／検証 {len(te)//6}レース（{cut}〜{last}）")

    # ---- 1) 着順モデル
    models = fit_place_models(tr)
    p1, p2, p3, P = predict_probs(models, te)
    te_keys = te["key"].to_numpy()[::6]
    wc, wp = winners(by_key, te_keys)
    valid = wc >= 0
    ll = -np.log(np.clip(P[valid, wc[valid]], 1e-6, None)).mean()
    top1 = (P.argmax(1) == wc)[valid].mean()
    hit1 = (p1.argmax(1) == te["y1"].to_numpy().reshape(-1, 6).argmax(1)).mean()
    print(f"1着的中 {hit1:.1%}／3連単1点目の的中 {top1:.1%}／logloss {ll:.3f}（でたらめなら {np.log(120):.3f}）")

    # ---- 2) 推定オッズモデル
    trk = tr["key"].to_numpy()[::6]
    twc, twp = winners(by_key, trk)
    Xo, ok = fit_odds_model(tr, twc)
    odds_model = lgb.train(ODDS_PARAMS, lgb.Dataset(Xo[ODDS_FEATS], np.log(twp[ok] / 100)), ODDS_ROUNDS)
    O0 = est_odds(odds_model, te)
    correction = float(np.median(O0[valid, wc[valid]] / (wp[valid] / 100)))
    O = O0 / correction
    eo = O[valid, wc[valid]]; ao = wp[valid] / 100
    corr = np.corrcoef(np.log(eo), np.log(ao))[0, 1]
    cls_ok = np.mean([payout_class(a * 100) == payout_class(e * 100) for a, e in zip(ao, eo)])
    print(f"推定オッズと実際の払戻の相関 {corr:.2f}／本命・中穴・大穴の区分一致 {cls_ok:.1%}")

    # ---- 3) イン逃げ2点買い（1レース200円・配分3通り）と「買い／見送り」判断のバックテスト（対象3場・展示あり）
    rows = []
    stad = te["j"].astype(int).to_numpy()[::6]
    dates = te["d"].to_numpy()[::6]
    cmat = te["c"].to_numpy(dtype=float).reshape(-1, 6)
    hasex = ~np.isnan(te["et"].to_numpy(dtype=float).reshape(-1, 6)).any(1)
    for n in range(len(te_keys)):
        if not valid[n] or int(stad[n]) not in TARGET_STADIUMS or not hasex[n]:
            continue
        ib = int(np.argmax(cmat[n] == 1)) + 1 if (cmat[n] == 1).any() else 1
        p_in = float(p1[n][ib - 1])
        pk = in_pick(P[n], O[n], ib)
        win = combo_name(int(wc[n]))
        for key, mx in pk["mixes"].items():
            hit = win in mx["combos"]
            rows.append({"mix": key, "j": int(stad[n]), "d": dates[n], "pin": p_in, "buy": p_in >= IN_MIN,
                         "hitp": mx["hitp"], "hit": hit, "ret": int(wp[n]) if hit else 0, "in_won": win.startswith(f"{ib}-")})
    bt = pd.DataFrame(rows)
    days = bt["d"].nunique()

    def summary(sub):
        if len(sub) == 0:
            return {"races": 0, "hits": 0, "hit_rate": 0, "roi": 0, "hitp": 0, "avg_ret": 0, "in_won": 0}
        return {"races": int(len(sub)), "hits": int(sub["hit"].sum()), "hit_rate": round(sub["hit"].mean() * 100, 1),
                "roi": round(sub["ret"].sum() / (len(sub) * 200) * 100, 1), "hitp": round(sub["hitp"].mean() * 100, 1),
                "avg_ret": int(sub.loc[sub["hit"], "ret"].mean()) if sub["hit"].any() else 0,
                "in_won": round(sub["in_won"].mean() * 100, 1)}

    bands = [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1.01)]
    report = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "train_to": cut, "test_from": cut,
              "test_to": last, "in_min": IN_MIN, "days": int(days), "mixes": MIXES, "default_mix": DEFAULT_MIX,
              "stadiums": {str(k): v for k, v in TARGET_STADIUMS.items()},
              "test_races": int(bt[bt["mix"] == DEFAULT_MIX].shape[0]),
              "model": {"win_hit": round(hit1 * 100, 1), "tri_top1": round(top1 * 100, 1), "logloss": round(ll, 3),
                        "odds_corr": round(corr, 2), "odds_class_match": round(cls_ok * 100, 1), "odds_correction": round(correction, 3)},
              "buy": {k: summary(bt[(bt["mix"] == k) & bt["buy"]]) for k in MIXES},
              "every": {k: summary(bt[bt["mix"] == k]) for k in MIXES},
              "bands": {k: [{"lo": lo, "hi": hi, **summary(bt[(bt["mix"] == k) & (bt["pin"] >= lo) & (bt["pin"] < hi)])}
                            for lo, hi in bands] for k in MIXES},
              "by_stadium": {k: {str(j): summary(bt[(bt["mix"] == k) & bt["buy"] & (bt["j"] == j)]) for j in TARGET_STADIUMS}
                             for k in MIXES},
              "monthly": {k: {m: summary(bt[(bt["mix"] == k) & bt["buy"] & (bt["d"].str[:7] == m)])
                              for m in sorted(bt["d"].str[:7].unique())} for k in MIXES}}
    report["buy_per_day"] = round(report["buy"][DEFAULT_MIX]["races"] / max(days, 1), 1)
    report["odds_correction"] = round(correction, 3)
    print(f"推定オッズの補正係数 {correction:.3f}")
    for k, lab in MIXES.items():
        print(lab, "買い", report["buy"][k], "／全", report["every"][k])
    print("1日あたり", report["buy_per_day"], "R")
    # ---- 本番用：全期間で学習し直して保存
    print("全期間で学習し直して保存…")
    MODELS.mkdir(parents=True, exist_ok=True)
    final = fit_place_models(df)
    for t, m in final.items():
        m.save_model(str(MODELS / f"place_{t}.txt"))
    allk = df["key"].to_numpy()[::6]
    awc, awp = winners(by_key, allk)
    Xa, oka = fit_odds_model(df, awc)
    lgb.train(ODDS_PARAMS, lgb.Dataset(Xa[ODDS_FEATS], np.log(awp[oka] / 100)), ODDS_ROUNDS).save_model(
        str(MODELS / "odds.txt"))
    imp = sorted(zip(FEATURES, final["y1"].feature_importance("gain")), key=lambda z: -z[1])
    tot = sum(v for _, v in imp)
    report["importance"] = [[k, round(v / tot * 100, 1)] for k, v in imp[:15]]
    report["trained_to"] = last
    (MODELS / "meta.json").write_text(json.dumps({"odds_correction": correction, "trained_to": last}))
    write_json(SITE_DATA / "backtest.json", report)
    print("保存しました")


if __name__ == "__main__":
    main()
