"""予想に使う「特徴量（AIに渡す材料）」を作る。学習と当日予想の両方で使う。"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from itertools import permutations

import numpy as np
import pandas as pd

BOAT_COLS = ["cls", "age", "wt", "f", "l", "ast", "nwr", "n2", "n3", "lwr", "l2", "l3", "m2", "m3", "b2", "b3"]
# イン逃げを判断するための材料（インの選手の逃げ率・まくられ率など）
IN_FEATS = ["nige_rate", "makurare_rate", "mzasare_rate", "sasare_rate", "in_starts",
            "makuri_rate", "mzashi_rate", "sashi_rate", "att_starts",
            "in_nige", "in_makurare", "in_mzasare", "in_sasare", "in_ast", "in_exst", "in_et", "in_m2", "in_nwr",
            "in_cls", "exst_vs_in", "et_vs_in", "out_best_exst_gap", "out_max_makuri", "out_max_mzashi",
            "st_in_rate", "st_wd_in_rate", "st_ws_in_rate"]
# 展示を深く見るための材料
EX_FEATS = ["et_z", "et_self", "et_self_rank", "et_self_n", "exst_vs_ast", "exst_z", "et_gap2", "ex_both_top"]
FEATURES = (
    ["j", "r", "g", "dist", "b", "c", "c_move", "ws", "wd", "wv", "wt_w"]
    + BOAT_COLS
    + ["et", "exst", "tilt", "wadj", "parts",
       "nwr_rank", "nwr_diff", "et_rank", "et_diff", "exst_rank", "m2_rank", "ast_rank", "lwr_rank",
       "a1_cnt", "max_nwr_other", "cw_rate", "c2_rate", "c_starts", "inside_nwr", "inside_exst"]
    + IN_FEATS
    + EX_FEATS
)
CAT = ["j", "wd"]
PERMS = list(permutations(range(6), 3))  # 120通り（0始まりの艇の位置）


def ws_bin(ws):
    if ws is None:
        return None
    return 0 if ws <= 2 else 1 if ws <= 4 else 2 if ws <= 6 else 3


class _Rolling:
    """日付つきの出来事を、指定した日数の範囲だけ数えておく箱（未来の情報は使わない）。"""

    def __init__(self, events, size, days=None):
        self.ev = sorted(events, key=lambda e: e[0])
        self.size, self.days = size, days
        self.ia = self.idel = 0
        self.cnt = {}

    def advance(self, day):
        ev = self.ev
        while self.ia < len(ev) and ev[self.ia][0] < day:
            _, k, v = ev[self.ia]
            c = self.cnt.get(k)
            if c is None:
                c = self.cnt[k] = [0] * self.size
            for i, x in enumerate(v):
                c[i] += x
            self.ia += 1
        if self.days:
            lo = (dt.date.fromisoformat(day) - dt.timedelta(days=self.days)).isoformat()
            while self.idel < self.ia and ev[self.idel][0] < lo:
                _, k, v = ev[self.idel]
                c = self.cnt[k]
                for i, x in enumerate(v):
                    c[i] -= x
                self.idel += 1

    def get(self, k):
        return self.cnt.get(k) or [0] * self.size


class CourseHistory:
    """その日より前のデータだけで数える成績。
      - 選手×コース : 出走・1着・2連対（直近1年・全国）
      - 選手の型   : インでの 逃げ・まくられ・まくり差され・差され、外からの まくり・まくり差し・差し（直近1年・全国）
      - 場の傾向   : 場ごと／場×風向／場×風速 のイン1着率（対象6場の全期間）
    """

    def __init__(self, results_all: list[dict], races6: list[dict] | None = None, days: int = 365):
        ce, se = [], []
        for x in results_all:
            t = x.get("t")
            win = next((b for b in x["boats"] if b[4] == 1), None)
            for _, num, c, _, pl in x["boats"]:
                if not num or not c or pl is None or pl == 16:
                    continue
                ce.append((x["d"], (num, c), (1, int(pl == 1), int(pl in (1, 2)))))
                if t is None or win is None:
                    continue
                won = win[1] == num
                if c == 1:  # 0:イン出走 1:逃げ 2:まくられ 3:まくり差され 4:差され
                    se.append((x["d"], num, (1, int(won and t == 1), int(not won and t == 3),
                                             int(not won and t == 4), int(not won and t == 2), 0, 0, 0, 0)))
                else:       # 5:外から出走 6:まくり 7:まくり差し 8:差し
                    se.append((x["d"], num, (0, 0, 0, 0, 0, 1, int(won and t == 3), int(won and t == 4),
                                             int(won and t == 2))))
        self.course = _Rolling(ce, 3, days)
        self.style = _Rolling(se, 9, days)
        ee = []
        for x in races6 or []:
            inb = next((b for b in x["boats"] if (b.get("res") or {}).get("c") == 1), None)
            if not inb or (inb.get("res") or {}).get("pl") is None:
                continue
            w = x.get("rw") or x.get("pw") or {}
            v = (1, int(inb["res"]["pl"] == 1))
            ee += [(x["d"], ("j", x["j"]), v), (x["d"], ("wd", x["j"], w.get("wd")), v),
                   (x["d"], ("ws", x["j"], ws_bin(w.get("ws"))), v)]
        self.env = _Rolling(ee, 2, None)
        # 選手×場ごとの展示タイムの履歴（ふだんの展示タイムとの比較用）
        import bisect as _b
        self._bisect = _b
        et_hist = defaultdict(list)
        for x in races6 or []:
            for b in x.get("boats") or []:
                et = (b.get("ex") or {}).get("et")
                if b.get("num") and et:
                    et_hist[(b["num"], x["j"])].append((x["d"], et))
        self.et_hist = {k: (sorted(v), [d for d, _ in sorted(v)]) for k, v in et_hist.items()}

    def advance(self, day: str):
        self.course.advance(day); self.style.advance(day); self.env.advance(day)

    def get(self, num, c):
        return self.course.get((num, c))

    def et_avg(self, num, j, day, last=10):
        h = self.et_hist.get((num, j))
        if not h:
            return np.nan, 0
        rows, ds = h
        i = self._bisect.bisect_left(ds, day)
        v = [e for _, e in rows[max(0, i - last):i]]
        return (float(np.mean(v)), len(v)) if len(v) >= 3 else (np.nan, len(v))

    def style_of(self, num):
        return self.style.get(num)

    def env_rate(self, key, min_n=30):
        n, w = self.env.get(key)
        return w / n if n >= min_n else np.nan


def _rank(vals, ascending=False):
    s = pd.Series(vals, dtype="float")
    return s.rank(ascending=ascending, method="min").tolist()


def _rate(a, n, min_n=5):
    return a / n if n >= min_n else np.nan


def race_rows(x: dict, hist: CourseHistory | None) -> list[dict] | None:
    boats = x.get("boats") or []
    if len(boats) != 6 or any(b.get("nwr") is None for b in boats):
        return None
    w = x.get("pw") or x.get("rw") or {}
    exs = [b.get("ex") or {} for b in boats]
    get = lambda k: [b.get(k) for b in boats]
    nwr, et, exst = get("nwr"), [e.get("et") for e in exs], [e.get("st") for e in exs]
    nwr_f = [v if v is not None else np.nan for v in nwr]
    et_f = [v if v else np.nan for v in et]
    exst_f = [v if v is not None else np.nan for v in exst]
    courses = [e.get("c") or b["b"] for e, b in zip(exs, boats)]
    ranks = {"nwr": _rank(nwr_f), "et": _rank(et_f, True), "exst": _rank(exst_f, True),
             "m2": _rank([v if v is not None else np.nan for v in get("m2")]),
             "ast": _rank([v if v is not None else np.nan for v in get("ast")], True),
             "lwr": _rank([v if v is not None else np.nan for v in get("lwr")])}
    mean_nwr = np.nanmean(nwr_f)
    min_et = np.nanmin(et_f) if not all(np.isnan(et_f)) else np.nan
    a1 = sum(1 for b in boats if b.get("cls") == 1)
    # イン（1コース）の艇と、選手の型
    ii = courses.index(1) if 1 in courses else 0
    sty = [hist.style_of(b.get("num")) if hist else [0] * 9 for b in boats]
    srate = [{"nige": _rate(s[1], s[0]), "makurare": _rate(s[2], s[0]), "mzasare": _rate(s[3], s[0]),
              "sasare": _rate(s[4], s[0]), "makuri": _rate(s[6], s[5], 10), "mzashi": _rate(s[7], s[5], 10),
              "sashi": _rate(s[8], s[5], 10)} for s in sty]
    outs = [k for k in range(6) if k != ii]
    in_exst = exst_f[ii]
    out_best_exst = np.nanmin([exst_f[k] for k in outs]) if any(not np.isnan(exst_f[k]) for k in outs) else np.nan
    nanmax = lambda a: np.nanmax(a) if any(not np.isnan(v) for v in a) else np.nan
    race_in = {"in_nige": srate[ii]["nige"], "in_makurare": srate[ii]["makurare"], "in_mzasare": srate[ii]["mzasare"],
               "in_sasare": srate[ii]["sasare"], "in_ast": boats[ii].get("ast"), "in_exst": in_exst,
               "in_et": et_f[ii], "in_m2": boats[ii].get("m2"), "in_nwr": boats[ii].get("nwr"),
               "in_cls": boats[ii].get("cls"), "out_best_exst_gap": out_best_exst - in_exst,
               "out_max_makuri": nanmax([srate[k]["makuri"] for k in outs]),
               "out_max_mzashi": nanmax([srate[k]["mzashi"] for k in outs]),
               "st_in_rate": hist.env_rate(("j", x["j"])) if hist else np.nan,
               "st_wd_in_rate": hist.env_rate(("wd", x["j"], w.get("wd"))) if hist else np.nan,
               "st_ws_in_rate": hist.env_rate(("ws", x["j"], ws_bin(w.get("ws")))) if hist else np.nan}
    # 展示の材料
    et_arr = np.array(et_f, float)
    et_sd = np.nanstd(et_arr) if np.sum(~np.isnan(et_arr)) >= 3 else np.nan
    ex_arr = np.array(exst_f, float)
    ex_sd = np.nanstd(ex_arr) if np.sum(~np.isnan(ex_arr)) >= 3 else np.nan
    selfs = [hist.et_avg(b.get("num"), x["j"], x["d"]) if hist else (np.nan, 0) for b in boats]
    et_self = [et_arr[k] - selfs[k][0] if not np.isnan(et_arr[k]) else np.nan for k in range(6)]
    et_self_rank = _rank(et_self, True)
    srt = np.sort(et_arr[~np.isnan(et_arr)])
    rows = []
    for i, (b, e) in enumerate(zip(boats, exs)):
        c = courses[i]
        h = hist.get(b.get("num"), c) if hist else [0, 0, 0]
        inside = [k for k in range(6) if courses[k] < c]
        row = {"j": x["j"], "r": x["r"], "g": x.get("g"), "dist": x.get("dist"), "b": b["b"], "c": c,
               "c_move": c - b["b"], "ws": w.get("ws"), "wd": w.get("wd"), "wv": w.get("wv"), "wt_w": w.get("wt"),
               **{k: b.get(k) for k in BOAT_COLS},
               "et": e.get("et") or np.nan, "exst": e.get("st"), "tilt": e.get("tilt"), "wadj": e.get("wadj"),
               "parts": e.get("parts"),
               "nwr_rank": ranks["nwr"][i], "nwr_diff": (b.get("nwr") or np.nan) - mean_nwr,
               "et_rank": ranks["et"][i], "et_diff": (e.get("et") or np.nan) - min_et,
               "exst_rank": ranks["exst"][i], "m2_rank": ranks["m2"][i], "ast_rank": ranks["ast"][i],
               "lwr_rank": ranks["lwr"][i], "a1_cnt": a1,
               "max_nwr_other": np.nanmax([nwr_f[k] for k in range(6) if k != i]),
               "cw_rate": h[1] / h[0] if h[0] >= 3 else np.nan, "c2_rate": h[2] / h[0] if h[0] >= 3 else np.nan,
               "c_starts": h[0],
               "inside_nwr": np.nanmax([nwr_f[k] for k in inside]) if inside else np.nan,
               "inside_exst": np.nanmin([exst_f[k] for k in inside]) if inside and any(not np.isnan(exst_f[k]) for k in inside) else np.nan,
               "nige_rate": srate[i]["nige"], "makurare_rate": srate[i]["makurare"], "mzasare_rate": srate[i]["mzasare"],
               "sasare_rate": srate[i]["sasare"], "in_starts": sty[i][0], "makuri_rate": srate[i]["makuri"],
               "mzashi_rate": srate[i]["mzashi"], "sashi_rate": srate[i]["sashi"], "att_starts": sty[i][5],
               "exst_vs_in": exst_f[i] - in_exst, "et_vs_in": et_f[i] - et_f[ii],
               "et_z": (et_arr[i] - np.nanmean(et_arr)) / et_sd if et_sd and not np.isnan(et_sd) and et_sd > 0 else np.nan,
               "et_self": et_self[i], "et_self_rank": et_self_rank[i], "et_self_n": selfs[i][1],
               "exst_vs_ast": exst_f[i] - (b.get("ast") if b.get("ast") is not None else np.nan),
               "exst_z": (ex_arr[i] - np.nanmean(ex_arr)) / ex_sd if ex_sd and not np.isnan(ex_sd) and ex_sd > 0 else np.nan,
               "et_gap2": (et_arr[i] - (srt[1] if et_arr[i] == srt[0] else srt[0])) if len(srt) >= 2 and not np.isnan(et_arr[i]) else np.nan,
               "ex_both_top": float(ranks["et"][i] <= 2 and ranks["exst"][i] <= 2) if not np.isnan(et_arr[i]) else np.nan,
               **race_in}
        res = b.get("res") or {}
        pl = res.get("pl")
        row.update(key=f'{x["d"]}_{x["j"]}_{x["r"]}', d=x["d"], num=b.get("num"),
                   y1=int(pl == 1), y2=int(pl in (1, 2)), y3=int(pl in (1, 2, 3)), pl=pl)
        rows.append(row)
    return rows


def in_diagnosis(rows: list[dict]) -> dict:
    """画面に出す「イン逃げ診断」の材料（インの艇の行から取り出す）。"""
    r = next((z for z in rows if z["c"] == 1), rows[0])
    f = lambda v, k=100, d=1: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v) * k, d)
    return {"b": int(r["b"]), "nige": f(r["nige_rate"]), "makurare": f(r["makurare_rate"]), "mzasare": f(r["mzasare_rate"]),
            "sasare": f(r["sasare_rate"]), "in_starts": int(r["in_starts"] or 0), "ast": f(r["ast"], 1, 2),
            "exst": f(r["exst"], 1, 2), "gap": f(r["out_best_exst_gap"], 1, 2), "m2": f(r["m2"], 1, 1),
            "st_in": f(r["st_in_rate"]), "st_wd_in": f(r["st_wd_in_rate"]), "st_ws_in": f(r["st_ws_in_rate"]),
            "out_makuri": f(r["out_max_makuri"]), "out_mzashi": f(r["out_max_mzashi"]),
            "ws": r.get("ws"), "wd": r.get("wd")}


def build_frame(races: list[dict], results_all: list[dict]) -> pd.DataFrame:
    hist = CourseHistory(results_all, races)
    rows = []
    for x in sorted(races, key=lambda z: (z["d"], z["j"], z["r"])):
        hist.advance(x["d"])
        rr = race_rows(x, hist)
        if rr:
            rows += rr
    return prep(pd.DataFrame(rows))


def prep(df: pd.DataFrame) -> pd.DataFrame:
    """数字の列は小数に、場コードと風向は「カテゴリ」にそろえる。"""
    for c in FEATURES:
        if c not in df:
            df[c] = np.nan
        if c not in CAT:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df["j"] = pd.to_numeric(df["j"]).astype(int).astype("category")
    df["wd"] = pd.to_numeric(df["wd"], errors="coerce").astype(float).astype("category")
    return df


def trifecta_probs(p1, p2, p3) -> np.ndarray:
    """各艇の 1着・2連対・3連対 の確率から、3連単120通りの確率を出す。

    1着は p1、2着の強さは (2連対−1着)、3着の強さは (3連対−2連対) として、
    「1着が決まったら残りの艇で2着を、さらに残りで3着を」の順に割り振る（ハービル方式の拡張）。
    p1, p2, p3 は (レース数, 6) の配列。戻り値は (レース数, 120)。
    """
    p1 = np.atleast_2d(np.asarray(p1, float)); p2 = np.atleast_2d(np.asarray(p2, float)); p3 = np.atleast_2d(np.asarray(p3, float))
    w1 = np.clip(p1, 1e-4, None)
    s2 = np.clip(p2 - p1, 1e-4, None)
    s3 = np.clip(p3 - p2, 1e-4, None)
    I, J, K = (np.array([p[t] for p in PERMS]) for t in range(3))
    out = (w1[:, I] / w1.sum(1, keepdims=True)
           * s2[:, J] / (s2.sum(1, keepdims=True) - s2[:, I])
           * s3[:, K] / (s3.sum(1, keepdims=True) - s3[:, I] - s3[:, J]))
    return out / out.sum(1, keepdims=True)


def combo_name(n: int) -> str:
    i, j, k = PERMS[n]
    return f"{i + 1}-{j + 1}-{k + 1}"


COMBO_INDEX = {combo_name(n): n for n in range(120)}

# ---- 推定オッズ（市場の人気）モデル用：組番ごとの材料
ODDS_FEATS = ["j", "r", "g", "a1_cnt", "spread"] + [f"{p}{k}" for k in (1, 2, 3)
                                                     for p in ("c", "cls", "nwr", "nwr_rank", "et_rank", "m2_rank", "exst_rank", "lwr_rank")]


COMBO_SRC = ("c", "cls", "nwr", "nwr_rank", "et_rank", "m2_rank", "exst_rank", "lwr_rank")


def race_matrix(df: pd.DataFrame):
    """6行ずつ並んだ表を (レース数, 6) の配列にする。df は key, b の順に並んでいること。"""
    n = len(df) // 6
    return {c: df[c].to_numpy(dtype=float).reshape(n, 6) for c in COMBO_SRC}, df.iloc[::6]


def combo_frame(df: pd.DataFrame, only: np.ndarray | None = None) -> pd.DataFrame:
    """レースごとに120通りの組番の材料を作る（only を渡すと各レース1通りだけ）。"""
    M, head = race_matrix(df)
    n = len(head)
    if only is None:
        ridx = np.repeat(np.arange(n), 120); pidx = np.tile(np.arange(120), n)
    else:
        ridx = np.arange(n); pidx = only
    P = np.array(PERMS)
    out = {"j": head["j"].to_numpy()[ridx], "r": head["r"].to_numpy()[ridx], "g": head["g"].to_numpy()[ridx],
           "a1_cnt": head["a1_cnt"].to_numpy()[ridx],
           "spread": (np.nanmax(M["nwr"], 1) - np.nanmin(M["nwr"], 1))[ridx]}
    for pos in range(3):
        boat = P[pidx, pos]
        for c in COMBO_SRC:
            out[f"{c}{pos + 1}"] = M[c][ridx, boat]
    f = pd.DataFrame(out)
    f["j"] = pd.Categorical(f["j"].astype(int), categories=[2, 12, 18, 19, 22, 24])
    return f
