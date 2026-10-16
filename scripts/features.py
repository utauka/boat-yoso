"""予想に使う「特徴量（AIに渡す材料）」を作る。学習と当日予想の両方で使う。"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict, deque
from itertools import permutations

import numpy as np
import pandas as pd

BOAT_COLS = ["cls", "age", "wt", "f", "l", "ast", "nwr", "n2", "n3", "lwr", "l2", "l3", "m2", "m3", "b2", "b3"]
FEATURES = (
    ["j", "r", "g", "dist", "b", "c", "c_move", "ws", "wd", "wv", "wt_w"]
    + BOAT_COLS
    + ["et", "exst", "tilt", "wadj", "parts",
       "nwr_rank", "nwr_diff", "et_rank", "et_diff", "exst_rank", "m2_rank", "ast_rank", "lwr_rank",
       "a1_cnt", "max_nwr_other", "cw_rate", "c2_rate", "c_starts", "inside_nwr", "inside_exst"]
)
CAT = ["j", "wd"]
PERMS = list(permutations(range(6), 3))  # 120通り（0始まりの艇の位置）


class CourseHistory:
    """選手ごとの「コース別1着率・2連率」を、その日より前の1年分だけで数える（未来の情報を使わない）。"""

    def __init__(self, results_all: list[dict], days: int = 365):
        self.events = sorted(
            [(x["d"], num, c, pl) for x in results_all for _, num, c, _, pl in x["boats"]
             if num and c and pl is not None and pl != 16],
            key=lambda e: e[0])
        self.days = days
        self.i_add = 0
        self.i_del = 0
        self.cnt = defaultdict(lambda: [0, 0, 0])

    def advance(self, day: str):
        """day の前日までの成績が数えられた状態にする。"""
        lo = (dt.date.fromisoformat(day) - dt.timedelta(days=self.days)).isoformat()
        ev = self.events
        while self.i_add < len(ev) and ev[self.i_add][0] < day:
            _, num, c, pl = ev[self.i_add]
            k = self.cnt[(num, c)]
            k[0] += 1; k[1] += pl == 1; k[2] += pl in (1, 2)
            self.i_add += 1
        while self.i_del < self.i_add and ev[self.i_del][0] < lo:
            _, num, c, pl = ev[self.i_del]
            k = self.cnt[(num, c)]
            k[0] -= 1; k[1] -= pl == 1; k[2] -= pl in (1, 2)
            self.i_del += 1

    def get(self, num, c):
        k = self.cnt.get((num, c))
        return k if k else [0, 0, 0]


def _rank(vals, ascending=False):
    s = pd.Series(vals, dtype="float")
    return s.rank(ascending=ascending, method="min").tolist()


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
    courses = [e.get("c") or b["b"] for e, b in zip(exs, boats)]
    ranks = {"nwr": _rank(nwr_f), "et": _rank(et_f, True), "exst": _rank([v if v is not None else np.nan for v in exst], True),
             "m2": _rank([v if v is not None else np.nan for v in get("m2")]),
             "ast": _rank([v if v is not None else np.nan for v in get("ast")], True),
             "lwr": _rank([v if v is not None else np.nan for v in get("lwr")])}
    mean_nwr = np.nanmean(nwr_f)
    min_et = np.nanmin(et_f) if not all(np.isnan(et_f)) else np.nan
    a1 = sum(1 for b in boats if b.get("cls") == 1)
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
               "inside_exst": np.nanmin([exst[k] if exst[k] is not None else np.nan for k in inside]) if inside else np.nan}
        res = b.get("res") or {}
        pl = res.get("pl")
        row.update(key=f'{x["d"]}_{x["j"]}_{x["r"]}', d=x["d"], num=b.get("num"),
                   y1=int(pl == 1), y2=int(pl in (1, 2)), y3=int(pl in (1, 2, 3)), pl=pl)
        rows.append(row)
    return rows


def build_frame(races: list[dict], results_all: list[dict]) -> pd.DataFrame:
    hist = CourseHistory(results_all)
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
