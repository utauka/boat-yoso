"""共通部品：場の設定、データ形式のそろえ方、保存と読み込み。

データ元は非公式の Boatrace Open API（MIT ライセンス）です。
  - 2026年1月1日以降 : https://boatraceopenapi.github.io/api/v1/YYYY/YYYYMMDD.json
  - 2025年12月31日まで: 旧版 results / programs / previews（v2）
どちらの形式も、このファイルの normalize_* 関数で同じ形にそろえます。
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"            # 集めた生データ（月ごとの圧縮ファイル）
SITE_DATA = ROOT / "site" / "data"  # サイトが読む集計済みデータ
MODELS = ROOT / "models"

JST = dt.timezone(dt.timedelta(hours=9))

# 対象6場（場コード : 名前）
STADIUMS = {2: "戸田", 12: "住之江", 18: "徳山", 19: "下関", 22: "福岡", 24: "大村"}
ALL_STADIUM_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖", 7: "蒲郡", 8: "常滑",
    9: "津", 10: "三国", 11: "びわこ", 12: "住之江", 13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島",
    17: "宮島", 18: "徳山", 19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}
CLASS_NAMES = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}
BRANCH_NAMES = {
    1: "北海道", 2: "青森", 3: "岩手", 4: "宮城", 5: "秋田", 6: "山形", 7: "福島", 8: "茨城", 9: "栃木",
    10: "群馬", 11: "埼玉", 12: "千葉", 13: "東京", 14: "神奈川", 15: "新潟", 16: "富山", 17: "石川",
    18: "福井", 19: "山梨", 20: "長野", 21: "岐阜", 22: "静岡", 23: "愛知", 24: "三重", 25: "滋賀",
    26: "京都", 27: "大阪", 28: "兵庫", 29: "奈良", 30: "和歌山", 31: "鳥取", 32: "島根", 33: "岡山",
    34: "広島", 35: "山口", 36: "徳島", 37: "香川", 38: "愛媛", 39: "高知", 40: "福岡", 41: "佐賀",
    42: "長崎", 43: "熊本", 44: "大分", 45: "宮崎", 46: "鹿児島", 47: "沖縄",
}
TECHNIQUES = {1: "逃げ", 2: "差し", 3: "まくり", 4: "まくり差し", 5: "抜き", 6: "恵まれ"}

# 本命・中穴・大穴の区切り（倍率）
HONMEI_MAX = 20.0   # 20倍以下 → 本命
CHUANA_MAX = 60.0   # 60倍以下 → 中穴、それより上 → 大穴

V1_URL = "https://boatraceopenapi.github.io/api/v1/{y}/{ymd}.json"
V2_URL = "https://boatraceopenapi.github.io/{kind}/v2/{y}/{ymd}.json"
V1_START = dt.date(2026, 1, 1)


def payout_class(amount: int | None) -> str | None:
    """払戻金（100円あたり）から 本命/中穴/大穴 を決める。"""
    if not amount:
        return None
    odds = amount / 100
    if odds <= HONMEI_MAX:
        return "honmei"
    if odds <= CHUANA_MAX:
        return "chuana"
    return "oana"


# ------------------------------------------------------------------ 取得
def http_json(url: str, retries: int = 3, wait: float = 2.0):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kyotei-yoso-site/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # 通信エラーは少し待ってやり直す
            last = e
        time.sleep(wait * (i + 1))
    raise RuntimeError(f"取得に失敗しました: {url} ({last})")


def load_day_raw(day: dt.date, local_root: str | None = None):
    """1日分の生データを返す。local_root を渡すと手元の複製から読む（初回の一括取得用）。

    戻り値: ("v1", json) または ("v2", {"results":…, "programs":…, "previews":…}) / None
    """
    y, ymd = day.strftime("%Y"), day.strftime("%Y%m%d")
    if day >= V1_START:
        if local_root:
            p = Path(local_root) / "api" / "docs" / "v1" / y / f"{ymd}.json"
            js = json.loads(p.read_text()) if p.exists() else None
        else:
            js = http_json(V1_URL.format(y=y, ymd=ymd))
        return ("v1", js) if js else None
    out = {}
    for kind in ("results", "programs", "previews"):
        if local_root:
            p = Path(local_root) / kind / "docs" / "v2" / y / f"{ymd}.json"
            out[kind] = json.loads(p.read_text())[kind] if p.exists() else []
        else:
            js = http_json(V2_URL.format(kind=kind, y=y, ymd=ymd))
            out[kind] = js[kind] if js else []
            time.sleep(0.5)
    return ("v2", out) if any(out.values()) else None


# ------------------------------------------------------------------ 形をそろえる
def _num(x):
    return None if x is None or x == "" else x


def _boats(p):
    bs = p.get("boats") or []
    return list(bs.values()) if isinstance(bs, dict) else bs


def _pay_v(lst, key):
    return [[p.get("combination"), p.get(key)] for p in (lst or []) if p.get(key)]


def normalize_v2(raw: dict) -> list[dict]:
    """旧版（2025年まで）の3ファイルを、レース単位の辞書にまとめる。"""
    races: dict[tuple, dict] = {}

    def get(d, j, r):
        k = (d, j, r)
        if k not in races:
            races[k] = {"d": d, "j": j, "r": r, "boats": {b: {"b": b} for b in range(1, 7)}, "pay": {}}
        return races[k]

    for p in raw.get("programs", []):
        x = get(p["race_date"], p["race_stadium_number"], p["race_number"])
        x.update(ca=p.get("race_closed_at"), g=p.get("race_grade_number"), title=p.get("race_title"),
                 sub=p.get("race_subtitle"), dist=p.get("race_distance"))
        for b in _boats(p):
            bb = x["boats"].setdefault(b["racer_boat_number"], {"b": b["racer_boat_number"]})
            bb.update(num=b.get("racer_number"), name=b.get("racer_name"), cls=b.get("racer_class_number"),
                      br=b.get("racer_branch_number"), age=b.get("racer_age"), wt=b.get("racer_weight"),
                      f=b.get("racer_flying_count"), l=b.get("racer_late_count"),
                      ast=_num(b.get("racer_average_start_timing")),
                      nwr=b.get("racer_national_top_1_percent"), n2=b.get("racer_national_top_2_percent"),
                      n3=b.get("racer_national_top_3_percent"), lwr=b.get("racer_local_top_1_percent"),
                      l2=b.get("racer_local_top_2_percent"), l3=b.get("racer_local_top_3_percent"),
                      mno=b.get("racer_assigned_motor_number"), m2=b.get("racer_assigned_motor_top_2_percent"),
                      m3=b.get("racer_assigned_motor_top_3_percent"), bno=b.get("racer_assigned_boat_number"),
                      b2=b.get("racer_assigned_boat_top_2_percent"), b3=b.get("racer_assigned_boat_top_3_percent"))
    for p in raw.get("previews", []):
        x = get(p["race_date"], p["race_stadium_number"], p["race_number"])
        x["pw"] = {"ws": p.get("race_wind"), "wd": p.get("race_wind_direction_number"), "wv": p.get("race_wave"),
                   "wt": p.get("race_weather_number"), "at": p.get("race_temperature"),
                   "wat": p.get("race_water_temperature")}
        for b in _boats(p):
            bb = x["boats"].setdefault(b["racer_boat_number"], {"b": b["racer_boat_number"]})
            bb["ex"] = {"c": b.get("racer_course_number"), "st": b.get("racer_start_timing"),
                        "et": b.get("racer_exhibition_time"), "tilt": b.get("racer_tilt_adjustment"),
                        "wadj": b.get("racer_weight_adjustment")}
    for p in raw.get("results", []):
        x = get(p["race_date"], p["race_stadium_number"], p["race_number"])
        x["rw"] = {"ws": p.get("race_wind"), "wd": p.get("race_wind_direction_number"), "wv": p.get("race_wave"),
                   "wt": p.get("race_weather_number")}
        x["tech"] = p.get("race_technique_number")
        for b in _boats(p):
            bb = x["boats"].setdefault(b["racer_boat_number"], {"b": b["racer_boat_number"]})
            bb["res"] = {"c": b.get("racer_course_number"), "st": b.get("racer_start_timing"),
                         "pl": b.get("racer_place_number")}
            bb.setdefault("num", b.get("racer_number"))
            bb.setdefault("name", b.get("racer_name"))
        x["pay"] = {k: _pay_v(v, "payout") for k, v in (p.get("payouts") or {}).items()}
    return [_finish(x) for x in races.values()]


def normalize_v1(js: dict) -> list[dict]:
    """新版（2026年から）の1ファイルを、レース単位の辞書にまとめる。"""
    out = []
    for jkey, st in (js.get("programs", {}).get("stadiums") or {}).items():
        for rkey, r in (st.get("races") or {}).items():
            x = {"d": r.get("date"), "j": r.get("stadium_number") or int(jkey), "r": r.get("race_number") or int(rkey),
                 "ca": r.get("closed_at"), "g": r.get("grade_number"), "dn": r.get("day_number"),
                 "title": r.get("title"), "sub": r.get("subtitle"), "dist": r.get("distance"),
                 "boats": {}, "pay": {}}
            for b in (r.get("racers") or {}).values():
                e = b.get("entry_number")
                x["boats"][e] = {"b": e, "num": b.get("number"), "name": b.get("name"), "cls": b.get("rank_number"),
                                 "br": b.get("branch_number"), "age": b.get("age"), "wt": b.get("weight"),
                                 "f": b.get("flying_count"), "l": b.get("late_count"),
                                 "ast": _num(b.get("average_start_timing")), "nwr": b.get("national_win_rate"),
                                 "n2": b.get("national_top_2_percent"), "n3": b.get("national_top_3_percent"),
                                 "lwr": b.get("local_win_rate"), "l2": b.get("local_top_2_percent"),
                                 "l3": b.get("local_top_3_percent"), "mno": b.get("motor_number"),
                                 "m2": b.get("motor_top_2_percent"), "m3": b.get("motor_top_3_percent"),
                                 "bno": b.get("boat_number"), "b2": b.get("boat_top_2_percent"),
                                 "b3": b.get("boat_top_3_percent")}
            pv = r.get("preview") or {}
            if pv.get("racers"):
                x["pw"] = {"ws": pv.get("wind_speed"), "wd": pv.get("wind_direction_number"), "wv": pv.get("wave_height"),
                           "wt": pv.get("weather_number"), "at": pv.get("air_temperature"),
                           "wat": pv.get("water_temperature")}
                for b in pv["racers"].values():
                    e = b.get("entry_number")
                    x["boats"].setdefault(e, {"b": e})["ex"] = {
                        "c": b.get("course_number"), "st": b.get("start_timing"), "et": b.get("exhibition_time"),
                        "tilt": b.get("tilt_adjustment"), "wadj": b.get("weight_adjustment"),
                        "parts": len(b.get("parts") or []), "prop": 1 if b.get("propeller") else 0}
            rs = r.get("result") or {}
            if rs.get("racers"):
                x["rw"] = {"ws": rs.get("wind_speed"), "wd": rs.get("wind_direction_number"), "wv": rs.get("wave_height"),
                           "wt": rs.get("weather_number")}
                x["tech"] = rs.get("technique_number")
                for b in rs["racers"].values():
                    e = b.get("entry_number")
                    x["boats"].setdefault(e, {"b": e})["res"] = {"c": b.get("course_number"), "st": b.get("start_timing"),
                                                                 "pl": b.get("place_number")}
                x["pay"] = {k: _pay_v(v, "amount") for k, v in (rs.get("payouts") or {}).items()}
            out.append(_finish(x))
    return out


def _finish(x: dict) -> dict:
    x["boats"] = [x["boats"][k] for k in sorted(x["boats"]) if k is not None]
    return x


def normalize(kind_js) -> list[dict]:
    if not kind_js:
        return []
    kind, js = kind_js
    return normalize_v1(js) if kind == "v1" else normalize_v2(js)


# ------------------------------------------------------------------ 保存
def month_path(folder: str, ymd: str) -> Path:
    return DATA / folder / f"{ymd[:7]}.jsonl.gz"


def read_store(folder: str) -> list[dict]:
    out = []
    for p in sorted((DATA / folder).glob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


def write_month(folder: str, month: str, rows: list[dict]):
    p = DATA / folder / f"{month}.jsonl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: (x["d"], x["j"], x["r"]))
    with gzip.open(p, "wt", encoding="utf-8", compresslevel=9) as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n")


def upsert(folder: str, rows: list[dict]):
    """同じレース（日付・場・R）は新しい内容で上書きして保存する。"""
    by_month: dict[str, list] = {}
    for x in rows:
        by_month.setdefault(x["d"][:7], []).append(x)
    for month, new in by_month.items():
        p = DATA / folder / f"{month}.jsonl.gz"
        cur = {}
        if p.exists():
            with gzip.open(p, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        y = json.loads(line)
                        cur[(y["d"], y["j"], y["r"])] = y
        for y in new:
            cur[(y["d"], y["j"], y["r"])] = y
        write_month(folder, month, list(cur.values()))


def slim_result(x: dict) -> dict | None:
    """全24場の結果を軽く保存する形（選手のコース別成績づくり用）。"""
    boats = [[b["b"], b.get("num"), b["res"].get("c"), b["res"].get("st"), b["res"].get("pl")]
             for b in x["boats"] if b.get("res")]
    if not boats:
        return None
    tri = (x.get("pay") or {}).get("trifecta") or []
    return {"d": x["d"], "j": x["j"], "r": x["r"], "boats": boats, "tri": tri[:1]}


def write_json(path: Path, obj, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, ensure_ascii=False, indent=1 if pretty else None,
                     separators=None if pretty else (",", ":"))
    path.write_text(txt, encoding="utf-8")


def today_jst() -> dt.date:
    return dt.datetime.now(JST).date()


def daterange(a: dt.date, b: dt.date):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def env_local_root():
    return os.environ.get("BOAT_LOCAL_ROOT")
