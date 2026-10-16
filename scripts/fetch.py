"""データを集めて data/ に貯める。

使い方:
  python scripts/fetch.py                       # 昨日と今日の分を取得（毎日の自動更新用）
  python scripts/fetch.py --from 2024-10-01     # 指定日から今日まで（初回の一括取得）
環境変数 BOAT_LOCAL_ROOT を指定すると、ネットではなく手元の複製から読みます。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time

from common import (DATA, STADIUMS, daterange, env_local_root, load_day_raw, normalize, slim_result,
                    today_jst, upsert)

MASTER = DATA / "racer_master.json"
KEEP_FIELDS = ("name", "cls", "br", "age", "wt", "f", "l", "ast", "nwr", "n2", "n3")


def update_master(master: dict, races: list[dict]):
    for x in sorted(races, key=lambda r: (r["d"], r.get("ca") or "")):
        for b in x["boats"]:
            num = b.get("num")
            if not num or b.get("nwr") is None:
                continue
            m = master.get(str(num))
            if m and m["d"] > x["d"]:
                continue
            master[str(num)] = {"d": x["d"], **{k: b.get(k) for k in KEEP_FIELDS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start")
    ap.add_argument("--to", dest="end")
    a = ap.parse_args()
    today = today_jst()
    start = dt.date.fromisoformat(a.start) if a.start else today - dt.timedelta(days=1)
    end = dt.date.fromisoformat(a.end) if a.end else today
    local = env_local_root()

    master = json.loads(MASTER.read_text()) if MASTER.exists() else {}
    buf6, buf_all, n = [], [], 0
    for day in daterange(start, end):
        races = normalize(load_day_raw(day, local))
        if not local:
            time.sleep(1.0)  # 相手のサーバーに負担をかけない
        if not races:
            continue
        update_master(master, races)
        buf6 += [x for x in races if x["j"] in STADIUMS]
        buf_all += [s for s in (slim_result(x) for x in races) if s]
        n += 1
        if len(buf6) > 3000:
            upsert("races6", buf6), upsert("results_all", buf_all)
            buf6, buf_all = [], []
            print(f"…{day} まで保存")
    upsert("races6", buf6)
    upsert("results_all", buf_all)
    MASTER.parent.mkdir(parents=True, exist_ok=True)
    MASTER.write_text(json.dumps(master, ensure_ascii=False, separators=(",", ":")))
    print(f"完了: {n}日分を取得（{start}〜{end}）／選手 {len(master)}人")


if __name__ == "__main__":
    main()
