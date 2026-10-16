"""登録番号で引ける選手データ site/data/racers.json を作る。

中身:
  - 最新の出走表にのった成績（級別・勝率・2連率・3連率・平均ST・F/L）
  - 直近1年・全24場でのコース別成績（出走数・1着・2着・3着・平均ST）
  - 直近2年・対象6場での当地成績
  - 直近10走
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict

from common import (ALL_STADIUM_NAMES, BRANCH_NAMES, CLASS_NAMES, DATA, SITE_DATA, STADIUMS, read_store,
                    write_json)


def main():
    master = json.loads((DATA / "racer_master.json").read_text())
    res_all = read_store("results_all")
    last = max(x["d"] for x in res_all)
    end = dt.date.fromisoformat(last)
    y1 = (end - dt.timedelta(days=365)).isoformat()
    y2 = (end - dt.timedelta(days=730)).isoformat()

    course = defaultdict(lambda: [[0, 0, 0, 0, 0.0, 0] for _ in range(6)])  # 出走,1着,2着,3着,ST合計,ST数
    local = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))         # 場 → 出走,1着,2連,3連
    recent = defaultdict(list)
    for x in sorted(res_all, key=lambda z: (z["d"], z["r"])):
        for b, num, c, st, pl in x["boats"]:
            if not num:
                continue
            if pl is None or pl == 16:  # 欠場は出走に数えない
                continue
            if x["d"] >= y1 and c:
                k = course[num][c - 1]
                k[0] += 1
                if pl == 1: k[1] += 1
                if pl == 2: k[2] += 1
                if pl == 3: k[3] += 1
                if st is not None and pl != 14:
                    k[4] += st; k[5] += 1
            if x["d"] >= y2 and x["j"] in STADIUMS:
                k = local[num][x["j"]]
                k[0] += 1
                if pl == 1: k[1] += 1
                if pl in (1, 2): k[2] += 1
                if pl in (1, 2, 3): k[3] += 1
            recent[num].append([x["d"], x["j"], x["r"], b, c, st, pl])

    out = {}
    for num_s, m in master.items():
        num = int(num_s)
        cs = course.get(num)
        out[num_s] = {
            "name": m.get("name"), "cls": CLASS_NAMES.get(m.get("cls"), "-"), "br": BRANCH_NAMES.get(m.get("br"), "-"),
            "age": m.get("age"), "wt": m.get("wt"), "f": m.get("f"), "l": m.get("l"), "ast": m.get("ast"),
            "wr": m.get("nwr"), "t2": m.get("n2"), "t3": m.get("n3"), "asof": m["d"],
            "course": [[k[0], k[1], k[2], k[3], round(k[4] / k[5], 2) if k[5] else None] for k in cs] if cs else None,
            "local": {str(j): v for j, v in local.get(num, {}).items()},
            "recent": list(reversed(recent.get(num, [])[-10:])),
        }
    write_json(SITE_DATA / "racers.json", {"to": last, "racers": out,
                                            "stadium_names": {str(k): v for k, v in ALL_STADIUM_NAMES.items()}})
    print(f"選手 {len(out)}人（成績集計 〜{last}）")


if __name__ == "__main__":
    main()
