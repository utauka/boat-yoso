"""場ごとの出目履歴を集計して site/data/stats.json を作る。

区分（3連単の倍率＝払戻金÷100）:
  本命 20倍以下 / 中穴 20倍より上〜60倍以下 / 大穴 60倍より上
期間: 結果がそろっている最新日から数えて直近2年。
"""
from __future__ import annotations

import datetime as dt
import statistics
from collections import Counter, defaultdict
from itertools import permutations

from common import SITE_DATA, STADIUMS, TECHNIQUES, payout_class, read_store, write_json

BETS = {"trifecta": "3連単", "exacta": "2連単", "trio": "3連複"}
ALL_TRI = ["-".join(map(str, p)) for p in permutations(range(1, 7), 3)]


def main(years: int = 2):
    races = read_store("races6")
    done = [x for x in races if (x.get("pay") or {}).get("trifecta")]
    last = max(x["d"] for x in done)
    end = dt.date.fromisoformat(last)
    start = end.replace(year=end.year - years) + dt.timedelta(days=1)
    s0, s1 = start.isoformat(), end.isoformat()
    win = [x for x in races if s0 <= x["d"] <= s1 and x.get("boats") and any(b.get("res") for b in x["boats"])]

    out = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "from": s0, "to": s1,
           "bounds": {"honmei": 20, "chuana": 60}, "stadiums": {}, "history": {}}
    for j, name in STADIUMS.items():
        rs = sorted([x for x in win if x["j"] == j], key=lambda x: (x["d"], x["r"]))
        st = {"name": name, "races": len(rs)}
        # 券種ごとの区分
        for key, label in BETS.items():
            cnt, pays = Counter(), []
            for x in rs:
                p = (x["pay"].get(key) or [])
                if not p:
                    cnt["void"] += 1
                    continue
                amt = p[0][1]
                if amt < 100:      # 特払い（70円）などは別扱い
                    cnt["special"] += 1
                    continue
                cnt[payout_class(amt)] += 1
                pays.append(amt)
            st[key] = {"label": label, "honmei": cnt["honmei"], "chuana": cnt["chuana"], "oana": cnt["oana"],
                       "void": cnt["void"] + cnt["special"], "avg": round(statistics.mean(pays)) if pays else 0,
                       "median": round(statistics.median(pays)) if pays else 0}
        # 3連単の出目ランキング（120通りすべて）
        combo = defaultdict(list)
        for x in rs:
            p = x["pay"].get("trifecta") or []
            if p and p[0][1] >= 100:
                combo[p[0][0]].append(p[0][1])
        valid = sum(len(v) for v in combo.values())
        st["combos"] = sorted(
            [{"c": c, "n": len(combo.get(c, [])), "avg": round(statistics.mean(combo[c])) if combo.get(c) else 0,
              "min": min(combo[c]) if combo.get(c) else 0, "max": max(combo[c]) if combo.get(c) else 0,
              "cls": Counter(payout_class(a) for a in combo.get(c, []))} for c in ALL_TRI],
            key=lambda z: -z["n"])
        for z in st["combos"]:
            z["cls"] = [z["cls"].get("honmei", 0), z["cls"].get("chuana", 0), z["cls"].get("oana", 0)]
        st["valid"] = valid
        # レース番号別・月別
        by_r, by_m = defaultdict(Counter), defaultdict(Counter)
        for x in rs:
            p = x["pay"].get("trifecta") or []
            if p and p[0][1] >= 100:
                k = payout_class(p[0][1])
                by_r[x["r"]][k] += 1
                by_m[x["d"][:7]][k] += 1
        pick = lambda c: [c.get("honmei", 0), c.get("chuana", 0), c.get("oana", 0)]
        st["by_race"] = {r: pick(by_r[r]) for r in sorted(by_r)}
        st["by_month"] = {m: pick(by_m[m]) for m in sorted(by_m)}
        # 1着になった進入コース・決まり手
        cw, tech = Counter(), Counter()
        for x in rs:
            for b in x["boats"]:
                res = b.get("res") or {}
                if res.get("pl") == 1 and res.get("c"):
                    cw[res["c"]] += 1
            if x.get("tech"):
                tech[TECHNIQUES.get(x["tech"], "その他")] += 1
        tot = sum(cw.values()) or 1
        st["course_win"] = [round(cw[c] / tot * 100, 1) for c in range(1, 7)]
        st["tech"] = dict(tech.most_common())
        out["stadiums"][j] = st
        # 日ごとの全レース結果（画面の一覧用）[日付, R, 3連単の組番, 払戻金]
        out["history"][j] = [[x["d"], x["r"], (x["pay"].get("trifecta") or [[None, 0]])[0][0],
                              (x["pay"].get("trifecta") or [[None, 0]])[0][1]] for x in rs]
    write_json(SITE_DATA / "stats.json", out)
    for j, st in out["stadiums"].items():
        t = st["trifecta"]; v = t["honmei"] + t["chuana"] + t["oana"]
        print(f'{st["name"]}: {st["races"]}R 本命{t["honmei"]/v:.1%} 中穴{t["chuana"]/v:.1%} 大穴{t["oana"]/v:.1%} '
              f'平均{t["avg"]}円 1コース{st["course_win"][0]}%')
    print(f"期間 {s0}〜{s1}")


if __name__ == "__main__":
    main()
