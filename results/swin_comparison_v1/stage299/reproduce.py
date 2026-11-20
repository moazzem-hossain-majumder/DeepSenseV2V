"""Run with Python 3 and NumPy. Reads aggregate validation statistics only."""
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def main():
    with (HERE / "block_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    saved = json.loads((HERE / "contrasts.json").read_text())
    verified = 0
    for metric in ("hit", "success", "regret", "top1_loss_db", "alternative"):
        factor = 100 if metric in ("hit", "success", "alternative") else 1
        groups = {}
        for r in rows:
            key = (int(r["scenario"]), r["block_key"])
            parts = groups.setdefault(key, {})
            cell = (r["split"], r["arm"], int(r["seed"]))
            assert cell not in parts
            parts[cell] = (float(r[metric]), int(r["n"]))
        for split, comp in [("pooled","gps"), ("transfer","gps"), ("pooled","geometry"),
                            ("transfer","geometry"), ("pooled_minus_transfer","matched_gps_gain")]:
            rng = np.random.default_rng(218)
            by_scenario, draws = [], np.zeros(4000)
            for scenario in (36, 37, 38, 39):
                sums, counts = [], []
                for key in sorted(groups):
                    if key[0] != scenario:
                        continue
                    cells = groups[key]
                    assert len(cells) == 18 and len({n for _,n in cells.values()}) == 1
                    def total(s, arm):
                        return np.mean([cells[s,arm,seed][0] for seed in (42,7,13)])
                    if split == "pooled_minus_transfer":
                        diff = total("pooled","rgb") - total("pooled","gps") - total("transfer","rgb") + total("transfer","gps")
                    else:
                        diff = total(split,"rgb") - total(split,comp)
                    sums.append(diff)
                    counts.append(next(iter(cells.values()))[1])
                sums, counts = np.array(sums), np.array(counts)
                by_scenario.append(factor*sums.sum()/counts.sum())
                ix = rng.integers(len(sums),size=(4000,len(sums)))
                draws += factor*sums[ix].sum(1)/counts[ix].sum(1)/4
            point = np.mean(by_scenario)
            interval = np.quantile(draws,[.025,.975])
            if metric == "alternative":
                if comp == "gps" and split == "transfer":
                    print(f"Alternative success, RGB minus GPS: {point:.3f} points, 95% [{interval[0]:.3f}, {interval[1]:.3f}]")
                    print("Per scenario:", [round(x,3) for x in by_scenario])
            else:
                record = next(r for r in saved if (r["metric"],r["split"],r["comparator"])==(metric,split,comp))
                np.testing.assert_allclose([point,*interval],[record["difference"],*record["interval"]],atol=.00001)
                verified += 1
    assert verified == len(saved) == 20
    print("Reproduced all 20 full-codebook contrasts and paired intervals.")

if __name__ == "__main__":
    main()
