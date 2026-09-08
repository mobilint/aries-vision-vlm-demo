"""Re-score saved result files with the current check_answer and print a
comparison table. Usage: python rescore.py [results-dir]"""

import json
import pathlib
import statistics
import sys

from harness import check_answer

CHECK_KEYS = ("coordinate_leak", "metadata_leak", "repetition", "wrong_language",
              "too_short", "too_long", "incomplete", "empty")


def rescore_file(path):
    data = json.loads(path.read_text())
    language = data.get("language", "en")
    ok = []
    for r in data["results"]:
        if r.get("error"):
            continue
        r["checks"] = check_answer(r["text"], language)
        ok.append(r)
    words = [r["checks"]["words"] for r in ok]
    failures = {k: sum(1 for r in ok if r["checks"][k]) for k in CHECK_KEYS}
    return {
        "label": data["label"],
        "model": (data.get("model") or "?").split("/")[-1],
        "category": data.get("category") or data.get("mode", "?"),
        "cases": len(data["results"]),
        "errors": len(data["results"]) - len(ok),
        "pass": sum(1 for r in ok if not any(r["checks"][k] for k in CHECK_KEYS)),
        "words_median": statistics.median(words) if words else 0,
        "words_min": min(words, default=0),
        "words_max": max(words, default=0),
        "decode_s_median": round(statistics.median(r["decode_s"] for r in ok), 2) if ok else 0,
        "failures": failures,
    }


def main():
    results_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                               pathlib.Path(__file__).parent / "results")
    rows = [rescore_file(p) for p in sorted(results_dir.glob("*.json"))]
    rows.sort(key=lambda r: (r["category"], r["label"]))
    for row in rows:
        flags = ", ".join(f"{k}:{v}" for k, v in row["failures"].items() if v) or "-"
        print(f"{row['label']:>18} {row['model']:>22} {row['category']:8} pass {row['pass']:2}/{row['cases']}"
              f" errs {row['errors']} words {row['words_min']}-{row['words_max']}"
              f" (med {row['words_median']}) decode_med {row['decode_s_median']}s | {flags}")


if __name__ == "__main__":
    main()
