"""Run captured detection cases against the live VLM server and score answers.

Each VLM process serves one category (weapon on :5000, fall on :5001), so
--category selects both the case set and the target port.

Usage:
    python run_eval.py --category weapon --label baseline
    python run_eval.py --category fall --bundle candidates/v1 --label v1

The bundle dir must contain system.txt (and optionally inter.txt).
Defaults to the live frontend bundle for the category. Results go to
results/<label>-<category>.json and a summary is printed.
"""

import argparse
import json
import pathlib
import statistics

import requests

from harness import (
    VlmClient,
    annotate_image_data_url,
    build_detection_prompt,
    check_answer,
    eligible_detections,
)

CATEGORY_PORTS = {"weapon": 5000, "fall": 5001}

ROOT = pathlib.Path(__file__).parent
REPO = ROOT.parent.parent


def ensure_model(url, model_id, timeout=900):
    """Switch the VLM server to model_id if it is not already active.
    Model loads on the NPU can take minutes, hence the long timeout."""
    state = requests.get(f"{url}/models", timeout=10).json()
    if state["model_id"] == model_id:
        return state
    print(f"switching VLM model {state['model_id']} -> {model_id} (may take minutes)...")
    response = requests.post(f"{url}/model", json={"model_id": model_id}, timeout=timeout)
    response.raise_for_status()
    state = response.json()
    if state.get("ok") is False:
        raise RuntimeError(f"model switch failed: {state.get('message')}")
    print(f"active model: {state['model_id']} (runtime {state.get('runtime_model_id')})")
    return state


def load_bundle(bundle_dir):
    system = (bundle_dir / "system.txt").read_text().strip()
    inter_path = bundle_dir / "inter.txt"
    inter = inter_path.read_text().strip() if inter_path.exists() else ""
    return system, inter


def run_case(client, case, category, language="en"):
    channel = case["channel"]
    marked = eligible_detections(channel["detections"], category, case["threshold"])
    prompt = build_detection_prompt(channel, case["trigger"], marked, category)
    image = annotate_image_data_url(
        channel["image_base64"], marked, channel["image_width"], channel["image_height"],
    )
    result = client.ask(prompt, image)
    result["checks"] = check_answer(result["text"], language)
    return result


def summarize(results):
    ok = [r for r in results if not r.get("error")]
    checks = [r["checks"] for r in ok]
    words = [c["words"] for c in checks]
    failures = {
        key: sum(1 for c in checks if c[key])
        for key in ("coordinate_leak", "metadata_leak", "repetition", "wrong_language",
                    "too_short", "too_long", "incomplete", "empty")
    }
    passed = sum(1 for c in checks if not any(c[k] for k in failures))
    return {
        "cases": len(results),
        "errors": len(results) - len(ok),
        "pass": passed,
        "words_median": statistics.median(words) if words else 0,
        "words_min": min(words, default=0),
        "words_max": max(words, default=0),
        "decode_s_median": statistics.median(r["decode_s"] for r in ok) if ok else 0,
        "total_s_median": statistics.median(r["total_s"] for r in ok) if ok else 0,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["weapon", "fall"], required=True)
    parser.add_argument("--bundle", default=None,
                        help="dir with system.txt/inter.txt (default: live frontend bundle, en)")
    parser.add_argument("--cases", default=None, help="cases dir (default: cases/<category>)")
    parser.add_argument("--label", required=True, help="run label used in the result filename")
    parser.add_argument("--vlm-host", default="localhost")
    parser.add_argument("--vlm-port", type=int, default=None,
                        help="override the category-derived port (weapon=5000, fall=5001)")
    parser.add_argument("--model", default=None,
                        help="frontend model id (e.g. Qwen/Qwen3-VL-2B-Instruct); switches the server if needed")
    parser.add_argument("--language", choices=["en", "ko", "ja", "zh"], default=None,
                        help="answer language for the checks (default: inferred from the bundle path, else en)")
    args = parser.parse_args()

    port = args.vlm_port if args.vlm_port is not None else CATEGORY_PORTS[args.category]
    url = f"http://{args.vlm_host}:{port}"

    model_state = ensure_model(url, args.model) if args.model else (
        requests.get(f"{url}/models", timeout=10).json()
    )

    # The shipped prompt-bundle directories still use the "<category>_detection"
    # naming; only the eval-internal identifiers migrated to the bare category.
    bundle_dir = pathlib.Path(args.bundle) if args.bundle else (
        REPO / "frontend" / "public" / "prompt-bundles" / f"{args.category}_detection" / (args.language or "en")
    )
    language = args.language or (bundle_dir.name if bundle_dir.name in ("en", "ko", "ja", "zh") else "en")
    cases_dir = pathlib.Path(args.cases) if args.cases else ROOT / "cases" / args.category
    case_paths = sorted(cases_dir.glob("case_*.json"))
    if not case_paths:
        raise SystemExit(f"no cases found in {cases_dir}; run capture_cases.py first")

    system_prompt, inter_prompt = load_bundle(bundle_dir)
    client = VlmClient(url)
    client.set_prompts(system_prompt, inter_prompt)

    results = []
    try:
        for path in case_paths:
            case = json.loads(path.read_text())
            try:
                result = run_case(client, case, args.category, language)
            except Exception as error:
                result = {"error": str(error)}
            result["case"] = path.name
            results.append(result)
            if result.get("error"):
                print(f"{path.name}: ERROR {result['error']}")
            else:
                c = result["checks"]
                flags = ",".join(k for k in c if k not in ("words", "length") and c[k]) or "ok"
                print(f"{path.name}: {c['words']}w {result['total_s']}s [{flags}]")
    finally:
        client.close()

    summary = summarize(results)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.label}-{args.category}.json"
    out_path.write_text(json.dumps({
        "label": args.label,
        "category": args.category,
        "url": url,
        "bundle": str(bundle_dir),
        "model": model_state.get("model_id"),
        "language": language,
        "summary": summary,
        "results": results,
    }, indent=2, ensure_ascii=False))

    print(f"\n== {args.label} / {args.category} ==")
    print(json.dumps(summary, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
