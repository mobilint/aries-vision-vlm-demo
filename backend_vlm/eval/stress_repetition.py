"""Repetition stress test: hammer the live VLM server with concurrent
sessions and an uncapped token budget to see whether runaway repetition
still reproduces.

Each VLM process serves one category (weapon on :5000, fall on :5001);
one invocation stresses one category / one port.

- N workers (default 4), each with its own Socket.IO session, cycling
  through the captured cases for --category.
- Temporarily raises max_new_tokens in src/generation_config.json (the
  server re-reads it on every ask) and restores it on exit.
- Flags per answer:
    repetition      exact loops (word- and char-level, numbers normalized)
    near_dup        3+ nearly-identical sentences (SequenceMatcher >= 0.85)
    token_runaway   more than --token-limit streamed tokens
    slow_decode     first-token -> end longer than --decode-limit seconds
    timeout         no end event within --ask-timeout seconds
- Appends one JSON line per trial to results/stress-<label>.jsonl as it
  goes, then prints a summary.

Usage:
    .venv/bin/python stress_repetition.py --category weapon --label qwen2 --trials 100
"""

import argparse
import difflib
import itertools
import json
import pathlib
import re
import threading
import time

import requests

from harness import (
    VlmClient,
    annotate_image_data_url,
    build_detection_prompt,
    eligible_detections,
    has_runaway_repetition,
)

CATEGORY_PORTS = {"weapon": 5000, "fall": 5001}

ROOT = pathlib.Path(__file__).parent
REPO = ROOT.parent.parent
# The shared config plus every per-model override (e.g. aya's) — all carry
# their own max_new_tokens, so the uncap has to touch each of them.
GEN_CONFIGS = sorted((REPO / "backend_vlm" / "src").glob("generation_config*.json"))

SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def near_duplicate_sentences(text, similarity=0.85, min_len=15, min_count=3):
    """Detect 'almost the same sentence, a few chars differ' loops."""
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) >= min_len]
    for i, base in enumerate(sentences):
        similar = 1
        for other in sentences[i + 1:]:
            if difflib.SequenceMatcher(None, base, other).ratio() >= similarity:
                similar += 1
                if similar >= min_count:
                    return True
    return False


def load_cases(category):
    cases = []
    for path in sorted((ROOT / "cases" / category).glob("case_*.json")):
        cases.append(json.loads(path.read_text()))
    if not cases:
        raise SystemExit(f"no cases for {category}; run capture_cases.py first")
    return cases


def load_bundle(category, language="en", bundle_root=None):
    # Prompt-bundle directories still use the "<category>_detection" naming
    # (shared with the shipped frontend); only the eval-internal identifiers
    # migrated to the bare category.
    root = pathlib.Path(bundle_root) if bundle_root else REPO / "frontend" / "public" / "prompt-bundles"
    bundle = root / f"{category}_detection" / language
    system = (bundle / "system.txt").read_text().strip()
    inter_path = bundle / "inter.txt"
    inter = inter_path.read_text().strip() if inter_path.exists() else ""
    return system, inter


def worker(worker_id, category, cases, trials, args, out_path, lock, stats):
    client = VlmClient(args.url)
    try:
        client.set_prompts(*load_bundle(category, bundle_root=args.bundle_root))
        case_cycle = itertools.cycle(cases)
        for trial in range(trials):
            case = next(case_cycle)
            channel = case["channel"]
            marked = eligible_detections(channel["detections"], category, case["threshold"])
            prompt = build_detection_prompt(channel, case["trigger"], marked, category)
            image = annotate_image_data_url(
                channel["image_base64"], marked,
                channel["image_width"], channel["image_height"],
            )
            record = {"worker": worker_id, "category": category, "trial": trial}
            try:
                result = client.ask(prompt, image, timeout=args.ask_timeout)
                record.update({
                    "tokens": result["token_count"],
                    "decode_s": result["decode_s"],
                    "total_s": result["total_s"],
                    "flags": sorted(k for k, v in {
                        "repetition": has_runaway_repetition(result["text"]),
                        "near_dup": near_duplicate_sentences(result["text"]),
                        "token_runaway": result["token_count"] > args.token_limit,
                        "slow_decode": result["decode_s"] > args.decode_limit,
                    }.items() if v),
                    "text": result["text"],
                })
            except TimeoutError:
                record.update({"flags": ["timeout"], "text": "".join(client._tokens)})
            with lock:
                out_path.open("a").write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["done"] += 1
                flags = record["flags"]
                if flags:
                    stats["flagged"] += 1
                    print(f"[w{worker_id} t{trial}] FLAGS={flags} tokens={record.get('tokens','?')} "
                          f"decode={record.get('decode_s','?')}s :: {record['text'][:80]!r}")
                if stats["done"] % 10 == 0:
                    print(f"progress: {stats['done']} trials, {stats['flagged']} flagged")
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=sorted(CATEGORY_PORTS), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--vlm-host", default="localhost")
    parser.add_argument("--vlm-port", type=int, default=None,
                        help="override the category-derived port (weapon=5000, fall=5001)")
    parser.add_argument("--model", default=None,
                        help="frontend model id; switches the server before the test")
    parser.add_argument("--bundle-root", default=None,
                        help="alternative prompt-bundles root (e.g. a git-show snapshot of master)")
    parser.add_argument("--config-file", default=None,
                        help="replace the shared generation config with this file during the test "
                             "(max_new_tokens still overridden by --max-new-tokens)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trials", type=int, default=100, help="total trials across all workers")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="temporary uncapped budget during the test")
    parser.add_argument("--token-limit", type=int, default=300,
                        help="tokens above this flag token_runaway")
    parser.add_argument("--decode-limit", type=float, default=30.0)
    parser.add_argument("--ask-timeout", type=float, default=240.0)
    args = parser.parse_args()

    port = args.vlm_port if args.vlm_port is not None else CATEGORY_PORTS[args.category]
    args.url = f"http://{args.vlm_host}:{port}"

    if args.model:
        state = requests.get(f"{args.url}/models", timeout=10).json()
        if state["model_id"] != args.model:
            print(f"switching model {state['model_id']} -> {args.model} ...")
            response = requests.post(f"{args.url}/model", json={"model_id": args.model}, timeout=900)
            response.raise_for_status()
            if response.json().get("ok") is False:
                raise SystemExit(f"model switch failed: {response.json().get('message')}")

    cases = load_cases(args.category)
    out_path = ROOT / "results" / f"stress-{args.label}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("")

    backups = {}
    for path in GEN_CONFIGS:
        backups[path] = path.read_text()
        if args.config_file and path.name == "generation_config.json":
            config = json.loads(pathlib.Path(args.config_file).read_text())
            print(f"{path.name}: replaced with {args.config_file} (temporary)")
        else:
            config = json.loads(backups[path])
        original_cap = config.get("max_new_tokens")
        config["max_new_tokens"] = args.max_new_tokens
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"{path.name}: max_new_tokens {original_cap} -> {args.max_new_tokens} (temporary)")

    lock = threading.Lock()
    stats = {"done": 0, "flagged": 0}
    per_worker = args.trials // args.workers
    started = time.monotonic()
    try:
        threads = []
        for i in range(args.workers):
            t = threading.Thread(
                target=worker,
                args=(i, args.category, cases, per_worker, args, out_path, lock, stats),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    finally:
        for path, content in backups.items():
            path.write_text(content)
        print("generation configs restored")

    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    flagged = [r for r in records if r["flags"]]
    by_flag = {}
    for r in flagged:
        for f in r["flags"]:
            by_flag[f] = by_flag.get(f, 0) + 1
    tokens = sorted(r.get("tokens", 0) for r in records if "tokens" in r)
    print(f"\n== stress-{args.label}: {len(records)} trials in {time.monotonic()-started:.0f}s ==")
    print(f"flagged: {len(flagged)}/{len(records)}  by type: {by_flag or 'none'}")
    if tokens:
        print(f"tokens: min {tokens[0]} / median {tokens[len(tokens)//2]} / max {tokens[-1]}")
    print(f"details: {out_path}")


if __name__ == "__main__":
    main()
