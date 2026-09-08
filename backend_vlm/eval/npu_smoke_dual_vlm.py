"""Wave-1 smoke: prove ARIES can hold two Qwen3-VL-2B instances simultaneously
on disjoint NPU core slots from core_allocation.yaml.

Loads two independent VLM instances, each pinned to its own vision-encoder and
text-decoder cores per the yaml, then runs a fixed prompt sequentially and
concurrently. If concurrent per-thread timings are much worse than sequential,
that is the context-switch signal — the disjoint pinning is either wrong in
the yaml or not respected by the runtime, and the 2-process architecture of
the dual-VLM refactor is invalid.

Usage (from backend_vlm/ with the local .venv):
    .venv/bin/python eval/npu_smoke_dual_vlm.py

Exit criteria: both sequential_ok and concurrent_ok true, and the concurrent
per-thread decode and image-encode times within ~2x of the sequential ones.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from typing import Callable, Optional

import yaml
from PIL import Image

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core_allocation import resolve_path  # noqa: E402

from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402

try:
    from mblt_tracker import NPUDeviceTracker
except Exception:  # pragma: no cover - tracker is optional for the smoke.
    NPUDeviceTracker = None


DEFAULT_MODEL_ID = "mobilint/Qwen3-VL-2B-Instruct"
DEFAULT_PROMPT = "describe this image in one short sentence"
MAX_NEW_TOKENS = 30

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("npu_smoke_dual_vlm")


def load_instance_configs(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    vlm_section = data.get("vlm") or {}
    result = {}
    for name in ("weapon", "fall"):
        entry = vlm_section.get(name)
        if not isinstance(entry, dict):
            raise KeyError(f"core_allocation.yaml missing vlm.{name}")
        result[name] = {
            "vision_target_cores": list(entry["vision_target_cores"]),
            "text_target_cores": list(entry["text_target_cores"]),
            "vision_core_mode": entry.get("vision_core_mode", "single"),
            "text_core_mode": entry.get("text_core_mode", "single"),
        }
    return result


def snapshot_npu_metrics(tracker) -> Optional[dict]:
    if tracker is None:
        return None
    try:
        if hasattr(tracker, "get_current_metrics"):
            return tracker.get_current_metrics() or None
    except Exception as exc:
        log.warning("NPU tracker read failed: %s", exc)
    return None


def load_instance(name: str, model_id: str, cfg: dict):
    log.info(
        "[%s] loading %s vision_cores=%s text_cores=%s vision_mode=%s text_mode=%s",
        name,
        model_id,
        cfg["vision_target_cores"],
        cfg["text_target_cores"],
        cfg["vision_core_mode"],
        cfg["text_core_mode"],
    )
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            trust_remote_code=True,
            vision_target_cores=cfg["vision_target_cores"],
            text_target_cores=cfg["text_target_cores"],
            vision_core_mode=cfg["vision_core_mode"],
            text_core_mode=cfg["text_core_mode"],
        )
        return model, processor
    except Exception as exc:
        log.error(
            "[%s] LOAD FAILED with vision_cores=%s text_cores=%s: %s\n%s",
            name,
            cfg["vision_target_cores"],
            cfg["text_target_cores"],
            exc,
            traceback.format_exc(),
        )
        raise


def make_test_image() -> Image.Image:
    return Image.new("RGB", (448, 448), color=(96, 128, 192))


def run_generate(name: str, model, processor, image: Image.Image, prompt: str,
                 on_image_encode_done: Optional[Callable] = None) -> dict:
    conversation = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to("cpu")

    # Patch get_image_features so we can measure the vision-encoder pass separately.
    image_encode_ms = {"value": None}
    original_get_image_features = getattr(model, "get_image_features", None)
    if original_get_image_features is not None:
        encode_start = {"t": None}

        def wrapped(*args, **kwargs):
            encode_start["t"] = time.perf_counter()
            out = original_get_image_features(*args, **kwargs)
            image_encode_ms["value"] = (time.perf_counter() - encode_start["t"]) * 1000.0
            if on_image_encode_done is not None:
                on_image_encode_done()
            return out

        model.get_image_features = wrapped

    try:
        t0 = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        if original_get_image_features is not None:
            model.get_image_features = original_get_image_features

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    result = {
        "name": name,
        "total_ms": total_ms,
        "image_encode_ms": image_encode_ms["value"],
        "decode_ms": total_ms - (image_encode_ms["value"] or 0.0),
        "answer": answer,
    }
    log.info(
        "[%s] total=%.0fms image_encode=%s decode~%.0fms answer=%r",
        name,
        total_ms,
        f"{image_encode_ms['value']:.0f}ms" if image_encode_ms["value"] is not None else "n/a",
        result["decode_ms"],
        answer[:80],
    )
    return result


def run_sequential(instances, image, prompt) -> list:
    results = []
    for name, model, processor in instances:
        results.append(run_generate(name, model, processor, image, prompt))
    return results


def run_concurrent(instances, image, prompt) -> tuple:
    results = {}
    errors = {}

    def worker(name, model, processor):
        try:
            results[name] = run_generate(name, model, processor, image, prompt)
        except Exception as exc:
            errors[name] = f"{exc}\n{traceback.format_exc()}"

    threads = []
    combined_start = time.perf_counter()
    for name, model, processor in instances:
        t = threading.Thread(target=worker, args=(name, model, processor), name=f"gen-{name}")
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    combined_ms = (time.perf_counter() - combined_start) * 1000.0
    return results, errors, combined_ms


def compare(seq: list, conc: dict, notes: list) -> tuple:
    concurrent_ok = True
    for seq_result in seq:
        name = seq_result["name"]
        conc_result = conc.get(name)
        if conc_result is None:
            notes.append(f"{name}: missing concurrent result")
            concurrent_ok = False
            continue
        seq_decode = seq_result["decode_ms"]
        conc_decode = conc_result["decode_ms"]
        seq_encode = seq_result["image_encode_ms"]
        conc_encode = conc_result["image_encode_ms"]

        if seq_decode > 0 and conc_decode > 2.0 * seq_decode:
            notes.append(
                f"{name}: apparent NPU contention (text side): concurrent decode "
                f"{conc_decode:.0f}ms vs sequential {seq_decode:.0f}ms"
            )
            concurrent_ok = False
        if (seq_encode and conc_encode and seq_encode > 0
                and conc_encode > 2.0 * seq_encode):
            notes.append(
                f"{name}: apparent NPU contention (vision side): concurrent image_encode "
                f"{conc_encode:.0f}ms vs sequential {seq_encode:.0f}ms"
            )
            concurrent_ok = False
    return concurrent_ok, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--core-allocation-path", default=None,
                        help="Path to core_allocation.yaml (defaults via core_allocation.resolve_path)")
    parser.add_argument("--sequential", dest="sequential", action="store_true", default=True)
    parser.add_argument("--no-sequential", dest="sequential", action="store_false")
    parser.add_argument("--concurrent", dest="concurrent", action="store_true", default=True)
    parser.add_argument("--no-concurrent", dest="concurrent", action="store_false")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    yaml_path = resolve_path(args.core_allocation_path)
    log.info("core_allocation.yaml: %s", yaml_path)

    cfgs = load_instance_configs(yaml_path)
    for name, cfg in cfgs.items():
        log.info(
            "instance=%s vision_target_cores=%s text_target_cores=%s",
            name, cfg["vision_target_cores"], cfg["text_target_cores"],
        )

    tracker = None
    if NPUDeviceTracker is not None:
        try:
            tracker = NPUDeviceTracker(interval=1.0)
            if hasattr(tracker, "start"):
                tracker.start()
        except Exception as exc:
            log.warning("NPU tracker unavailable: %s", exc)
            tracker = None

    pre_load_metrics = snapshot_npu_metrics(tracker)
    log.info("pre-load NPU metrics: %s", pre_load_metrics)

    instances = []
    for name in ("weapon", "fall"):
        model, processor = load_instance(name, args.model_id, cfgs[name])
        instances.append((name, model, processor))

    post_load_metrics = snapshot_npu_metrics(tracker)
    log.info("post-load NPU metrics: %s", post_load_metrics)

    image = make_test_image()

    seq_results = []
    seq_ok = False
    if args.sequential:
        log.info("=== sequential phase ===")
        seq_results = run_sequential(instances, image, args.prompt)
        seq_ok = all(bool(r["answer"]) for r in seq_results)

    conc_results = {}
    conc_errors = {}
    combined_ms = None
    conc_ok = False
    notes: list = []
    if args.concurrent:
        log.info("=== concurrent phase ===")
        conc_results, conc_errors, combined_ms = run_concurrent(instances, image, args.prompt)
        if conc_errors:
            for name, err in conc_errors.items():
                notes.append(f"{name}: concurrent generate raised: {err.splitlines()[0]}")
            conc_ok = False
        else:
            all_have_answers = all(bool(r["answer"]) for r in conc_results.values())
            if args.sequential and seq_ok:
                conc_ok, notes = compare(seq_results, conc_results, notes)
                conc_ok = conc_ok and all_have_answers
            else:
                conc_ok = all_have_answers

    summary = {
        "model_id": args.model_id,
        "core_allocation_path": yaml_path,
        "cores": {
            "weapon_vision_target_cores": cfgs["weapon"]["vision_target_cores"],
            "weapon_text_target_cores": cfgs["weapon"]["text_target_cores"],
            "fall_vision_target_cores": cfgs["fall"]["vision_target_cores"],
            "fall_text_target_cores": cfgs["fall"]["text_target_cores"],
        },
        "sequential": {
            "ran": args.sequential,
            "per_instance": [
                {
                    "name": r["name"],
                    "total_ms": r["total_ms"],
                    "image_encode_ms": r["image_encode_ms"],
                    "decode_ms": r["decode_ms"],
                    "answer": r["answer"],
                }
                for r in seq_results
            ],
        },
        "concurrent": {
            "ran": args.concurrent,
            "combined_ms": combined_ms,
            "per_thread": [
                {
                    "name": name,
                    "total_ms": r["total_ms"],
                    "image_encode_ms": r["image_encode_ms"],
                    "decode_ms": r["decode_ms"],
                    "answer": r["answer"],
                }
                for name, r in conc_results.items()
            ],
            "errors": conc_errors,
        },
        "sequential_ok": seq_ok,
        "concurrent_ok": conc_ok,
        "npu_metrics_pre_load": pre_load_metrics,
        "npu_metrics_post_load": post_load_metrics,
        "notes": notes,
    }

    print("\n===== SMOKE SUMMARY =====")
    print(json.dumps(summary, indent=2, default=str))

    overall_ok = (
        (not args.sequential or seq_ok)
        and (not args.concurrent or conc_ok)
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
