"""Capture real trigger snapshots from the running vision backend.

Polls GET /detections and stores channel snapshots that would auto-trigger
the VLM in the frontend (an eligible detection above the threshold).
The unified vision backend runs weapon and fall channels concurrently, so
this filters the returned channels by their `category` field instead of
switching the backend mode.

Usage:
    python capture_cases.py --category weapon --count 10
    python capture_cases.py --category fall --count 10
"""

import argparse
import json
import pathlib
import time

import requests

from harness import DEFAULT_DETECTION_THRESHOLD, eligible_detections

VISION_URL = "http://localhost:8081"


def capture(category, count, out_dir, threshold, min_gap_s=2.0, timeout_s=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    last_saved_per_channel = {}
    deadline = time.monotonic() + timeout_s

    while saved < count:
        if time.monotonic() > deadline:
            raise TimeoutError(f"captured only {saved}/{count} cases in {timeout_s}s")

        payload = requests.get(f"{VISION_URL}/detections", timeout=5).json()
        now = time.monotonic()
        for channel in payload["channels"]:
            if saved >= count:
                break
            if channel.get("category") != category:
                continue
            eligible = eligible_detections(channel["detections"], category, threshold)
            if not eligible:
                continue
            index = channel["channel_index"]
            if now - last_saved_per_channel.get(index, -1e9) < min_gap_s:
                continue
            last_saved_per_channel[index] = now
            trigger = max(eligible, key=lambda d: d["confidence"])
            case = {
                "category": category,
                "threshold": threshold,
                "trigger": trigger,
                "channel": channel,
            }
            path = out_dir / f"case_{saved:02d}.json"
            path.write_text(json.dumps(case))
            saved += 1
            print(f"saved {path.name} (channel {index}, {trigger['label_name']} {trigger['confidence']:.2f})")
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["weapon", "fall"], required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DETECTION_THRESHOLD)
    parser.add_argument("--out", default=None, help="output dir (default: cases/<category>)")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).parent / "cases" / args.category
    capture(args.category, args.count, out_dir, args.threshold)


if __name__ == "__main__":
    main()
