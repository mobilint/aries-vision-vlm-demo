"""Shared helpers for the VLM prompt evaluation harness.

Replicates what the frontend does before an auto-triggered ask:
- picks the trigger detection for a captured /detections channel snapshot
- draws the red detection markers on the image (createAnnotatedDetectionImageDataUrl)
- builds the detection_event user prompt (buildDetectionPrompt)
and then drives the live backend_vlm Socket.IO server to collect the
streamed answer with timing metrics and rule checks.
"""

import base64
import io
import re
import time

import socketio
from PIL import Image, ImageDraw

DEFAULT_DETECTION_THRESHOLD = 0.6

# Mirrors VISION_AUTO_TRIGGER_CONFIG_BY_CATEGORY in frontend/app/settings.ts:
# label filtering is by string name, not integer index, so the eval harness
# stays label-map-agnostic (the vision backend can renumber labels without
# breaking the eval).
AUTO_TRIGGER_CONFIG = {
    "weapon": {"allowed_label_names": None},
    "fall": {"allowed_label_names": {"falling"}},
}


def eligible_detections(detections, category, threshold=DEFAULT_DETECTION_THRESHOLD):
    allowed = AUTO_TRIGGER_CONFIG[category]["allowed_label_names"]
    return [
        d for d in detections
        if d["confidence"] > threshold and (allowed is None or d["label_name"] in allowed)
    ]


def location_hint(roi, image_width, image_height):
    x, y, w, h = roi
    cx = (x + w / 2) / image_width if image_width > 0 else 0.5
    cy = (y + h / 2) / image_height if image_height > 0 else 0.5
    horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    vertical = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    return f"{horizontal} area" if vertical == "middle" else f"{vertical}-{horizontal} area"


def certainty_word(confidence):
    return "high" if confidence >= 0.85 else "medium" if confidence >= 0.7 else "low"


VLM_ANSWER_INSTRUCTION = (
    "Inspect every red visual marker on the image as primary alert evidence "
    "and answer following your instructions."
)


def _same_detection(a, b):
    # Case JSON round-trips lose object identity, so the frontend's `!==` filter
    # is emulated by content match on roi + label_name (unique per frame).
    return tuple(a["roi"]) == tuple(b["roi"]) and a["label_name"] == b["label_name"]


def build_detection_prompt(channel, trigger, marked_detections, category):
    """Numeric-free user message mirroring frontend/app/page.tsx
    buildDetectionPrompt exactly: no roi arrays, no confidence floats, no
    channel metadata — only words. Spatial info reaches the model via the
    red boxes drawn on the image plus the location hints."""
    width, height = channel["image_width"], channel["image_height"]
    lines = [
        "detection_event:",
        f"category: {category}",
        "trigger_detection:",
        f"  object: {trigger['label_name']}",
        f"  certainty: {certainty_word(trigger['confidence'])}",
        f"  location: {location_hint(trigger['roi'], width, height)}",
    ]
    others = [d for d in marked_detections if not _same_detection(d, trigger)]
    if others:
        lines.append("other_red_marked_detections:")
        for d in others:
            lines.extend([
                f"  - object: {d['label_name']}",
                f"    certainty: {certainty_word(d['confidence'])}",
                f"    location: {location_hint(d['roi'], width, height)}",
            ])
    lines.append(VLM_ANSWER_INSTRUCTION)
    return "\n".join(lines)


def annotate_image_data_url(image_base64, detections, image_width, image_height):
    image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    if image.size != (image_width, image_height):
        image = image.resize((image_width, image_height))
    draw = ImageDraw.Draw(image)
    stroke = max(5, round(min(image_width, image_height) * 0.01))
    for detection in detections:
        x, y, w, h = detection["roi"]
        x, y = max(0, x), max(0, y)
        w = max(0, min(w, image_width - x))
        h = max(0, min(h, image_height - y))
        if w <= 0 or h <= 0:
            continue
        draw.rectangle([x, y, x + w, y + h], outline=(255, 59, 48), width=stroke)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


COORDINATE_PATTERNS = [
    re.compile(r"\(\s*\d{1,4}\s*,\s*\d{1,4}"),                     # (168, 52, ...
    re.compile(r"\[\s*\d{1,4}\s*,\s*\d{1,4}"),                     # [168, 52, ...
    re.compile(r"\b\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\s*,\s*\d{1,4}\b"),  # bare 4-tuple
    re.compile(r"\broi\b|bounding box|coordinates?\b|\bpixels?\b", re.IGNORECASE),
    re.compile(r"바운딩\s*박스|좌표|픽셀|バウンディング\s*ボックス|座標|ピクセル|边界框|坐标|像素"),
]

# Script presence per language: an answer in the wrong language is a failure.
_HANGUL = re.compile(r"[가-힣]")
_KANA = re.compile(r"[ぁ-んァ-ヶ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN_WORD = re.compile(r"\b[A-Za-z]{2,}\b")

CJK_LANGUAGES = ("ko", "ja", "zh")


def is_wrong_language(text, language):
    if language == "ko":
        return not _HANGUL.search(text)
    if language == "ja":
        # Japanese prose always carries kana; han alone would be Chinese.
        return not _KANA.search(text)
    if language == "zh":
        return not _HAN.search(text) or bool(_KANA.search(text)) or bool(_HANGUL.search(text))
    return bool(_HANGUL.search(text) or _KANA.search(text) or _HAN.search(text))

METADATA_PATTERN = re.compile(
    r"channel_index|feeder_index|model_index|detection_event|label_name", re.IGNORECASE
)


_PUNCT = str.maketrans("", "", ".,:;!?\"'()[]“”")
_DIGIT = re.compile(r"\d")


def _normalize_tokens(text):
    """Lowercase, strip punctuation, and collapse number-bearing tokens to
    <num> so 'confidence: 0.9721' and 'confidence: 0.9593' compare equal."""
    tokens = []
    for raw in text.split():
        token = raw.translate(_PUNCT).lower()
        if not token:
            continue
        tokens.append("<num>" if _DIGIT.search(token) else token)
    return tokens


def has_runaway_repetition(text, ngram=4, max_total=3, max_consecutive=3):
    """Detect degenerate loops in two ways:
    - the same (normalized) n-gram occurring more than max_total times
      anywhere ('label: falling confidence: 0.97 ... label: falling
      confidence: 0.95 ...' — numbers normalized so the loop matches)
    - any 1-4 word unit repeated max_consecutive+ times back to back
      ('falling falling falling')."""
    tokens = _normalize_tokens(text)

    counts = {}
    for i in range(max(0, len(tokens) - ngram + 1)):
        key = tuple(tokens[i:i + ngram])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > max_total:
            return True

    for n in range(1, 5):
        for i in range(len(tokens) - n):
            unit = tokens[i:i + n]
            count = 1
            j = i + n
            while tokens[j:j + n] == unit:
                count += 1
                if count >= max_consecutive:
                    return True
                j += n

    # Character-level pass for unspaced scripts (ja/zh have no word
    # boundaries): any 6+ char chunk repeated 3+ times back to back.
    condensed = _DIGIT.sub("0", re.sub(r"\s+", "", text))
    if re.search(r"(.{6,30})\1{2}", condensed):
        return True
    return False


SENTENCE_ENDINGS = (".", "!", "?", '"', "”", ")", "。", "！", "？", "」", "』", "다.")


def check_answer(text, language="en", min_words=15, max_words=80,
                 min_chars=20, max_chars=140):
    """Score one answer. For CJK languages length is measured in
    non-whitespace characters instead of words."""
    words = len(text.split())
    stripped = text.strip()
    if language in CJK_LANGUAGES:
        length = len(re.sub(r"\s+", "", stripped))
        too_short, too_long = length < min_chars, length > max_chars
    else:
        length = words
        too_short, too_long = words < min_words, words > max_words
    return {
        "words": words,
        "length": length,
        "coordinate_leak": any(p.search(text) for p in COORDINATE_PATTERNS),
        "metadata_leak": bool(METADATA_PATTERN.search(text)),
        "repetition": has_runaway_repetition(text),
        "wrong_language": is_wrong_language(text, language),
        "too_short": too_short,
        "too_long": too_long,
        "incomplete": not stripped.endswith(SENTENCE_ENDINGS),
        "empty": words == 0,
    }


class VlmClient:
    """Socket.IO client wrapper mirroring the frontend ask flow."""

    def __init__(self, url="http://localhost:5000", connect_timeout=30):
        self.sio = socketio.Client()
        self._prompt_ready = False
        self._tokens = []
        self._first_token_at = None
        self._ended = None

        @self.sio.on("prompt_config_state")
        def on_prompt_config_state(payload):
            self._prompt_ready = bool(payload.get("is_ready"))

        @self.sio.on("token")
        def on_token(token):
            if self._first_token_at is None:
                self._first_token_at = time.monotonic()
            self._tokens.append(token)

        @self.sio.on("end")
        def on_end(is_aborted):
            self._ended = bool(is_aborted)

        self.sio.connect(url, wait_timeout=connect_timeout)

    def set_prompts(self, system_prompt, inter_prompt="", timeout=60):
        self._prompt_ready = False
        self.sio.emit("prompt_config", {
            "system_prompt": system_prompt,
            "inter_prompt": inter_prompt,
        })
        deadline = time.monotonic() + timeout
        while not self._prompt_ready:
            if time.monotonic() > deadline:
                raise TimeoutError("prompt_config was not acknowledged in time")
            time.sleep(0.05)

    def ask(self, question, image_data_url, timeout=180):
        self.sio.emit("reset")
        time.sleep(0.2)
        self._tokens = []
        self._first_token_at = None
        self._ended = None

        started = time.monotonic()
        self.sio.emit("ask", (question, image_data_url))
        deadline = started + timeout
        while self._ended is None:
            if time.monotonic() > deadline:
                raise TimeoutError("generation did not finish in time")
            time.sleep(0.05)
        finished = time.monotonic()

        text = "".join(self._tokens)
        first = self._first_token_at or finished
        return {
            "text": text,
            "aborted": self._ended,
            "token_count": len(self._tokens),
            "ttft_s": round(first - started, 3),
            "decode_s": round(finished - first, 3),
            "total_s": round(finished - started, 3),
        }

    def close(self):
        self.sio.disconnect()
