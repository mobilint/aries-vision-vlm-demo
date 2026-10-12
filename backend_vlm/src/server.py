import argparse
import base64
from functools import partial, wraps
import logging
import os
import time
from threading import Event, Lock

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, disconnect, emit

import core_allocation
from ImageTextToTextPipeline import ImageTextToTextPipeline
from mblt_tracker import CPUDeviceTracker, DRAMDeviceTracker, NPUDeviceTracker
from mblt_tracker.static_info import (
    get_all_pcie_devices,
    get_host_static_info,
    get_pcie_static_info,
    get_windows_npu_driver_firmware_info,
)

try:
    import psutil
except Exception:  # pragma: no cover - psutil is provided by mblt-tracker, but keep server resilient.
    psutil = None

try:
    import mbltml
except Exception:  # pragma: no cover - mbltml ships with mblt-tracker; keep server resilient.
    mbltml = None

# ARIES chip = 80 TOPS per device (static).
ARIES_TOPS_PER_DEVICE = 80


def _parse_args():
    parser = argparse.ArgumentParser(description="Category-scoped VLM server")
    parser.add_argument("--category", choices=["weapon", "fall"], required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--core-allocation-path", default=None)
    return parser.parse_args()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

_args = _parse_args()
CATEGORY = _args.category
HOST = _args.host
PORT = _args.port

_core_allocation_path = core_allocation.resolve_path(_args.core_allocation_path)
_core_config = core_allocation.load_core_allocation(_core_allocation_path, CATEGORY)
logging.info("[%s] Loaded core allocation from %s: %s", CATEGORY, _core_allocation_path, _core_config)

# Only the weapon process owns the shared mblt-tracker so the two VLM processes
# don't fight over the single NPU driver handle. The fall process emits an
# unavailable envelope on request so a misconfigured frontend doesn't crash.
SYSTEM_METRICS_ENABLED = CATEGORY == "weapon"
if not SYSTEM_METRICS_ENABLED:
    logging.info("[%s] System metrics disabled for this process (weapon owns mblt-tracker).", CATEGORY)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=3600, ping_interval=1800)

pipeline = ImageTextToTextPipeline(category=CATEGORY, core_config=_core_config)
cpu_tracker = None
dram_tracker = None
dram_tracker_unavailable_reason = None
npu_tracker = None
npu_device_count = None
system_static_info = None
system_metrics_history = []
prompt_config_ready = set()
tasks = []
task_lock = Lock()
system_metrics_lock = Lock()
vlm_model_switch_lock = Lock()
vlm_model_switch_state_lock = Lock()
vlm_model_switching = False

SYSTEM_METRICS_SAMPLE_INTERVAL_SECONDS = 60
SYSTEM_METRICS_MAX_SAMPLES = 24 * 60


def init_system_tracker():
    global cpu_tracker, dram_tracker, dram_tracker_unavailable_reason, npu_tracker, system_static_info

    if not SYSTEM_METRICS_ENABLED:
        return

    if cpu_tracker is not None and dram_tracker is not None and npu_tracker is not None:
        if system_static_info is None:
            system_static_info = collect_system_static_info()
        return

    if cpu_tracker is None:
        try:
            cpu_tracker = CPUDeviceTracker(interval=1.0)
            if hasattr(cpu_tracker, "start"):
                cpu_tracker.start()
        except Exception as exc:
            logging.warning("[system-metrics] CPU tracker unavailable: %s", exc)
            cpu_tracker = None

    if dram_tracker is None:
        try:
            dram_tracker = DRAMDeviceTracker(interval=1.0)
            if hasattr(dram_tracker, "start"):
                dram_tracker.start()
            dram_tracker_unavailable_reason = None
        except Exception as exc:
            logging.warning("[system-metrics] DRAM tracker unavailable: %s", exc)
            dram_tracker_unavailable_reason = str(exc) or exc.__class__.__name__
            dram_tracker = None

    if npu_tracker is None:
        try:
            npu_tracker = NPUDeviceTracker(interval=1.0)
            if hasattr(npu_tracker, "start"):
                npu_tracker.start()
        except Exception as exc:
            logging.warning("[system-metrics] NPU tracker unavailable: %s", exc)
            npu_tracker = None

    if system_static_info is None:
        system_static_info = collect_system_static_info()


def collect_system_static_info():
    info = {}

    try:
        info.update(get_host_static_info() or {})
    except Exception as exc:
        logging.warning("[system-metrics] Host static info unavailable: %s", exc)

    npu_info = {}
    try:
        if npu_tracker is not None and hasattr(npu_tracker, "get_static_info"):
            npu_info = npu_tracker.get_static_info() or {}
    except Exception as exc:
        logging.warning("[system-metrics] NPU tracker static info unavailable: %s", exc)

    if not npu_info:
        try:
            npu_info = get_windows_npu_driver_firmware_info() or {}
        except Exception as exc:
            logging.warning("[system-metrics] Windows NPU static info unavailable: %s", exc)

    if not npu_info:
        try:
            npu_info = get_pcie_static_info(devices=get_all_pcie_devices()) or {}
        except Exception as exc:
            logging.warning("[system-metrics] PCIe static info unavailable: %s", exc)

    if npu_info:
        info = deep_merge_dicts(info, npu_info)

    return info


def deep_merge_dicts(base, update):
    merged = dict(base or {})
    for key, value in (update or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_nested(mapping, path, default=None):
    current = mapping
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bytes_to_mb(value):
    numeric = safe_float(value)
    if numeric is None:
        return None
    return numeric / (1024 * 1024)


def format_gb_from_bytes(value):
    numeric = safe_float(value)
    if numeric is None:
        return None
    return f"{numeric / (1024 ** 3):.1f}GB"


def get_cpu_name():
    name = get_nested(system_static_info, ["hardware", "cpu", "model_name"])
    if name:
        return name
    vendor = get_nested(system_static_info, ["hardware", "cpu", "vendor"])
    architecture = get_nested(system_static_info, ["hardware", "cpu", "architecture"])
    return " ".join(part for part in (vendor, architecture, "CPU") if part) or "CPU"


def get_npu_name():
    npus = get_nested(system_static_info, ["hardware", "npus"], [])
    if isinstance(npus, list) and npus:
        names = [npu.get("name") or npu.get("driver_description") for npu in npus if isinstance(npu, dict)]
        names = [name for name in names if name]
        if names:
            unique_names = list(dict.fromkeys(names))
            return ", ".join(unique_names)
    return "NPU"


def get_npu_device_count():
    global npu_device_count

    if npu_device_count is not None:
        return npu_device_count

    if mbltml is not None and npu_tracker is not None:
        try:
            count = int(mbltml.mbltmlGetDeviceCount())
            if count > 0:
                npu_device_count = count
                return npu_device_count
        except Exception as exc:
            logging.warning("[system-metrics] mbltml device count unavailable: %s", exc)

    npus = get_nested(system_static_info, ["hardware", "npus"], [])
    if isinstance(npus, list) and npus:
        npu_device_count = len(npus)
        return npu_device_count

    return None


def get_ram_name():
    dram = get_nested(system_static_info, ["hardware", "dram"], {})
    if not isinstance(dram, dict):
        return "RAM"

    total = format_gb_from_bytes(dram.get("total_bytes"))
    dimms = dram.get("dimms")
    if isinstance(dimms, list) and dimms:
        first_dimm = next((dimm for dimm in dimms if isinstance(dimm, dict)), {})
        manufacturer = first_dimm.get("manufacturer")
        memory_type = first_dimm.get("type")
        speed = first_dimm.get("configured_speed_mhz") or first_dimm.get("speed_mhz")
        parts = [manufacturer, memory_type, total]
        if speed:
            parts.append(f"{speed}MT/s")
        return " ".join(str(part) for part in parts if part) or "RAM"

    return f"{total} RAM" if total else "RAM"


def get_npu_metrics_snapshot():
    device_count = get_npu_device_count()
    total_tops = device_count * ARIES_TOPS_PER_DEVICE if device_count else None

    if npu_tracker is None:
        snapshot = {
            "name": get_npu_name(),
            "available": False,
            "temperature_c": None,
            "utilization_pct": None,
            "power_w": None,
            "total_power_w": None,
            "source": "mblt_tracker",
        }
        if total_tops is not None:
            snapshot["total_tops"] = total_tops
        return snapshot

    metrics = {}

    try:
        metrics = npu_tracker.get_metric() or {}
    except Exception as exc:
        logging.warning("[system-metrics] Failed to read tracker metrics: %s", exc)

    temperature_c = safe_float(metrics.get("avg_temperature_c"))
    utilization_pct = safe_float(metrics.get("avg_utilization_pct"))
    power_w = safe_float(metrics.get("avg_npu_rail_power_w") or metrics.get("avg_power_w"))
    total_power_w = safe_float(metrics.get("avg_total_power_w"))

    snapshot = {
        "name": get_npu_name(),
        "available": any(value is not None for value in (temperature_c, utilization_pct, power_w, total_power_w)),
        "temperature_c": temperature_c,
        "utilization_pct": utilization_pct,
        "power_w": power_w,
        "total_power_w": total_power_w,
        "source": "mblt_tracker",
    }
    if total_tops is not None:
        snapshot["total_tops"] = total_tops
    return snapshot


def get_cpu_metrics_snapshot():
    if cpu_tracker is None:
        utilization_pct = None
        if psutil is not None:
            try:
                utilization_pct = safe_float(psutil.cpu_percent(interval=None))
            except Exception as exc:
                logging.warning("[system-metrics] Failed to read psutil CPU metrics: %s", exc)

        return {
            "name": get_cpu_name(),
            "available": utilization_pct is not None,
            "temperature_c": None,
            "utilization_pct": utilization_pct,
            "power_w": None,
            "total_power_w": None,
            "source": "psutil" if utilization_pct is not None else "mblt_tracker",
        }

    metrics = {}

    try:
        metrics = cpu_tracker.get_metric() or {}
    except Exception as exc:
        logging.warning("[system-metrics] Failed to read CPU tracker metrics: %s", exc)

    temperature_c = safe_float(metrics.get("avg_temperature_c"))
    utilization_pct = safe_float(metrics.get("avg_utilization_pct"))
    power_w = safe_float(metrics.get("avg_power_w"))
    total_power_w = safe_float(metrics.get("avg_power_w"))

    return {
        "name": get_cpu_name(),
        "available": any(value is not None for value in (temperature_c, utilization_pct, power_w, total_power_w)),
        "temperature_c": temperature_c,
        "utilization_pct": utilization_pct,
        "power_w": power_w,
        "total_power_w": total_power_w,
        "source": "mblt_tracker",
    }


def get_ram_metrics_snapshot():
    metrics = {}
    dram_metrics = {}
    if cpu_tracker is not None:
        try:
            metrics = cpu_tracker.get_metric() or {}
        except Exception as exc:
            logging.warning("[system-metrics] Failed to read tracker RAM metrics: %s", exc)

    if dram_tracker is not None:
        try:
            dram_metrics = dram_tracker.get_metric() or {}
        except Exception as exc:
            logging.warning("[system-metrics] Failed to read tracker DRAM metrics: %s", exc)

    used_mb = safe_float(metrics.get("avg_memory_used_mb"))
    total_mb = safe_float(metrics.get("total_memory_mb"))
    utilization_pct = safe_float(metrics.get("avg_memory_used_pct"))
    available_mb = None
    source = "mblt_tracker"

    if psutil is not None:
        try:
            memory = psutil.virtual_memory()
            used_mb = bytes_to_mb(memory.used)
            total_mb = bytes_to_mb(memory.total)
            available_mb = bytes_to_mb(memory.available)
            utilization_pct = safe_float(memory.percent)
            source = "mblt_tracker+psutil" if metrics else "psutil"
        except Exception as exc:
            logging.warning("[system-metrics] Failed to read psutil RAM metrics: %s", exc)

    if available_mb is None and total_mb is not None and used_mb is not None:
        available_mb = max(total_mb - used_mb, 0)

    dram_power_w = safe_float(dram_metrics.get("avg_dram_power_w"))
    dram_p99_power_w = safe_float(dram_metrics.get("p99_dram_power_w"))
    dram_max_power_w = safe_float(dram_metrics.get("max_dram_power_w"))
    dram_power_samples = dram_metrics.get("samples") if isinstance(dram_metrics, dict) else None
    if dram_power_samples is not None:
        try:
            dram_power_samples = int(dram_power_samples)
        except (TypeError, ValueError):
            dram_power_samples = None

    dram_power_status = "available" if dram_power_w is not None else "unavailable"
    dram_power_reason = None
    if dram_power_w is None:
        dram_power_reason = dram_tracker_unavailable_reason or "RAPL DRAM domain unsupported on this system"

    return {
        "name": get_ram_name(),
        "available": any(
            value is not None
            for value in (used_mb, total_mb, available_mb, utilization_pct, dram_power_w)
        ),
        "temperature_c": None,
        "utilization_pct": utilization_pct,
        "power_w": dram_power_w,
        "dram_power_w": dram_power_w,
        "power_status": dram_power_status,
        "power_error": dram_power_reason,
        "p99_power_w": dram_p99_power_w,
        "max_power_w": dram_max_power_w,
        "total_power_w": None,
        "power_samples": dram_power_samples,
        "used_mb": used_mb,
        "total_mb": total_mb,
        "available_mb": available_mb,
        "source": source + "+dram_tracker" if dram_metrics else source,
    }


def get_system_metrics_snapshot():
    return {
        "timestamp": int(time.time()),
        "cpu": get_cpu_metrics_snapshot(),
        "npu": get_npu_metrics_snapshot(),
        "ram": get_ram_metrics_snapshot(),
    }


def get_unavailable_system_metrics_payload():
    now = int(time.time())
    empty_component = {
        "name": None,
        "available": False,
        "temperature_c": None,
        "utilization_pct": None,
        "power_w": None,
        "total_power_w": None,
        "source": None,
    }
    return {
        "current": {
            "timestamp": now,
            "cpu": dict(empty_component),
            "npu": dict(empty_component),
            "ram": dict(empty_component),
        },
        "history": [],
        "sample_interval_seconds": SYSTEM_METRICS_SAMPLE_INTERVAL_SECONDS,
        "max_samples": SYSTEM_METRICS_MAX_SAMPLES,
        "available": False,
    }


def get_system_metrics_payload():
    if not SYSTEM_METRICS_ENABLED:
        return get_unavailable_system_metrics_payload()

    init_system_tracker()

    with system_metrics_lock:
        now = int(time.time())
        should_sample = (
            len(system_metrics_history) == 0
            or now - system_metrics_history[-1].get("timestamp", 0) >= SYSTEM_METRICS_SAMPLE_INTERVAL_SECONDS
        )

        if should_sample:
            system_metrics_history.append(get_system_metrics_snapshot())
            del system_metrics_history[:-SYSTEM_METRICS_MAX_SAMPLES]

        current = system_metrics_history[-1] if system_metrics_history else get_system_metrics_snapshot()
        return {
            "current": current,
            "history": list(system_metrics_history),
            "sample_interval_seconds": SYSTEM_METRICS_SAMPLE_INTERVAL_SECONDS,
            "max_samples": SYSTEM_METRICS_MAX_SAMPLES,
        }


def getsid(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_id = request.sid  # type: ignore
        if not session_id:
            logging.error(f"[{session_id}] No session Id found in request.")
            disconnect()
            return
        return f(session_id, *args, **kwargs)

    return decorated_function


def on_image_processing_done(session_id):
    logging.info(f"[{session_id}] - Image processing finished. Emitting signal.")
    socketio.emit("image", {}, to=session_id)


def on_token(token, session_id):
    socketio.emit("token", token, to=session_id)


def on_end(is_aborted, session_id):
    socketio.emit("end", is_aborted, to=session_id)
    logging.info(f"[{session_id}] - Stream ended. Aborted: {is_aborted}")


def emit_system_metrics(session_id):
    socketio.emit("system_metrics", get_system_metrics_payload(), to=session_id)


def emit_vlm_model_state(session_id=None, is_switching=False, message=None):
    payload = {
        **pipeline.get_model_state(),
        "is_switching": is_switching,
        "message": message,
    }
    if session_id is None:
        socketio.emit("vlm_model_state", payload)
        socketio.emit("model", payload["model_id"])
    else:
        socketio.emit("vlm_model_state", payload, to=session_id)
        socketio.emit("model", payload["model_id"], to=session_id)


def set_vlm_model_switching(is_switching):
    global vlm_model_switching
    with vlm_model_switch_state_lock:
        vlm_model_switching = is_switching


def is_vlm_model_switching():
    with vlm_model_switch_state_lock:
        return vlm_model_switching


def abort_all_vlm_work(wait_timeout=None):
    with task_lock:
        tasks.clear()

    session_ids = list(pipeline.sessions.keys())
    for session_id in session_ids:
        pipeline.abort_generation(session_id)

    for session_id in session_ids:
        pipeline.wait_for_generation(session_id, timeout=wait_timeout)
        socketio.emit("tasks", 0, to=session_id)


def change_vlm_model(requested_model_id):
    if not requested_model_id:
        raise ValueError('Invalid request. "model_id" is required.')

    with vlm_model_switch_lock:
        set_vlm_model_switching(True)
        emit_vlm_model_state(is_switching=True, message="Switching VLM model...")
        try:
            abort_all_vlm_work()
            model_state = pipeline.switch_model(requested_model_id)
            payload = {
                **model_state,
                "is_switching": False,
                "message": None,
            }
            set_vlm_model_switching(False)
            socketio.emit("vlm_model_state", payload)
            socketio.emit("model", payload["model_id"])
            return payload
        except Exception as exc:
            set_vlm_model_switching(False)
            emit_vlm_model_state(is_switching=False, message=str(exc))
            raise
        finally:
            set_vlm_model_switching(False)


def emit_tasks_for_sessions():
    with task_lock:
        unique_session_ids = []
        for task in tasks:
            if task["sid"] not in unique_session_ids:
                unique_session_ids.append(task["sid"])

        for session_id in unique_session_ids:
            first_index = next(index for index, task in enumerate(tasks) if task["sid"] == session_id)
            socketio.emit("tasks", first_index + 1, to=session_id)


def enqueue_task(task):
    with task_lock:
        tasks.append(task)
        session_id = task["sid"]
        first_index = next(index for index, item in enumerate(tasks) if item["sid"] == session_id)
        socketio.emit("tasks", first_index + 1, to=session_id)


def remove_tasks_for_session(session_id):
    with task_lock:
        tasks[:] = [task for task in tasks if task["sid"] != session_id]

    emit_tasks_for_sessions()


def pop_next_task():
    with task_lock:
        if tasks:
            task = tasks.pop(0)
        else:
            task = None

    if task is not None:
        socketio.emit("tasks", 0, to=task["sid"])
        emit_tasks_for_sessions()

    return task


def run_vlm_generation(session_id, question, base64image=None):
    temp_image_path = None
    on_image_done_callback = None
    generation_done = Event()

    try:
        if base64image:
            _, encoded = base64image.split(",", 1)
            image_data = base64.b64decode(encoded)
            temp_image_path = os.path.join("./src", f"temp-{session_id}-{time.time_ns()}.jpg")

            with open(temp_image_path, "wb") as file_handle:
                file_handle.write(image_data)
            logging.info(f"[{session_id}] - Saved temp image to {temp_image_path}")

            on_image_done_callback = partial(on_image_processing_done, session_id=session_id)

        on_token_callback = partial(on_token, session_id=session_id)

        def on_end_callback(is_aborted):
            try:
                on_end(is_aborted, session_id)
            finally:
                if temp_image_path and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
                generation_done.set()

        socketio.emit("start", to=session_id)
        pipeline.generate_stream(
            session_id,
            temp_image_path,
            question,
            on_token_callback,
            on_end_callback,
            on_image_done_callback,
        )
        generation_done.wait()

    except Exception as exc:
        logging.error(f"[{session_id}] - Error during ask inference: {exc}")
        socketio.emit("error", {"message": "Failed to process the ask request."}, to=session_id)
        socketio.emit("end", True, to=session_id)

        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


def task_worker():
    logging.info("Task worker thread started.")

    while True:
        task = pop_next_task()
        if task is None:
            time.sleep(0.05)
            continue

        session_id = task["sid"]
        task_value = task["value"]
        logging.info(f"[{session_id}] - Processing VLM task.")
        run_vlm_generation(session_id, **task_value)


@socketio.on("connect")
@getsid
def handle_connect(session_id):
    logging.info(f"[{session_id}] - Session connected.")
    init_system_tracker()
    pipeline.reset_session(session_id)
    prompt_config_ready.discard(session_id)
    socketio.emit("prompt_config_state", {"is_ready": False, "message": "Prompt bundle is not synced yet."}, to=session_id)
    socketio.emit("model", pipeline.original_model_id, to=session_id)
    emit_vlm_model_state(session_id)
    emit_system_metrics(session_id)


@socketio.on("disconnect")
@getsid
def handle_disconnect(session_id):
    pipeline.abort_generation(session_id)
    remove_tasks_for_session(session_id)
    prompt_config_ready.discard(session_id)
    logging.info(f"[{session_id}] - Session disconnected.")


@socketio.on("prompt_config")
@getsid
def handle_prompt_config(session_id, prompt_config):
    if is_vlm_model_switching():
        emit("error", {"message": "VLM model is switching. Please try again shortly."}, to=session_id)
        return

    if not isinstance(prompt_config, dict):
        emit("error", {"message": "Prompt config payload is invalid."}, to=session_id)
        return

    system_prompt = prompt_config.get("system_prompt", "")
    inter_prompt = prompt_config.get("inter_prompt", "")

    socketio.emit("prompt_config_state", {"is_ready": False, "message": "Applying prompt bundle..."}, to=session_id)
    prompt_config_ready.discard(session_id)
    pipeline.abort_generation(session_id)
    pipeline.wait_for_generation(session_id)
    pipeline.set_session_prompts(session_id, system_prompt, inter_prompt)
    prompt_config_ready.add(session_id)
    socketio.emit("prompt_config_state", {"is_ready": True, "message": None}, to=session_id)
    emit("prompt_config_saved", to=session_id)


@socketio.on("ask")
@getsid
def handle_ask(session_id, question, base64image=None):
    if is_vlm_model_switching():
        emit("error", {"message": "VLM model is switching. Please try again shortly."}, to=session_id)
        return

    if session_id not in prompt_config_ready:
        emit("error", {"message": "Prompt bundle is not ready yet."}, to=session_id)
        return

    if not question:
        logging.warning(f"[{session_id}] - Invalid request received. Missing question.")
        emit("error", {"message": 'Invalid request. "question" is required.'}, to=session_id)
        return

    logging.info(f"[{session_id}] - Received 'ask' request.")
    enqueue_task({
        "sid": session_id,
        "value": {
            "question": question,
            "base64image": base64image,
        },
    })


@socketio.on("abort")
@getsid
def handle_abort(session_id):
    pipeline.abort_generation(session_id)
    remove_tasks_for_session(session_id)


@socketio.on("reset")
@getsid
def handle_reset(session_id):
    if is_vlm_model_switching():
        emit("error", {"message": "VLM model is switching. Please try again shortly."}, to=session_id)
        return

    pipeline.abort_generation(session_id)
    remove_tasks_for_session(session_id)
    pipeline.wait_for_generation(session_id)
    pipeline.reset_session(session_id)
    socketio.emit("tasks", 0, to=session_id)
    socketio.emit("reset_done", to=session_id)


@socketio.on("system_metrics:get")
@getsid
def handle_system_metrics_get(session_id):
    emit_system_metrics(session_id)


@socketio.on("vlm_models:get")
@getsid
def handle_vlm_models_get(session_id):
    emit_vlm_model_state(session_id)


@socketio.on("vlm_model:set")
@getsid
def handle_vlm_model_set(session_id, payload):
    try:
        requested_model_id = payload.get("model_id") if isinstance(payload, dict) else payload
        change_vlm_model(requested_model_id)
    except Exception as exc:
        logging.error(f"[{session_id}] - Failed to switch VLM model: {exc}")
        emit_vlm_model_state(session_id=None, is_switching=False, message=str(exc))
        emit("error", {"message": str(exc)}, to=session_id)


@app.route("/models", methods=["GET"])
def get_vlm_models():
    return jsonify(pipeline.get_model_state())


@app.route("/model", methods=["POST"])
def post_vlm_model():
    payload = request.get_json(silent=True) or {}
    requested_model_id = payload.get("model_id")
    try:
        return jsonify(change_vlm_model(requested_model_id))
    except Exception as exc:
        logging.error("Failed to switch VLM model via HTTP: %s", exc)
        emit_vlm_model_state(session_id=None, is_switching=False, message=str(exc))
        return jsonify({
            **pipeline.get_model_state(),
            "ok": False,
            "message": str(exc),
        }), 400


if __name__ == "__main__":
    socketio.start_background_task(target=task_worker)
    socketio.run(app, host=HOST, port=PORT, allow_unsafe_werkzeug=True)
