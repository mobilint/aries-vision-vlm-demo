import base64
from functools import partial, wraps
import logging
import os
import time
from threading import Event, Lock

from flask import Flask, request
from flask_socketio import SocketIO, disconnect, emit

from ImageTextToTextPipeline import ImageTextToTextPipeline
from mblt_tracker import CPUDeviceTracker, NPUDeviceTracker

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=3600, ping_interval=1800)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

pipeline = ImageTextToTextPipeline()
cpu_tracker = None
npu_tracker = None
prompt_config_ready = set()
tasks = []
task_lock = Lock()


def init_system_tracker():
    global cpu_tracker, npu_tracker

    if cpu_tracker is not None and npu_tracker is not None:
        return

    if cpu_tracker is None:
        try:
            cpu_tracker = CPUDeviceTracker(interval=1.0)
            if hasattr(cpu_tracker, "start"):
                cpu_tracker.start()
        except Exception as exc:
            logging.warning("[system-metrics] CPU tracker unavailable: %s", exc)
            cpu_tracker = None

    if npu_tracker is None:
        try:
            npu_tracker = NPUDeviceTracker(interval=1.0)
            if hasattr(npu_tracker, "start"):
                npu_tracker.start()
        except Exception as exc:
            logging.warning("[system-metrics] NPU tracker unavailable: %s", exc)
            npu_tracker = None


def safe_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_npu_metrics_snapshot():
    if npu_tracker is None:
        return {
            "available": False,
            "temperature_c": None,
            "utilization_pct": None,
            "power_w": None,
            "total_power_w": None,
            "source": "mblt_tracker",
        }

    metrics = {}

    try:
        if hasattr(npu_tracker, "get_current_metrics"):
            metrics = npu_tracker.get_current_metrics() or {}
        elif hasattr(npu_tracker, "get_metric"):
            metrics = npu_tracker.get_metric() or {}
        elif hasattr(npu_tracker, "_fetch_metrics"):
            fetched = npu_tracker._fetch_metrics()
            if fetched is not None:
                metrics = {
                    "npu_power_w": fetched[0],
                    "total_power_w": fetched[1],
                    "npu_util_pct": fetched[2],
                    "npu_temperature_c": fetched[6] if len(fetched) > 6 else None,
                }
    except Exception as exc:
        logging.warning("[system-metrics] Failed to read tracker metrics: %s", exc)

    temperature_c = safe_float(
        metrics.get(
            "temperature_c",
            metrics.get("npu_temperature_c", metrics.get("avg_temperature_c")),
        )
    )
    utilization_pct = safe_float(
        metrics.get("utilization_pct", metrics.get("npu_util_pct", metrics.get("avg_npu_util_pct")))
    )
    power_w = safe_float(
        metrics.get("power_w", metrics.get("npu_power_w", metrics.get("avg_npu_power_w")))
    )
    total_power_w = safe_float(
        metrics.get("total_power_w", metrics.get("avg_total_power_w", metrics.get("avg_power_w")))
    )

    return {
        "available": any(value is not None for value in (temperature_c, utilization_pct, power_w, total_power_w)),
        "temperature_c": temperature_c,
        "utilization_pct": utilization_pct,
        "power_w": power_w,
        "total_power_w": total_power_w,
        "source": "mblt_tracker",
    }


def get_cpu_metrics_snapshot():
    if cpu_tracker is None:
        return {
            "available": False,
            "temperature_c": None,
            "utilization_pct": None,
            "power_w": None,
            "total_power_w": None,
            "source": "mblt_tracker",
        }

    metrics = {}

    try:
        if hasattr(cpu_tracker, "get_current_metrics"):
            metrics = cpu_tracker.get_current_metrics() or {}
        elif hasattr(cpu_tracker, "get_metric"):
            metrics = cpu_tracker.get_metric() or {}
    except Exception as exc:
        logging.warning("[system-metrics] Failed to read CPU tracker metrics: %s", exc)

    temperature_c = safe_float(metrics.get("temperature_c", metrics.get("avg_temperature_c")))
    utilization_pct = safe_float(
        metrics.get("utilization_pct", metrics.get("avg_utilization_pct"))
    )
    power_w = safe_float(metrics.get("power_w", metrics.get("avg_power_w")))
    total_power_w = safe_float(
        metrics.get("total_power_w", metrics.get("avg_total_power_w", metrics.get("avg_power_w")))
    )

    return {
        "available": any(value is not None for value in (temperature_c, utilization_pct, power_w, total_power_w)),
        "temperature_c": temperature_c,
        "utilization_pct": utilization_pct,
        "power_w": power_w,
        "total_power_w": total_power_w,
        "source": "mblt_tracker",
    }


def get_system_metrics_snapshot():
    return {
        "timestamp": int(time.time()),
        "cpu": get_cpu_metrics_snapshot(),
        "npu": get_npu_metrics_snapshot(),
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
    socketio.emit("system_metrics", get_system_metrics_snapshot(), to=session_id)


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


if __name__ == "__main__":
    socketio.start_background_task(target=task_worker)
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
