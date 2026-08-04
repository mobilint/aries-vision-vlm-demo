# Install & Run

The VLM backend runs as **two category-scoped processes**, one per demo
category. Each process serves one Socket.IO server on its own port and loads
its own Qwen2-VL-2B instance.

```
uv venv
uv run src/server.py --category weapon --port 5000
uv run src/server.py --category fall   --port 5001
```

# Ports and Endpoints
- `5000`: weapon-category VLM Socket.IO server (`--category weapon`)
- `5001`: fall-category   VLM Socket.IO server (`--category fall`)

Both processes speak the same Socket.IO API (`ask`, `reset`, `prompt_config`,
`system_metrics:get`, ...); the category only affects which
`core_allocation.yaml` entry is loaded and — for `weapon` — that
`mblt-tracker` is initialized.

# Development Notes

Two VLM processes must run for full-demo functionality — one per category:

```
uv run src/server.py --category weapon --port 5000
uv run src/server.py --category fall   --port 5001
```

`docker compose up` brings both up as separate services
(`aries_vision_vlm_backend_vlm_weapon` on :5000 and
`aries_vision_vlm_backend_vlm_fall` on :5001), sharing a single built image and
the host Hugging Face cache — `~/.cache/huggingface` on the host is bind-mounted
to `/root/.cache/huggingface` inside both containers so the two processes never
duplicate the multi-GB model weights on disk or re-download them. The
Dockerfile's `CMD` runs the weapon process by default; each service's
`command:` in `docker-compose.yml` overrides that with the correct
`--category`/`--port` flags. `core_allocation.yaml` from the repo root is
mounted read-only into both containers at `/etc/aries/core_allocation.yaml`.

VLM NPU core assignment is **split** across two submodules (vision encoder +
text decoder) and lives per-category in the repo-root `core_allocation.yaml`
under `vlm.<category>.{vision,text}_target_cores`. Each process pins both
submodules to disjoint cores so the two VLMs don't context-switch on the same
NPU cores.

Only the `weapon` process initializes `mblt-tracker` (single NPU driver
handle); the `fall` process emits an unavailable envelope on
`system_metrics:get`, and the frontend reads system metrics from the weapon
connection.

Per-category default model is hardcoded in
`ImageTextToTextPipeline.CATEGORY_DEFAULT_MODEL` (weapon → Qwen3-VL-2B-Instruct,
fall → aya-vision-8b). Override via the UI selector at runtime.
