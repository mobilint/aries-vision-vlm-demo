# ARIES Vision VLM Demo

`ARIES Vision VLM Demo` is a three-service demo that connects:

- a native ARIES vision pipeline (`backend_vision`)
- a VLM inference server (`backend_vlm`)
- a Next.js dashboard (`frontend`)

The current codebase is built around automatic weapon-detection events. The vision backend publishes an MJPEG stream and JSON detection snapshots, and the frontend forwards the highest-confidence detection above a configurable threshold to the VLM backend for response generation.

## Architecture

```text
backend_vision (C++ / OpenCV / qbruntime)
  - Runs the ARIES demo binary
  - Loads local YAML layout/model/feeder configs
  - Serves MJPEG stream and detection JSON on :8081

frontend (Next.js / React / MUI)
  - Displays the live vision stream on :3000
  - Polls /detections every 500 ms
  - Auto-triggers a VLM request when detection confidence exceeds threshold
  - Loads language-specific prompt bundles from public/prompt-bundles

backend_vlm (Flask-SocketIO / transformers)
  - Accepts ask/reset/prompt_config events on :5000
  - Loads Qwen2-VL-2B-Instruct
  - Uses Mobilint NPU when available, otherwise CUDA GPU
```

## Repository Layout

```text
.
|- frontend/             # Next.js UI
|- backend_vlm/          # Flask-SocketIO VLM server
|- backend_vision/       # Native C++ vision pipeline and configs
|- docker-compose.yml    # Main multi-service stack
|- docker-compose.gpu.yml
|- run.sh                # docker compose up --remove-orphans
|- stop.sh               # stops all Docker containers on the machine
`- update.sh             # install/build/setup script for Ubuntu-based targets
```

## Ports and Endpoints

- `3000`: frontend UI
- `5000`: VLM Socket.IO server
- `8081`: vision HTTP server

Vision backend endpoints:

- `GET /stream.mjpg`: MJPEG stream
- `GET /detections`: JSON detection snapshot
- `GET /healthz`: health check

## Current Runtime Behavior

### frontend

- Connects to `ws://<host>:5000`
- Reads the vision stream from `http://<host>:8081/stream.mjpg`
- Polls `http://<host>:8081/detections`
- Supports `en`, `ko`, `ja`, `zh` prompt bundles
- Lets the user change the detection threshold in the UI

### backend_vlm

- Entrypoint: `backend_vlm/src/server.py`
- Model pipeline: `backend_vlm/src/ImageTextToTextPipeline.py`
- Default model: `Qwen/Qwen2-VL-2B-Instruct`
- If Mobilint NPU is available, rewrites the model ID to `mobilint/...`
- If no NPU is available, requires CUDA; CPU-only execution is not supported by the current code path

### backend_vision

- Container command:

```bash
/workspace/build/src/demo/demo --http-port 8081
```

- Runs in headless mode when `--http-port` is used
- Loads these fixed config files at startup:
  - `backend_vision/rc/LayoutSetting_MLA100.yaml`
  - `backend_vision/rc/ModelSetting_MLA100.yaml`
  - `backend_vision/rc/FeederSetting_MLA100.yaml`
- Current sample assets include:
  - `backend_vision/mxq/yolo26s-weapon_uint8_input.mxq`
  - `backend_vision/rc/video/positive/*.mp4`
  - `backend_vision/rc/video/negative/*.mp4`

## Running the Demo

### Prerequisites

- Docker Engine with Docker Compose plugin
- Existing Docker network named `mblt_int`
- For `backend_vision`:
  - Mobilint runtime/device access on the host
  - built binary at `backend_vision/build/src/demo/demo`
- For `backend_vlm`:
  - Mobilint NPU or CUDA-capable GPU
  - model cache access

The main compose file mounts the native vision build output from the host:

```text
./backend_vision/build:/workspace/build:ro
```

If `backend_vision/build/src/demo/demo` does not exist, the vision container will not start correctly.

### Standard startup

```bash
docker compose up --remove-orphans
```

or:

```bash
./run.sh
```

Then open `http://localhost:3000`.

### GPU override for VLM backend

To use the GPU-specific VLM image override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --remove-orphans
```

The override changes:

- VLM Dockerfile from `backend.Dockerfile` to `backend-gpu.Dockerfile`
- `HF_HUB_OFFLINE=1` to `HF_HUB_OFFLINE=0`
- enables `gpus: all`

## Build and Setup Script

`update.sh` is the environment bootstrap script for Ubuntu-like deployment targets. It does much more than updating source code.

It currently performs all of the following:

- cleans frontend build artifacts and temporary JPG files
- installs kernel headers, build tools, Mobilint packages, Docker, and other dependencies
- configures Git credential cache
- runs `git pull`
- runs `git lfs pull` when `git-lfs` is available
- builds `backend_vision` with CMake and `make`
- creates a Python virtual environment for `backend_vlm`
- installs Python dependencies with `uv`
- downloads Mobilint VLM model snapshots into the local cache
- builds Docker images
- creates the external Docker network `mblt_int` if missing
- installs desktop entries and icons

Because it installs packages, configures system services, and modifies Docker/user-group state, treat it as a provisioning script rather than a simple project update.

## Operational Notes

- `run.sh` assumes the repository lives at `$HOME/aries-vision-vlm-demo` for the effective user.
- `stop.sh` runs `docker stop $(docker ps -a -q)`, which stops every container on the machine, not only this project's containers.
- `docker-compose.yml` uses `privileged: true` for both backends and mounts `/dev` into the containers.
- The compose stack expects the external network `mblt_int`; it is not auto-created by `docker compose up`.
- The frontend auto-arms again only after no detection remains above the current threshold.

## Development Notes

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### VLM backend

```bash
cd backend_vlm
uv venv
uv pip install -r pyproject.toml
python src/server.py
```

### Vision backend

```bash
cd backend_vision
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
./src/demo/demo --http-port 8081
```

The native demo reads `../rc/...` and `../mxq/...` relative to the working directory, so run it from `backend_vision/build`.
