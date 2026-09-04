# ARIES Vision VLM Demo

`ARIES Vision VLM Demo` is a demo stack made up of four processes across three
services that connects:

- a native ARIES vision pipeline (`backend_vision`), one process
- two category-scoped VLM inference servers (`backend_vlm`), one process each
- a Next.js dashboard (`frontend`)

The vision backend runs weapon and fall detection concurrently on a single
8-channel layout and publishes an MJPEG stream plus JSON detection snapshots.
The frontend renders two VLM columns side by side — one per category — and
forwards each column's highest-confidence detection above the configured
threshold to the matching backend_vlm process for response generation.

## Architecture

```text
backend_vision (C++ / OpenCV / qbruntime)
  - Runs the ARIES demo binary
  - Loads local YAML layout/model/feeder configs
  - Runs weapon + fall vision models concurrently on one 8-channel layout
    (each channel is tagged with a `category` field on /detections)
  - Serves MJPEG stream and detection JSON on :8081

frontend (Next.js / React / MUI)
  - Displays the live vision stream on :3000
  - Polls /detections every 500 ms and splits channels by `category`
  - Runs two VLM columns concurrently, one per category:
      * weapon column -> ws://<host>:5000
      * fall   column -> ws://<host>:5001
  - Auto-triggers a VLM request per column when its category's top detection
    exceeds the configured threshold
  - Loads language-specific prompt bundles per category from
    public/prompt-bundles/{weapon_detection,fall_detection}/<lang>

backend_vlm (Flask-SocketIO / transformers)
  - Runs as two category-scoped processes (one image per process):
      * weapon on :5000 (owns mblt-tracker / system metrics)
      * fall   on :5001
  - Selected via `--category weapon|fall` on the CLI
  - Each process pins its vision encoder and text decoder to disjoint NPU
    cores per `core_allocation.yaml` so the two VLMs don't context-switch
  - Accepts ask/reset/prompt_config events over Socket.IO
  - Loads Qwen2-VL-2B-Instruct by default
  - Uses Mobilint NPU when available, otherwise CUDA GPU
```

## Repository Layout

```text
.
|- frontend/             # Next.js UI
|- backend_vlm/          # Flask-SocketIO VLM server (one process per category)
|- backend_vision/       # Native C++ vision pipeline and configs
|- core_allocation.yaml  # NPU core assignment for vision + VLM (per category)
|- docker-compose.yml    # Main multi-service stack
|- docker-compose.gpu.yml
|- download.sh           # downloads large vision assets from Hugging Face buckets
|- download.bat          # Windows version of the Hugging Face bucket downloader
|- run.sh                # docker compose up --remove-orphans
|- stop.sh               # stops all Docker containers on the machine
`- update.sh             # install/build/setup script for Ubuntu-based targets
```

## Ports and Endpoints

- `3000`: frontend UI
- `5000`: VLM Socket.IO server, weapon category
- `5001`: VLM Socket.IO server, fall category
- `8081`: vision HTTP server

Vision backend endpoints:

- `GET /stream.mjpg`: MJPEG stream (single 8-channel layout, weapon + fall side by side)
- `GET /detections`: JSON detection snapshot; each channel carries a `category` field
- `GET /layout`: current vision layout metadata
- `GET /healthz`: health check

There is no runtime mode-switching endpoint; both categories always stream in
parallel.

## NPU Core Allocation

`core_allocation.yaml` at the repo root is the single source of truth for how
the vision and VLM processes divide the ARIES NPU. It is mounted read-only into
every container at `/etc/aries/core_allocation.yaml`; the vision binary and the
two backend_vlm processes each read the entry for their own category (and, for
VLM, split their two submodules across the listed cores).

Schema:

```yaml
vision:
  <category>:                     # "weapon" | "fall"
    - {cluster: <int>, core: <int>}   # one or more cores; multi-threaded YOLO
    ...
vlm:
  <category>:                     # "weapon" | "fall"
    vision_target_cores: ["<cluster>:<core>", ...]  # single core, VLM vision encoder
    text_target_cores:   ["<cluster>:<core>", ...]  # single core, VLM text decoder
    vision_core_mode: "single"
    text_core_mode:   "single"
```

Rules that keep the four processes from stepping on each other:

- YOLO (vision) is multi-threaded and gets multiple cores; VLM submodules are
  single-request and get one core each.
- The VLM `vision` and `text` submodules pin to separate cores per category
  (mblt-model-zoo takes two kwargs, not one).
- All four categories (vision.weapon, vision.fall, vlm.weapon, vlm.fall) use
  disjoint NPU cores so nothing context-switches on a shared slot.

Override the path with `ARIES_CORE_ALLOCATION_PATH` (both C++ and Python
loaders honor it). Local runs outside Docker fall back to a repo-relative
`core_allocation.yaml`.

## Current Runtime Behavior

### frontend

- Renders two VLM columns side by side (weapon left, fall right). There is no
  mode selector: both columns are always live and share the same 8-channel
  vision stream.
- Opens two Socket.IO connections, one per column: `ws://<host>:5000` (weapon)
  and `ws://<host>:5001` (fall).
- Reads the vision stream from `http://<host>:8081/stream.mjpg`
- Polls `http://<host>:8081/detections` and splits channels by their `category`
  field so each column only sees its own detections.
- Loads category-specific prompt bundles from
  `public/prompt-bundles/{weapon_detection,fall_detection}/<lang>` on start
  and pushes each one to its column's VLM process.
- Supports `en`, `ko`, `ja`, `zh` prompt bundles
- Lets the user change the detection threshold in the UI (applies to both columns)
- Auto-trigger eligibility is category-configurable in `frontend/app/settings.ts`
  and respects the UI detection threshold by default:
  - `weapon`: any label is eligible when its confidence is above the current UI threshold
  - `fall`: only label index `0` (`falling`) is eligible when its confidence is above the current UI threshold

### backend_vlm

- Entrypoint: `backend_vlm/src/server.py`, one process per category
- Selected via `--category weapon|fall` and `--port <5000|5001>`
- Model pipeline: `backend_vlm/src/ImageTextToTextPipeline.py`
- Default model: `Qwen/Qwen2-VL-2B-Instruct`
- If Mobilint NPU is available, rewrites the model ID to `mobilint/...` and
  pins the vision encoder and text decoder to the disjoint core slots in
  `core_allocation.yaml` for its category.
- Only the `weapon` process initializes `mblt-tracker` (single NPU driver
  handle). The `fall` process emits an unavailable envelope on
  `system_metrics:get`, and the frontend surfaces system metrics from the
  weapon connection.
- If no NPU is available, requires CUDA; CPU-only execution is not supported by the current code path

### backend_vision

- Container command:

```bash
/workspace/build/src/demo/demo --http-port 8081
```

- Runs in headless mode when `--http-port` is used
- Loads these fixed config files at startup:
  - `backend_vision/assets/config/LayoutSetting_MLA100.yaml`
  - `backend_vision/assets/config/ModelSetting_MLA100.yaml`
  - `backend_vision/assets/config/FeederSetting_MLA100.yaml`
- Model YAML supports `pipeline_config` for detection post-processing and overlay rendering:
  - `labels`: label names used for overlay text and `/detections` metadata
  - `conf_threshold`, `iou_threshold`, `display_confidence_threshold`
  - `bbox_thickness`, `draw_label_text`, `draw_score_text`, `draw_detection_border`
  - legacy top-level `labels` remains supported
- Current sample assets include:
  - `backend_vision/assets/mxq/yolo26s-weapon_uint8_input_260513.mxq`
  - `backend_vision/assets/layout/*.png`
  - `backend_vision/assets/video/positive/*.mp4`
  - `backend_vision/assets/video/negative/*.mp4`

## Running the Demo

### Prerequisites

- Docker Engine with Docker Compose plugin
- Existing Docker network named `mblt_int`
- For `backend_vision`:
  - Mobilint runtime/device access on the host
- For `backend_vlm`:
  - Mobilint NPU or CUDA-capable GPU
  - model cache access

Large vision assets are no longer stored with Git LFS. They are downloaded from Hugging Face buckets into `backend_vision/assets` before building or running the demo. The required asset layout is:

- `backend_vision/assets/config/*.yaml`
- `backend_vision/assets/layout/*`
- `backend_vision/assets/mxq/*`
- `backend_vision/assets/video/**/*.mp4`
- `backend_vision/assets/fall/mxq/*`
- `backend_vision/assets/fall/video/*.mp4`

Use the provided downloader for your platform:

```bash
./download.sh
```

```bat
download.bat
```

The downloader uses `uv` to create a local `.hf_venv`, installs `huggingface-hub`, and runs `hf buckets sync` from two default buckets:

- Weapon detection: `mobilint/aries-weapon-detection-demo-assets`
- Fall detection: `mobilint/aries-fall-detection-demo-assets`

```text
hf://buckets/mobilint/aries-weapon-detection-demo-assets
```

To override the source bucket or downloader virtualenv location, set:

- `HF_ASSET_BUCKET_ID`: backward-compatible weapon bucket override
- `HF_WEAPON_ASSET_BUCKET_ID`: weapon Hugging Face bucket ID, default `mobilint/aries-weapon-detection-demo-assets`
- `HF_FALL_ASSET_BUCKET_ID`: fall Hugging Face bucket ID, default `mobilint/aries-fall-detection-demo-assets`
- `HF_DOWNLOAD_VENV_DIR`: virtualenv path for the downloader, default `.hf_venv`

The vision binary is built inside `backend_vision/vision.Dockerfile` and copied into the runtime image. This keeps the OpenCV libraries used at build time and runtime consistent across Ubuntu host versions.

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
- runs `download.sh` to sync large vision assets from Hugging Face buckets
- creates a Python virtual environment for `backend_vlm`
- installs Python dependencies with `uv`
- downloads Mobilint VLM model snapshots into the local cache
- builds Docker images, including the native `backend_vision` binary inside Docker
- creates the external Docker network `mblt_int` if missing
- installs desktop entries and icons

Because it installs packages, configures system services, and modifies Docker/user-group state, treat it as a provisioning script rather than a simple project update.

## Operational Notes

- `run.sh` assumes the repository lives at `$HOME/aries-vision-vlm-demo` for the effective user.
- `stop.sh` runs `docker stop $(docker ps -a -q)`, which stops every container on the machine, not only this project's containers.
- `docker-compose.yml` uses `privileged: true` for both backends and mounts `/dev` into the containers.
- `docker compose up` starts four services: `aries_vision_vlm_frontend`, `aries_vision_vlm_backend_vision`, and two VLM services (`aries_vision_vlm_backend_vlm_weapon` on :5000 and `aries_vision_vlm_backend_vlm_fall` on :5001). Both VLM services share one built image and the host HF cache.
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

Two processes must run for the full demo — one per category:

```bash
cd backend_vlm
uv venv
uv pip install -r pyproject.toml
python src/server.py --category weapon --port 5000
python src/server.py --category fall   --port 5001
```

Outside Docker, both processes fall back to reading `core_allocation.yaml`
from the repo root; override with `--core-allocation-path` or
`ARIES_CORE_ALLOCATION_PATH`.

### Vision backend

#### Linux

```bash
cd backend_vision
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
./src/demo/demo --http-port 8081
```

The native demo reads `../assets/...` relative to the working directory, so run it from `backend_vision/build`.

#### Windows (Visual Studio + CMake)

The `CMakeLists.txt` already handles MSVC (UTF-8 source, `ws2_32` link, OpenCV runtime DLL copy including the FFmpeg backend); it just needs the paths to prebuilt `qbruntime` and `OpenCV` since neither has a system package on Windows.

Prerequisites:

- Visual Studio 2019 or 2022 with the *Desktop development with C++* workload (includes MSVC, Windows SDK, and CMake)
- `qbruntime` Windows distribution laid out as `<root>/include/`, `<root>/lib/qbruntime.lib` (and `qbruntimed.lib` for Debug), `<root>/bin/qbruntime.dll` (and `qbruntimed.dll` for Debug)
- Prebuilt OpenCV for Windows (e.g. `C:\opencv\build` from the official Windows release); note the `vcXX` folder that matches your Visual Studio (`vc16` = VS 2019, `vc17` = VS 2022)
- Mobilint MLA driver for Windows must be installed on the host at runtime, otherwise the exe will start but `qbruntime` calls will fail

Configure and build from a *Developer PowerShell for VS*:

```powershell
cd backend_vision
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 `
  -DQBRUNTIME_PATH="C:\path\to\qbruntime" `
  -DOpenCV_DIR="C:\path\to\opencv\build"
cmake --build . --config Release
```

Adjust the generator to `"Visual Studio 16 2019"` if you are on VS 2019. The build produces `build\src\demo\Release\demo.exe`, and CMake's post-build step copies `qbruntime.dll`, `opencv_worldXXX.dll`, and the versioned `opencv_videoio_ffmpegXXX_64.dll` next to it so the exe is portable inside the `Release/` folder.

Alternatively, open the folder in Visual Studio itself (*File → Open → CMake…* on `backend_vision/CMakeLists.txt`) and set `QBRUNTIME_PATH` / `OpenCV_DIR` in *CMake Settings*; the same targets are available in the Solution Explorer.

Run it:

```powershell
cd backend_vision\build
set ARIES_CORE_ALLOCATION_PATH=%CD%\..\..\core_allocation.yaml
.\src\demo\Release\demo.exe --http-port 8081
```

The `--http-port` mode reads `..\..\assets\...` relative to `CMAKE_BINARY_DIR` (which VS sets as the debugger working directory), and `ARIES_CORE_ALLOCATION_PATH` points the exe at the repo-root `core_allocation.yaml` that `run.sh` normally materializes at boot. If you are running only `backend_vision` in isolation (no `run.sh`), copy one of the profiles manually first — `copy core_allocation.1card.yaml core_allocation.yaml` or `.2card.yaml`, matching the number of MLA cards attached.

> Docker Desktop on Windows cannot pass MLA100 through to WSL2 containers, so the full stack cannot run under `docker compose` on Windows. `backend_vlm` and `frontend` also need to be started as native processes (Python venv and `npm run dev` respectively) if you want the whole demo on a Windows host.
