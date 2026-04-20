#!/bin/bash
set -euo pipefail

DEMO_NAME="ARIES Vision VLM Demo"
DEMO_DIR_NAME="aries-vision-vlm-demo"

if [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
else
  USER_HOME="$HOME"
fi

if [ "$EUID" -eq 0 ] && [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  RUN_AS_USER="$SUDO_USER"
else
  RUN_AS_USER="$USER"
fi

run_as_user() {
  if [ "$EUID" -eq 0 ] && [ "${SUDO_USER-}" ] && [ "$RUN_AS_USER" != "root" ]; then
    sudo -H -u "$RUN_AS_USER" "$@"
  else
    "$@"
  fi
}

APP_DIR="$USER_HOME/$DEMO_DIR_NAME"

cd "$APP_DIR"

# Green banner text
printf '\033[1;32m=========== %s ============\033[0m\n' "$DEMO_NAME"

sudo rm -rf frontend/.next  # clean up old build files
sudo rm -rf frontend/node_modules  # clean up old dependencies
sudo rm -rf frontend/next-env.d.ts  # clean up old env file
sudo rm -rf backend_vlm/src/*.jpg # clean up old temp images
sudo rm -rf ~/.mblt_model_zoo # clean up old cache folder

sudo apt install -y linux-headers-$(uname -r) build-essential

# Add Mobilint's official GPG key:
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://dl.mobilint.com/apt/gpg.pub -o /etc/apt/keyrings/mblt.asc
sudo chmod a+r /etc/apt/keyrings/mblt.asc

# Add the repository to apt sources:
printf "%s\n" \
  "deb [signed-by=/etc/apt/keyrings/mblt.asc] https://dl.mobilint.com/apt \
    stable multiverse" |
  sudo tee /etc/apt/sources.list.d/mobilint.list >/dev/null

# Update available packages
sudo apt update

# Install driver & utilities
sudo apt install -y mobilint-aries-driver mobilint-qb-runtime mobilint-cli

CREDENTIAL_TIMEOUT="${GIT_CREDENTIAL_TIMEOUT:-3600}"

echo "Configuring Git credential cache (timeout: ${CREDENTIAL_TIMEOUT}s)..."
if ! git config --global credential.helper "cache --timeout=${CREDENTIAL_TIMEOUT}"; then
  echo "Failed to configure Git credential cache for user $USER."
  exit 1
fi

if [ "$RUN_AS_USER" != "$USER" ]; then
  if ! run_as_user git config --global credential.helper "cache --timeout=${CREDENTIAL_TIMEOUT}"; then
    echo "Failed to configure Git credential cache for user $RUN_AS_USER."
    exit 1
  fi
fi

run_as_user git pull

echo "Preparing build..."

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs not found. Install git-lfs to fetch LFS assets (mxq/mp4)."
else
  echo "Pulling LFS assets (mxq/mp4) from repository..."
  run_as_user git lfs pull
fi

echo "Installing system dependency (libopencv-dev)..."
if ! sudo apt-get install -y libopencv-dev; then
  echo "Failed to install libopencv-dev. Please resolve the apt issue and rerun."
  exit 1
fi

echo "Installing build dependency (cmake)..."
if ! sudo apt-get install -y cmake; then
  echo "Failed to install cmake. Please resolve the apt issue and rerun."
  exit 1
fi

echo "Starting backend_vision native build..."

if [ ! -d backend_vision/build ]; then
  run_as_user mkdir -p backend_vision/build
fi

cd backend_vision/build

if run_as_user cmake ..; then
  echo "CMake completed successfully."
else
  echo "CMake failed."
  exit 1
fi

if run_as_user make -j $(nproc); then
  echo "Build completed successfully."
else
  echo "Build failed."
  exit 1
fi

cd "$APP_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  run_as_user bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if [ -d "$USER_HOME/.local/bin" ]; then
  export PATH="$USER_HOME/.local/bin:$PATH"
fi

# model_name
MODELS=(
  "Qwen2-VL-2B-Instruct"
)

echo "Preparing backend venv and downloading model..."

CACHE_DIR="$USER_HOME/.cache/"
mkdir -p "$CACHE_DIR"
sudo chown -R "$RUN_AS_USER:$RUN_AS_USER" "$CACHE_DIR"

pushd backend_vlm >/dev/null
  if [ ! -d ".venv" ]; then
    run_as_user uv venv
  fi
  run_as_user uv pip install -r pyproject.toml
  for model in "${MODELS[@]}"; do
    run_as_user uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('mobilint/${model}')"
  done
popd >/dev/null

if ! command -v docker >/dev/null 2>&1; then
  echo "Cannot find docker. Installing..."

  # Add Docker's official GPG key:
  sudo apt-get update
  sudo apt-get install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  # Add the repository to Apt sources:
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update

  sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  echo "Docker installed successfully."
fi

echo "Starting Docker build..."
if sudo docker compose build; then
  echo "Docker build completed successfully."
else
  echo "Docker build failed."
  exit 1
fi

docker builder prune -f || true

NETWORK_NAME="mblt_int"

if sudo docker network inspect "$NETWORK_NAME" >/dev/null 2>&1; then
  echo "Docker network '$NETWORK_NAME' already exists."
else
  echo "Docker network '$NETWORK_NAME' not found. Creating..."
  sudo docker network create "$NETWORK_NAME"
  echo "Docker network '$NETWORK_NAME' created."
fi

if ! docker ps >/dev/null 2>&1; then
  echo "Cannot run docker command without sudo. Adding user \`$USER\` to docker group..."

  if sudo usermod -aG docker "$USER"; then
    echo "Docker group updated successfully."
    printf '\033[1;31m!!!!! [WARNING] YOU SHOULD REBOOT MACHINE TO USE DEMO APPROPRIATELY !!!!!\033[0m\n' "$DEMO_NAME"
    echo "Please log out and log back in (or reboot) so the new docker group membership takes effect."
  else
    echo "Failed to update docker group membership for $USER."
    exit 1
  fi
fi

echo "Updating desktop shortcut..."
# delete old desktop file
if [ -f /usr/share/applications/vision-vlm-demo.desktop ]; then
  sudo rm /usr/share/applications/vision-vlm-demo.desktop
fi
if [ -f /usr/share/applications/vision-vlm-demo.desktop ]; then
  sudo rm /usr/share/applications/vision-vlm-demo.desktop
fi
sudo mkdir -p "$USER_HOME/.local/share/applications/" || {
  echo "Failed to create desktop directory at $USER_HOME/.local/share/applications/."
  exit 1
}
sudo cp *.desktop "$USER_HOME/.local/share/applications/"
if [ $? -eq 0 ]; then
  echo "Updating desktop shortcut completed successfully."
else
  echo "Updating desktop shortcut failed."
  exit 1
fi

echo "Updating desktop icon..."
sudo mkdir -p "$USER_HOME/.icons/" || {
  echo "Failed to create icon directory at $USER_HOME/.icons/."
  exit 1
}
sudo cp *.png "$USER_HOME/.icons/"
if [ $? -eq 0 ]; then
  echo "Updating desktop icon completed successfully."
else
  echo "Updating desktop icon failed."
  exit 1
fi

exit 0
