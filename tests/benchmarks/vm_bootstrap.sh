#!/usr/bin/env bash
set -euo pipefail

BRANCH="feat/pufferlib-rl"
REPO_URL="${FLE_REPO_URL:-https://github.com/bordeauxred/factorio-learning-environment.git}"
REPO_DIR="${FLE_REPO_DIR:-${HOME}/factorio-learning-environment}"
BOOTSTRAP_USER="${SUDO_USER:-${USER:-$(id -un)}}"

sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git tmux htop curl ca-certificates
sudo usermod -aG docker "${BOOTSTRAP_USER}"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

if [[ -d "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" fetch origin "${BRANCH}"
  git -C "${REPO_DIR}" checkout "${BRANCH}"
  git -C "${REPO_DIR}" pull --ff-only origin "${BRANCH}"
elif [[ -e "${REPO_DIR}" ]]; then
  echo "Bootstrap stopped: ${REPO_DIR} exists but is not a git repository." >&2
  exit 1
else
  git clone --branch "${BRANCH}" --single-branch "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
uv venv .venv --python 3.12
uv pip install -e . --python .venv/bin/python

if ! .venv/bin/pytest -q tests/rl; then
  echo "Bootstrap stopped: offline RL tests failed; no cluster or sweep was started." >&2
  exit 1
fi

echo "Bootstrap complete. Log out and back in before using Docker if group membership is not active."
