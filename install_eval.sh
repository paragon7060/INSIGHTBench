#!/usr/bin/env bash
# InsightBench evaluation-only environment setup.
#
# Installs everything needed to RUN benchmark evaluations.
# Does NOT install CuRobo — evaluation policies do not require motion planning.
#
# For data collection and interact scenes (which need CuRobo), use install.sh instead.
#
# Usage:
#   bash install_eval.sh [ENV_NAME]    # default: env_insightbench_eval
#
# Default layout:
#   parent/
#     InsightBench/
#     IsaacLab/
#     lerobot/
#
# Optional overrides:
#   ISAACLAB_ROOT=/custom/path/to/IsaacLab
#   LEROBOT_ROOT=/custom/path/to/lerobot
#   LEROBOT_EXTRAS=pi,smolvla,groot
#
# Recommended LeRobot baseline for unified Pi0/Diffusion/SmolVLA/GR00T eval:
#   git clone https://github.com/huggingface/lerobot.git /path/to/lerobot
#   git -C /path/to/lerobot checkout f6b16f6d97155e3ce34ab2a1ec145e9413588197
#
# After install, activate with:
#   conda activate ENV_NAME

set -e

ENV_NAME="${1:-env_insightbench_eval}"
INSIGHTBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${INSIGHTBENCH_ROOT}/.." && pwd)"
DEFAULT_ISAACLAB_ROOT="${WORKSPACE_ROOT}/IsaacLab"
DEFAULT_LEROBOT_ROOT="${WORKSPACE_ROOT}/lerobot"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${DEFAULT_ISAACLAB_ROOT}}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${DEFAULT_LEROBOT_ROOT}}"
LEROBOT_EXTRAS="${LEROBOT_EXTRAS:-pi,smolvla,groot}"
LEROBOT_RECOMMENDED_URL="https://github.com/huggingface/lerobot.git"
LEROBOT_RECOMMENDED_COMMIT="f6b16f6d97155e3ce34ab2a1ec145e9413588197"
MINICONDA_ROOT="${MINICONDA_ROOT:-${HOME}/miniconda3}"
CONDA_BIN="${MINICONDA_ROOT}/bin/conda"

require_dir() {
    local path="$1"
    local name="$2"
    local hint="$3"

    if [[ ! -d "${path}" ]]; then
        echo "ERROR: ${name} does not exist: ${path}" >&2
        echo "       ${hint}" >&2
        exit 1
    fi
}

require_file() {
    local path="$1"
    local name="$2"
    local hint="$3"

    if [[ ! -f "${path}" ]]; then
        echo "ERROR: ${name} does not exist: ${path}" >&2
        echo "       ${hint}" >&2
        exit 1
    fi
}

require_executable() {
    local path="$1"
    local name="$2"
    local hint="$3"

    if [[ ! -x "${path}" ]]; then
        echo "ERROR: ${name} is not executable or does not exist: ${path}" >&2
        echo "       ${hint}" >&2
        exit 1
    fi
}

require_dir "${ISAACLAB_ROOT}" "ISAACLAB_ROOT" "Expected repo-adjacent IsaacLab at ${DEFAULT_ISAACLAB_ROOT}; clone it there or set ISAACLAB_ROOT=/custom/path/to/IsaacLab."
require_file "${ISAACLAB_ROOT}/_isaac_sim/setup_conda_env.sh" "Isaac Sim conda setup script" "Check that ISAACLAB_ROOT points to an IsaacLab checkout with Isaac Sim installed."
require_dir "${ISAACLAB_ROOT}/source/isaaclab" "Isaac Lab source package" "Check that ISAACLAB_ROOT points to a complete IsaacLab checkout."
require_dir "${ISAACLAB_ROOT}/source/isaaclab_assets" "Isaac Lab assets package" "Check that ISAACLAB_ROOT points to a complete IsaacLab checkout."
require_dir "${ISAACLAB_ROOT}/source/isaaclab_tasks" "Isaac Lab tasks package" "Check that ISAACLAB_ROOT points to a complete IsaacLab checkout."
require_dir "${LEROBOT_ROOT}" "LeRobot checkout" "Expected repo-adjacent LeRobot at ${DEFAULT_LEROBOT_ROOT}; clone ${LEROBOT_RECOMMENDED_URL} there and checkout ${LEROBOT_RECOMMENDED_COMMIT}, or set LEROBOT_ROOT=/custom/path/to/lerobot."
require_dir "${LEROBOT_ROOT}/src/lerobot/policies/pi0" "LeRobot Pi0 policy" "Use the unified LeRobot baseline with LEROBOT_EXTRAS including pi."
require_dir "${LEROBOT_ROOT}/src/lerobot/policies/diffusion" "LeRobot Diffusion policy" "Use ${LEROBOT_RECOMMENDED_URL}@${LEROBOT_RECOMMENDED_COMMIT}."
require_dir "${LEROBOT_ROOT}/src/lerobot/policies/smolvla" "LeRobot SmolVLA policy" "Use the unified LeRobot baseline with LEROBOT_EXTRAS including smolvla."
require_dir "${LEROBOT_ROOT}/src/lerobot/policies/groot" "LeRobot GR00T policy" "Use ${LEROBOT_RECOMMENDED_URL}@${LEROBOT_RECOMMENDED_COMMIT} with LEROBOT_EXTRAS including groot."
require_dir "${MINICONDA_ROOT}" "MINICONDA_ROOT" "Set MINICONDA_ROOT=/custom/path/to/miniconda3 and rerun."
require_executable "${CONDA_BIN}" "conda executable" "Check MINICONDA_ROOT or set MINICONDA_ROOT=/path/to/miniconda3."

echo "============================================================"
echo " InsightBench  [EVAL ONLY]  environment setup"
echo " Conda env    : ${ENV_NAME}"
echo " IsaacLab     : ${ISAACLAB_ROOT}"
echo " LeRobot      : ${LEROBOT_ROOT}"
echo " LeRobot extra: ${LEROBOT_EXTRAS}"
echo " LeRobot base : ${LEROBOT_RECOMMENDED_URL}@${LEROBOT_RECOMMENDED_COMMIT}"
echo " InsightBench : ${INSIGHTBENCH_ROOT}"
echo " Miniconda    : ${MINICONDA_ROOT}"
echo " (CuRobo NOT installed — use install.sh for full install)"
echo "============================================================"

# ─── 1. Create conda env ──────────────────────────────────────────────────────
if "${CONDA_BIN}" env list | grep -q "^${ENV_NAME} "; then
    echo "[SKIP] Conda env '${ENV_NAME}' already exists."
else
    echo "[1/5] Creating conda env: ${ENV_NAME} (Python 3.11)"
    "${CONDA_BIN}" create -y -n "${ENV_NAME}" python=3.11
fi

CONDA_PIP="${MINICONDA_ROOT}/envs/${ENV_NAME}/bin/pip"
CONDA_PYTHON="${MINICONDA_ROOT}/envs/${ENV_NAME}/bin/python"

# ─── 2. Add Isaac Sim to the conda env (via activate hook) ───────────────────
ACTIVATE_DIR="${MINICONDA_ROOT}/envs/${ENV_NAME}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/isaac_sim_paths.sh" << EOF
#!/usr/bin/env bash
# Auto-generated by InsightBench install_eval.sh
source ${ISAACLAB_ROOT}/_isaac_sim/setup_conda_env.sh
# libstdc++ fix for Isaac Sim linker errors:
export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH}
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# IsaacLab root for asset paths (used by franka.py and scene configs)
export ISAACLAB_PATH=${ISAACLAB_ROOT}
EOF

echo "[2/5] Isaac Sim activate hook written."

# ─── 3. Install PyTorch (cu128 build — works with CUDA driver 12.4+) ──────────
echo "[3/5] Installing PyTorch 2.7.0+cu128"
${CONDA_PIP} install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --quiet

# ─── 4. Install Isaac Lab (editable) ──────────────────────────────────────────
# Bootstrap the validated IsaacLab-safe metadata/runtime pins before editable
# resolution. In particular, do not let IsaacLab metadata resolve against NumPy 2.
# Keep the legacy-compatible setuptools bootstrap scoped to these IsaacLab
# editable installs. LeRobot requires a newer setuptools version below.
# toml is imported by IsaacLab setup.py during editable metadata generation.
${CONDA_PIP} install "numpy==1.26.4" "pillow==11.2.1" "setuptools<70" toml --quiet
echo "[4/5] Installing Isaac Lab packages (editable)"
${CONDA_PIP} install -e "${ISAACLAB_ROOT}/source/isaaclab" --no-build-isolation --quiet
${CONDA_PIP} install -e "${ISAACLAB_ROOT}/source/isaaclab_assets" --no-build-isolation --quiet
${CONDA_PIP} install -e "${ISAACLAB_ROOT}/source/isaaclab_tasks" --no-build-isolation --quiet

# ─── 5. Install LeRobot checkout + InsightBench + pinned deps ─────────────────
echo "[5/5] Installing LeRobot checkout and InsightBench"
# LeRobot 0.4.1 requires setuptools>=71,<81; this version was validated with
# IsaacLab and flatdict==4.0.1 after the editable metadata bootstrap above.
${CONDA_PIP} install "setuptools==80.9.0" --quiet
# Pi's flash-attn setup runs in this runtime env: it imports packaging/psutil
# and declares ninja as a setup prerequisite before the Pi extra can install.
${CONDA_PIP} install psutil packaging ninja --quiet
# Pi's flash-attn build must import the runtime torch installed above.
${CONDA_PIP} install -e "${LEROBOT_ROOT}[${LEROBOT_EXTRAS}]" --no-build-isolation --quiet
${CONDA_PIP} install -e "${INSIGHTBENCH_ROOT}" --quiet

# Pin versions to match the validated IsaacLab + LeRobot runtime:
#   numpy<2 and gymnasium==1.2.0 are required by IsaacLab;
#   opencv-python-headless is LeRobot's declared OpenCV dependency, pinned to
#   the NumPy 1.x-compatible version used in the validated unified env;
#   transformers==4.57.1 is required by GR00T's Eagle processor; InsightBench
#   adds the narrow OpenPI SigLIP compatibility shim for Pi0 after installation.
${CONDA_PIP} install \
    "numpy==1.26.4" \
    "gymnasium==1.2.0" \
    "opencv-python-headless==4.11.0.86" \
    "pillow==11.2.1" \
    "transformers==4.57.1" \
    omegaconf packaging h5py botocore click --quiet
${CONDA_PYTHON} -c "import transformers; from insightbench.utils.pi0_siglip_compat import install_pi0_siglip_compat; install_pi0_siglip_compat(); from transformers.models.siglip import check; from lerobot.policies.pi0.modeling_pi0 import PI0Policy; from lerobot.policies.groot.eagle2_hg_model.modeling_eagle2_5_vl import Eagle25VLForConditionalGeneration; assert transformers.__version__ == '4.57.1'; assert check.check_whether_transformers_replace_is_installed_correctly(); print('Pi0 + GR00T Transformers contract OK')"
# LeRobot 0.4.1 declares rerun-sdk for visualization tooling. Its current
# compatible range declares numpy>=2, so pip check may report a metadata
# conflict with IsaacLab's numpy<2 runtime pin; policy evaluation does not
# import rerun. See README dependency hygiene notes.

echo ""
echo "============================================================"
echo " DONE. Activate with:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo " Verify:"
echo "   python -c \"from isaaclab.app import AppLauncher; print('OK')\""
echo "   python -c \"import transformers; print(transformers.__version__)\""
echo ""
echo " Run evaluation:"
echo "   python scripts/evaluate.py --config configs/eval/pi0.yaml \\"
echo "       --object door --asset_path <ASSET_ID> --task_idx 0 \\"
echo "       policy.checkpoint=your-hf-user-or-org/pi0-repo \\"
echo "       --num_envs 8 --headless --enable_cameras"
echo "============================================================"
