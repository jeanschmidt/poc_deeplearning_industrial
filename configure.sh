#!/bin/bash

PIP_TORCH_URL="http://download.pytorch.org/whl/cu75/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl"

function stop_at_fail() {
    echo "${1}"
    exit 1
}

pushd data/contrast || stop_at_fail "Failed to change directory"
./generate.sh || stop_at_fail "Failed to generate images"
popd || stop_at_fail "Failed to cd back"
virtualenv venv || stop_at_fail "Failed to create venv Virtualenv"
venv/bin/pip install -r requirements.txt || stop_at_fail "Failed to install basic requirements.txt"
venv/bin/pip install "${PIP_TORCH_URL}" || stop_at_fail "Failed to install pytorch from URL"
venv/bin/pip install torchvision || stop_at_fail "Failed to install torchvision"

echo "All should be fine"
