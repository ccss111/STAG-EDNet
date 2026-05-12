#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/CMAPSS-release"
cd "$PROJECT_ROOT"

DATASET="FD002"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
METHOD_CODES="${METHOD_CODES:-A B C D}"

# Shared hyper-parameters for A-D ablation runs.
COMMON_ARGS=(
    --sub-dataset "$DATASET"
    --gat-num-layers 2
    --gat-embed-dim 8
    --gat-topk 7
    --lr-scheduler step
    --lr 0.002
    --disable-aof
)

method_enabled() {
    local target="$1"
    for code in $METHOD_CODES; do
        if [[ "$code" == "$target" ]]; then
            return 0
        fi
    done
    return 1
}

run_ablation() {
    local code="$1"
    local desc="$2"

    echo "====================================="
    echo "开始运行方法 ${code}: ${desc}"
    echo "数据集: ${DATASET}"
    echo "====================================="

    PYTHONPATH="$PROJECT_ROOT" python scipt/train_model.py \
        "${COMMON_ARGS[@]}" \
        --apply-code-ablation \
        --model-code "${DATASET}_${code}_${RUN_TAG}"
}

# (A) KNN graph + GAT-LSTM with encoder and decoder
if method_enabled "A"; then
    run_ablation "A" "KNN graph + GAT-LSTM with encoder and decoder"
fi

# (B) Full-connected graph + GAT-LSTM with encoder and decoder
if method_enabled "B"; then
    run_ablation "B" "Full-connected graph + GAT-LSTM with encoder and decoder"
fi

# (C) KNN graph + GAT-LSTM with encoder only
if method_enabled "C"; then
    run_ablation "C" "KNN graph + GAT-LSTM with encoder only"
fi

# (D) Original GAT-LSTM without encoder and decoder
if method_enabled "D"; then
    run_ablation "D" "Original GAT-LSTM without encoder and decoder"
fi

echo "FD002 的 A-D 消融实验已执行完成。RUN_TAG=${RUN_TAG}, METHOD_CODES=${METHOD_CODES}"