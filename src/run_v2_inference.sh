#!/bin/bash
# Script to run V2 Inference for Actin and Mitochondria
# Usage: nohup ./src/run_v2_inference.sh > inference_v2.log 2>&1 &

# 1. Activate Environment
source /home/luisa/aneurysm_project/venv/bin/activate
source /home/luisa/aneurysm_project/src/setup_env.sh

BASE_OUT="/home/luisa/aneurysm_project/experiments/V2/inference_results"
mkdir -p "$BASE_OUT"

echo "========================================"
echo "Starting V2 Inference"
echo "Date: $(date)"
echo "========================================"

# 2. Inference Actin (Dataset 001)
INPUT_001="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw/Dataset001_Actin/imagesTs"
OUTPUT_001="$BASE_OUT/Dataset001_Actin"
mkdir -p "$OUTPUT_001"

if [ -d "$INPUT_001" ] && [ "$(ls -A $INPUT_001)" ]; then
    echo ""
    echo ">>> Running Inference for Actin (Dataset 001)"
    echo "Input: $INPUT_001"
    echo "Output: $OUTPUT_001"
    
    nnUNetv2_predict -d 001 -i "$INPUT_001" -o "$OUTPUT_001" -f  0 -c 3d_fullres --save_probabilities
    
    if [ $? -eq 0 ]; then
        echo "✅ Actin Inference Complete"
    else
        echo "❌ Actin Inference Failed"
    fi
else
    echo "⚠️  Skipping Actin: Input directory empty or missing."
fi

# 3. Inference Mito (Dataset 002)
INPUT_002="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw/Dataset002_Mito/imagesTs"
OUTPUT_002="$BASE_OUT/Dataset002_Mito"
mkdir -p "$OUTPUT_002"

if [ -d "$INPUT_002" ] && [ "$(ls -A $INPUT_002)" ]; then
    echo ""
    echo ">>> Running Inference for Mitochondria (Dataset 002)"
    echo "Input: $INPUT_002"
    echo "Output: $OUTPUT_002"
    
    nnUNetv2_predict -d 002 -i "$INPUT_002" -o "$OUTPUT_002" -f  0 -c 3d_fullres --save_probabilities
    
    if [ $? -eq 0 ]; then
        echo "✅ Mito Inference Complete"
    else
        echo "❌ Mito Inference Failed"
    fi
else
    echo "⚠️  Skipping Mito: Input directory empty or missing."
fi

echo ""
echo "========================================"
echo "Inference Run Finished"
echo "Date: $(date)"
echo "========================================"
