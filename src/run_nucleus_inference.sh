#!/bin/bash
# Script to run Inference for Nucleus (Dataset 003)

# 1. Activate Environment
source /home/luisa/aneurysm_project/venv/bin/activate
source /home/luisa/aneurysm_project/src/setup_env.sh

BASE_OUT="/home/luisa/aneurysm_project/experiments/V2/inference_results"
mkdir -p "$BASE_OUT"

# 2. Inference Nucleus (Dataset 003)
INPUT_003="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw/Dataset003_Nucleus/imagesTs"
OUTPUT_003="$BASE_OUT/Dataset003_Nucleus"
mkdir -p "$OUTPUT_003"

echo "========================================"
echo "Starting Nucleus Inference"
echo "Date: $(date)"
echo "Input: $INPUT_003"
echo "Output: $OUTPUT_003"
echo "========================================"

if [ -d "$INPUT_003" ] && [ "$(ls -A $INPUT_003)" ]; then
    # Run prediction
    # -d 003: Dataset ID
    # -f 0: Fold 0
    # -c 3d_fullres: Configuration
    nnUNetv2_predict -d 003 -i "$INPUT_003" -o "$OUTPUT_003" -f 0 -c 3d_fullres --save_probabilities
    
    if [ $? -eq 0 ]; then
        echo "✅ Nucleus Inference Complete"
    else
        echo "❌ Nucleus Inference Failed"
    fi
else
    echo "⚠️  Skipping Nucleus: Input directory empty or missing."
fi

echo "========================================"
echo "Finished"
echo "Date: $(date)"
echo "========================================"
