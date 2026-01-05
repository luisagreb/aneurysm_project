#!/bin/bash
# Source this file to set up your nnUNet environment variables
# Usage: source src/setup_env.sh

export nnUNet_raw="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/luisa/aneurysm_project/data/nnUNet/nnUNet_results"

echo "nnUNet environment variables set:"
echo "  nnUNet_raw:          $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results:      $nnUNet_results"
