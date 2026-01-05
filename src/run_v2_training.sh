#!/bin/bash
# Script to run V2 Training Sequentially
# Usage: nohup ./src/run_v2_training.sh > training_log.txt 2>&1 &

# 1. Activate Environment
source /home/luisa/aneurysm_project/venv/bin/activate
source /home/luisa/aneurysm_project/src/setup_env.sh

echo "========================================"
echo "Starting Sequential Training for V2"
echo "Date: $(date)"
echo "========================================"

# 2. Train Actin (Dataset 001)
echo ""
echo ">>> Starting Dataset 001: Actin"
nnUNetv2_train 001 3d_fullres 0
if [ $? -eq 0 ]; then
    echo "✅ Actin Training Complete"
else
    echo "❌ Actin Training Failed"
    exit 1
fi

# 3. Train Mito (Dataset 002)
echo ""
echo ">>> Starting Dataset 002: Mitochondria"
nnUNetv2_train 002 3d_fullres 0
if [ $? -eq 0 ]; then
    echo "✅ Mitochondria Training Complete"
else
    echo "❌ Mitochondria Training Failed"
    exit 1
fi

# 4. Train Nucleus (Dataset 003)
echo ""
echo ">>> Starting Dataset 003: Nucleus"
nnUNetv2_train 003 3d_fullres 0
if [ $? -eq 0 ]; then
    echo "✅ Nucleus Training Complete"
else
    echo "❌ Nucleus Training Failed"
    exit 1
fi

echo ""
echo "========================================"
echo "All Trainings Completed Successfully!"
echo "Date: $(date)"
echo "========================================"
