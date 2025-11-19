#!/bin/bash
# Run this script on your RTX 3090 to test all optimizations
#
# Prerequisites:
# - PyTorch installed with CUDA support
# - SageAttention installed
# - CUDA 11.0+

echo "======================================================================"
echo "SageAttention Optimization Test Suite for RTX 3090"
echo "======================================================================"
echo ""
echo "This script will test all 5 optimizations and measure their impact."
echo "Estimated time: 5-10 minutes"
echo ""
read -p "Press Enter to continue..."

# Navigate to SageAttention directory
cd /home/user/SageAttention

# Check if PyTorch is installed
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: PyTorch not found!"
    echo "Please install PyTorch first:"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# Check if CUDA is available
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'CUDA available: {torch.cuda.get_device_name(0)}')"
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA not available!"
    echo "Please ensure CUDA drivers are installed and GPU is accessible."
    exit 1
fi

# Check if SageAttention is installed
python3 -c "import sageattention; print(f'SageAttention imported successfully')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: SageAttention not installed in site-packages"
    echo "Installing from source..."
    pip install -e .
fi

echo ""
echo "======================================================================"
echo "Running Test 1: CUDA Graphs (Expected: 1.15-1.25x on RTX 3090)"
echo "======================================================================"
python3 examples/test_cuda_graphs.py

echo ""
echo "======================================================================"
echo "Running Test 2: All Optimizations"
echo "======================================================================"
python3 examples/test_all_optimizations.py

echo ""
echo "======================================================================"
echo "Test Complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Review the speedup results above"
echo "2. Integrate the most effective optimizations into your code"
echo "3. See examples/README_OPTIMIZATIONS.md for usage examples"
echo ""
