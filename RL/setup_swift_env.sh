#!/bin/bash
# ms-swift环境配置脚本
# 要求：python=3.10/3.11, cuda12.*

set -e  # 遇到错误立即退出

echo "=========================================="
echo "ms-swift Environment Setup Script"
echo "=========================================="

# 检查Python版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Current Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
    echo "Warning: Python version should be 3.10 or 3.11, but found $PYTHON_VERSION"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 检查CUDA版本
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "Current CUDA version: $CUDA_VERSION"
    
    if [[ ! "$CUDA_VERSION" =~ ^12\. ]]; then
        echo "Warning: CUDA version should be 12.*, but found $CUDA_VERSION"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "Warning: nvcc not found. Make sure CUDA 12.* is installed."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting installation..."
echo ""

# 升级pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装基础依赖（注意版本兼容性）
echo ""
echo "Installing transformers first (for compatibility)..."
pip install "transformers==4.57.1" -U

echo ""
echo "Installing sglang..."
pip install "sglang<0.5.6" -U

echo ""
echo "Installing vllm..."
pip install "vllm>=0.5.1,<0.11.1" -U
# 修复 vllm 的 outlines_core 依赖
pip install "outlines-core==0.2.11" -U || echo "Warning: outlines-core installation failed"

echo ""
echo "Installing lmdeploy..."
pip install "lmdeploy>=0.5,<0.10.2" -U

echo ""
echo "Installing peft (compatible with lmdeploy)..."
pip install "peft<=0.14.0" -U

echo ""
echo "Installing trl..."
pip install "trl<0.25" -U

echo ""
echo "Installing quantization and optimization tools..."
pip install auto_gptq optimum bitsandbytes "gradio<5.33" -U

echo ""
echo "Installing ms-swift..."
pip install git+https://github.com/modelscope/ms-swift.git#egg=ms-swift[all]

echo ""
echo "Installing additional dependencies..."
pip install timm "deepspeed<0.18" -U

echo ""
echo "Installing Qwen utilities..."
pip install "qwen_vl_utils>=0.0.6" qwen_omni_utils keye_vl_utils -U

echo ""
echo "Installing multimedia libraries..."
pip install decord librosa icecream soundfile -U

echo ""
echo "Installing development and monitoring tools..."
pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab -U

echo ""
echo "=========================================="
echo "Flash Attention Installation"
echo "=========================================="
echo "Flash Attention needs to be installed from GitHub releases."
echo "Please visit: https://github.com/Dao-AILab/flash-attention/releases"
echo ""
echo "Or install manually with:"
echo "  pip install flash-attn --no-build-isolation"
echo ""
read -p "Do you want to install flash-attn now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing flash-attn (this may take a while)..."
    pip install flash-attn --no-build-isolation || {
        echo "Warning: flash-attn installation failed. You may need to install it manually."
        echo "Please check: https://github.com/Dao-AILab/flash-attention/releases"
    }
fi

echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo ""
echo "Verifying key packages..."

# 验证关键包
echo -n "  ms-swift: "
python -c "import swift; print(swift.__version__)" 2>/dev/null || echo "Not found"

echo -n "  transformers: "
python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Not found"

echo -n "  vllm: "
python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "Not found"

echo -n "  sglang: "
python -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "Not found"

echo -n "  lmdeploy: "
python -c "import lmdeploy; print(lmdeploy.__version__)" 2>/dev/null || echo "Not found"

echo -n "  flash-attn: "
python -c "import flash_attn; print('Installed')" 2>/dev/null || echo "Not installed (optional)"

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If flash-attn installation failed, install it manually from:"
echo "   https://github.com/Dao-AILab/flash-attention/releases"
echo ""
echo "2. Test the installation:"
echo "   swift --version"
echo ""
echo "3. Start training:"
echo "   bash RL/train_rationale_sft.sh"
echo ""
