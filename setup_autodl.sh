#!/bin/bash
# ERPDiff environment setup for AutoDL
# Run once on a fresh instance: bash setup_autodl.sh
set -e

echo "=== [1/4] 检查 PyTorch 环境 ==="
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"

echo "=== [2/4] 安装缺失依赖（清华镜像）==="
pip install scipy pyyaml matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== [3/4] 写入数据集路径环境变量 ==="
grep -q "ERPDIFF_DATASET" ~/.bashrc || cat >> ~/.bashrc << 'ENVEOF'

# ERPDiff dataset paths
export ERPDIFF_DATASET1_DIR=/root/autodl-tmp/datasets/dataset1
export ERPDIFF_DATASET2_DIR=/root/autodl-tmp/datasets/dataset2
export ERPDIFF_DATASET3_DIR=/root/autodl-tmp/datasets/dataset3
ENVEOF
source ~/.bashrc
echo "Dataset env vars set."

echo "=== [4/4] 检查数据集目录 ==="
for ds in dataset1 dataset2 dataset3; do
    dir="/root/autodl-tmp/datasets/$ds"
    if [ -d "$dir" ]; then
        count=$(ls "$dir"/*.pkl 2>/dev/null | wc -l)
        echo "  $ds: OK ($count .pkl files in $dir)"
    else
        echo "  WARNING: $dir 不存在，请先上传数据集"
    fi
done

echo ""
echo "完成。请执行: source ~/.bashrc"
