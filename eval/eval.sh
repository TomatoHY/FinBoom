#!/bin/bash
# FinBoom 批量评估脚本
# 用于批量评估多个模型（并行运行）

# 注意：不使用 set -e，因为我们需要等待所有后台进程

# 配置
DATASET_PATH="data/finboom_dataset.json"
TOOL_TREE_PATH="data/tool_tree.json"
TOOL_DESC_PATH="data/tool_description.json"
OUTPUT_DIR="eval/output"
MAX_TURNS=10
MAX_TOKENS=2048
MAX_MODEL_LEN=8192
GPU_MEMORY_UTIL=0.9

# 创建输出目录和日志目录
mkdir -p "$OUTPUT_DIR"
LOG_DIR="eval/log"
mkdir -p "$LOG_DIR"

# 检查screen是否安装
if ! command -v screen &> /dev/null; then
    echo "错误: 未找到 'screen' 命令"
    echo "请安装 screen: sudo apt-get install screen 或 sudo yum install screen"
    exit 1
fi

# 模型配置列表
# 格式: "模型名称:模型路径:张量并行大小:GPU设备ID"
# GPU设备ID格式: "0,1" 表示使用GPU 0和1
# 注意: 模型路径请根据您的实际环境配置（可以是绝对路径或相对路径）
MODELS=(
    # 示例配置（请根据您的实际环境修改）:
    # "Qwen2.5-Coder-7B-Instruct:/path/to/your/models/Qwen2.5-Coder-7B-Instruct:2:0,1"
    # "Qwen2.5-Coder-32B-Instruct:/path/to/your/models/Qwen2.5-Coder-32B-Instruct:2:2,3"
    # 添加更多模型...
    # 示例: "Model-Name:/path/to/model:2:4,5"  # 使用GPU 4和5
)

echo "=========================================="
echo "FinBoom 批量评估"
echo "=========================================="
echo "数据集: $DATASET_PATH"
echo "最大轮次: $MAX_TURNS"
echo "输出目录: $OUTPUT_DIR"
echo "模型数量: ${#MODELS[@]}"
echo ""
echo "GPU配置说明:"
echo "  - 每个模型使用指定的GPU设备"
echo "  - 张量并行大小应与GPU数量匹配"
echo "  - 例如: 2个GPU使用 tensor-parallel-size=2"
echo "  - 所有模型将在独立的screen会话中并行运行"
echo "=========================================="
echo ""

# 存储screen会话名称和模型信息
declare -a SCREEN_NAMES=()
declare -a MODEL_NAMES=()
declare -a RESULTS_FILES=()
declare -a METRICS_FILES=()

# 启动所有模型的评估（并行）
for model_info in "${MODELS[@]}"; do
    IFS=':' read -r model_name model_path tensor_parallel gpu_ids <<< "$model_info"
    
    echo "----------------------------------------"
    echo "启动模型评估: $model_name"
    echo "模型路径: $model_path"
    echo "张量并行: $tensor_parallel"
    echo "GPU设备: $gpu_ids"
    echo "----------------------------------------"
    
    # 检查模型路径是否存在
    if [ ! -d "$model_path" ]; then
        echo "警告: 模型路径不存在，跳过: $model_path"
        continue
    fi
    
    # 验证GPU数量与张量并行大小是否匹配
    if [ -n "$gpu_ids" ]; then
        gpu_count=$(echo "$gpu_ids" | tr ',' '\n' | wc -l)
        if [ "$gpu_count" -ne "$tensor_parallel" ]; then
            echo "警告: GPU数量($gpu_count)与张量并行大小($tensor_parallel)不匹配！"
            echo "      建议: 张量并行大小应等于GPU数量"
        fi
    fi
    
    # 生成输出文件名和screen会话名
    model_name_safe=$(echo "$model_name" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    results_file="$OUTPUT_DIR/results_${model_name_safe}.jsonl"
    metrics_file="$OUTPUT_DIR/metrics_${model_name_safe}.json"
    log_file="$LOG_DIR/${model_name_safe}.log"
    screen_name="finboom_${model_name_safe}"
    
    # 保存模型信息
    MODEL_NAMES+=("$model_name")
    RESULTS_FILES+=("$results_file")
    METRICS_FILES+=("$metrics_file")
    SCREEN_NAMES+=("$screen_name")
    
    # 检查screen会话是否已存在
    if screen -list | grep -q "$screen_name"; then
        echo "警告: screen会话 '$screen_name' 已存在，将先终止它"
        screen -S "$screen_name" -X quit 2>/dev/null || true
        sleep 1
    fi
    
    # 创建评估脚本文件
    eval_script="$LOG_DIR/${model_name_safe}_eval.sh"
    cat > "$eval_script" << EOF
#!/bin/bash
# 模型评估脚本: $model_name
# GPU设备: ${gpu_ids:-默认}

export CUDA_VISIBLE_DEVICES='$gpu_ids'
cd $(pwd)

# 确保输出实时刷新
export PYTHONUNBUFFERED=1

echo '========================================'
echo "模型: $model_name"
echo "GPU设备: ${gpu_ids:-默认}"
echo "开始时间: \$(date '+%Y-%m-%d %H:%M:%S')"
echo '========================================'
echo ''

# 运行模型评估，同时输出到屏幕和日志文件
python eval/eval.py evaluate \\
    --model-type local \\
    --model-path '$model_path' \\
    --tensor-parallel-size $tensor_parallel \\
    --gpu-memory-utilization $GPU_MEMORY_UTIL \\
    --max-model-len $MAX_MODEL_LEN \\
    --max-tokens $MAX_TOKENS \\
    --max-turns $MAX_TURNS \\
    --dataset-file-path '$DATASET_PATH' \\
    --tool-tree-path '$TOOL_TREE_PATH' \\
    --tool-desc-path '$TOOL_DESC_PATH' \\
    --output-file-path '$results_file' 2>&1 | tee -a '$log_file'

eval_exit_code=\${PIPESTATUS[0]}

echo ''
echo '========================================'
if [ \$eval_exit_code -eq 0 ]; then
    echo '✓ 模型评估完成'
    echo '开始计算指标...'
    python eval/eval.py calculate-metrics \\
        --results-file '$results_file' \\
        --output-file '$metrics_file' \\
        --max-turns $MAX_TURNS \\
        --pretty 2>&1 | tee -a '$log_file'
    if [ \${PIPESTATUS[0]} -eq 0 ]; then
        echo '✓ 指标计算完成'
    else
        echo '✗ 指标计算失败'
    fi
else
    echo "✗ 模型评估失败 (退出码: \$eval_exit_code)"
fi
echo "结束时间: \$(date '+%Y-%m-%d %H:%M:%S')"
echo '========================================'
echo ''
echo '评估完成！按任意键关闭此窗口...'
read -n 1
EOF
    
    chmod +x "$eval_script"
    
    # 启动screen会话（使用-L参数启用日志记录，-Logfile指定日志文件）
    screen -L -Logfile "$LOG_DIR/${model_name_safe}_screen.log" -dmS "$screen_name" bash -c "$eval_script"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Screen会话已启动: $screen_name"
        echo "  查看日志: tail -f $log_file"
        echo "  进入会话: screen -r $screen_name"
    else
        echo "  ✗ 启动screen会话失败"
    fi
    echo ""
done

echo "=========================================="
echo "所有模型评估已启动（在独立的screen会话中运行）"
echo "=========================================="
echo ""
echo "Screen会话列表:"
screen -list | grep "finboom_" || echo "  (无活动会话)"
echo ""
echo "常用命令:"
echo "  查看所有screen会话: screen -list"
echo "  进入某个会话: screen -r <会话名>"
echo "  退出会话（不断开）: 按 Ctrl+A 然后按 D"
echo "  终止会话: screen -S <会话名> -X quit"
echo ""
echo "实时查看日志:"
for i in "${!SCREEN_NAMES[@]}"; do
    model_name=${MODEL_NAMES[$i]}
    model_name_safe=$(echo "$model_name" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')
    log_file="$LOG_DIR/${model_name_safe}.log"
    screen_log="$LOG_DIR/${model_name_safe}_screen.log"
    echo "  tail -f $log_file  # $model_name (评估日志)"
    echo "  tail -f $screen_log  # $model_name (screen输出)"
done
echo ""
echo "=========================================="
echo "提示: 所有评估任务在后台运行，您可以:"
echo "  1. 断开SSH连接，任务会继续运行"
echo "  2. 使用 'screen -list' 查看运行状态"
echo "  3. 使用 'screen -r <会话名>' 查看实时输出"
echo "  4. 查看日志文件了解进度"
echo "=========================================="
echo ""

# 显示所有模型的简要结果
for i in "${!MODEL_NAMES[@]}"; do
    model_name=${MODEL_NAMES[$i]}
    metrics_file=${METRICS_FILES[$i]}
    
    if [ -f "$metrics_file" ]; then
        echo "----------------------------------------"
        echo "模型: $model_name"
        echo "----------------------------------------"
        python -c "
import json
try:
    with open('$metrics_file', 'r') as f:
        data = json.load(f)
        metrics = data['overall_metrics']
        print(f\"  TSR: {metrics['tsr']:.4f}\")
        print(f\"  FAA: {metrics['faa']:.4f}\")
        print(f\"  记忆作弊率: {metrics['memory_cheating_rate']:.4f}\")
        print(f\"  CER: {metrics['cer']:.4f}\")
        print(f\"  AR: {metrics['ar']:.4f}\")
        print(f\"  Avg. EEP: {metrics['avg_eep']:.4f}\")
        print(f\"  FRR: {metrics['frr']:.4f}\")
        print(f\"  Avg. LC: {metrics['avg_lc']:.4f}\")
except Exception as e:
    print(f\"  错误: 无法读取指标文件 - {e}\")
" 2>/dev/null || echo "  错误: 无法读取指标文件"
        echo ""
    fi
done

echo "=========================================="
echo "批量评估完成！"
echo "=========================================="
echo ""
echo "所有结果文件保存在: $OUTPUT_DIR"
echo "日志文件保存在: $LOG_DIR"
echo ""
echo "查看结果:"
echo "  ls -lh $OUTPUT_DIR/results_*.jsonl"
echo "  ls -lh $OUTPUT_DIR/metrics_*.json"
echo ""
echo "查看日志:"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "管理screen会话:"
echo "  查看所有会话: screen -list"
echo "  进入会话: screen -r <会话名>"
echo "  终止会话: screen -S <会话名> -X quit"
echo ""

