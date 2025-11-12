# /mnt/data/kw/models/Qwen/Qwen3-8B
# /mnt/data/kw/models/Qwen/Qwen2.5-Coder-7B-Instruct
# /mnt/data/kw/models/Qwen/Qwen2.5-Coder-32B-Instruct
# /mnt/data/kw/models/Qwen/Qwen3-Next-80B-A3B-Thinking
CUDA_VISIBLE_DEVICES=2,3 screen -L -Logfile log/eval_Qwen2.5-Coder-32B-Instruct.log python eval/eval.py \
    --model-path "/mnt/data/kw/models/Qwen/Qwen2.5-Coder-32B-Instruct" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32000 \
    --max-tokens 1024 \
    --max-turns 15 \
    --output-file-path "eval/output/evaluation_results_Qwen2.5-Coder-32B-Instruct.jsonl" \
    --dataset-file-path "data/finbench_dataset.json" \
    --tool-tree-path "data/tool_tree.json" \
    --tool-desc-path "data/tool_description.json" 