#!/usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import re
import sys
import math
import argparse
import traceback
from typing import List, Dict, Any, Optional, Set, Tuple, Union
import numpy as np
# = a========================================================
# 0. VLLM 和 Tools 导入
# ==========================================================
try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("未找到 'vllm' 库。")
    print("请先安装 vLLM: pip install vllm")
    sys.exit(1)
try:
    from data.tool_library import *
except ImportError:
    print("找不到 tool_library, 请确保您在 'finbench' 根目录下运行")
    sys.exit(1)
# ==========================================================
# 1. 配置
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="运行 FinBench 评估")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="VLLM模型的本地或Hugging Face路径。"
    )
    parser.add_argument(
        "--tensor-parallel-size", 
        type=int, 
        required=True,
        help="模型跨越的GPU数量（张量并行大小）。"
    )
    parser.add_argument(
        "--gpu-memory-utilization", 
        type=float, 
        required=True,
        help="分配给模型的GPU内存百分比。"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        required=True,  
        help="VLLM模型的最大上下文长度（max_model_len）。"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="模型生成每个响应的最大token数（max_tokens）。"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        required=True,
        help="Agent 循环的最大推理轮次。"
    )
    parser.add_argument(
        "--output-file-path", 
        type=str, 
        required=True,
        help="评估结果的输出文件路径。"
    )
    parser.add_argument(
        "--dataset-file-path", 
        type=str, 
        required=True, 
        help="数据集路径。"
        )
    parser.add_argument(
        "--tool-tree-path", 
        type=str, 
        required=True, 
        help="工具树路径。"
        )
    parser.add_argument(
        "--tool-desc-path", 
        type=str, 
        required=True, 
        help="工具定义路径。"
        )
    return parser.parse_args()

args = parse_args()
MODEL_PATH = args.model_path
OUTPUT_FILE_PATH = args.output_file_path
DATASET_FILE_PATH = args.dataset_file_path
TOOL_TREE_PATH = args.tool_tree_path
TOOL_DESC_PATH = args.tool_desc_path
TENSOR_PARALLEL_SIZE = args.tensor_parallel_size
GPU_MEM_UTILIZATION = args.gpu_memory_utilization
MAX_MODEL_LEN = args.max_model_len
MAX_TOKENS = args.max_tokens
MAX_TURNS = args.max_turns

# ==========================================================
# 1.5 加载工具定义和树状结构
# ==========================================================
try:
    print("--- 正在加载工具定义和树状结构 ---")
    if not os.path.exists(TOOL_TREE_PATH): raise FileNotFoundError(f"Missing: {TOOL_TREE_PATH}")
    if not os.path.exists(TOOL_DESC_PATH): raise FileNotFoundError(f"Missing: {TOOL_DESC_PATH}")
    with open(TOOL_TREE_PATH, 'r', encoding='utf-8') as f: tool_tree = json.load(f)
    with open(TOOL_DESC_PATH, 'r', encoding='utf-8') as f: all_tools_description = list(json.load(f).values()) 
    print(f"--- 工具定义 ({len(all_tools_description)}个) 和树状结构加载成功 ---")
except Exception as e:
    print(f"加载工具文件时发生意外错误: {e}"); sys.exit(1)

# ==========================================================
# 2. 初始化 VLLM 模型
# ==========================================================
model: Optional[LLM] = None
tokenizer: Optional[Any] = None
sampling_params: Optional[SamplingParams] = None
# ==========================================================
# 3. 定义工具映射器
# ==========================================================
function_mapper = {
        name: globals()[name] 
        for name in json.load(open(TOOL_DESC_PATH, 'r', encoding='utf-8')).keys() 
        if name in globals() and callable(globals()[name])
    }
print(f"--- function_mapper (共 {len(function_mapper)} 个) ---")
# ==========================================================
# 4. 构建混合模式 System Prompt
# ==========================================================
def format_tool_tree_for_prompt(tool_tree_dict, indent_level=0):
    markdown_str = ""
    indent = "  " * indent_level
    for key, value in tool_tree_dict.items():
        if key == "_description": continue
        if isinstance(value, dict):
            description = value.get("_description", "")
            markdown_str += f"{indent}- **{key}**: {description}\n"
            if isinstance(value.get("children"), dict):
                markdown_str += format_tool_tree_for_prompt(value["children"], indent_level + 1)
        else:
            markdown_str += f"{indent}- **{key}**\n"
    return markdown_str
# ==========================================================
# 4. 构建最终答案提取模式 System Prompt
# ==========================================================
def build_final_answer_prompt(tool_tree_dict: dict, all_tools_description: list) -> str:
    """
    构建一个指导模型进行“思考->行动”循环，并最终只输出 \boxed{{}} 答案的 System Prompt。
    """
    tool_hierarchy_text = format_tool_tree_for_prompt(tool_tree_dict)
    tools_json_str = json.dumps(all_tools_description, indent=2, ensure_ascii=False)
    template = """You are a professional financial question answering assistant. Your task is to use tools to find the answer and provide ONLY the final result in a specific format.

# Workflow
1.  **Thinking & Tool Use**: Use the provided tools in a step-by-step manner to gather all necessary information. You can think and call tools for multiple turns.
2.  **Final Answer**: Once you have all the information, your **FINAL response MUST ONLY contain the `\\boxed{{}}` expression** and nothing else.

# Final Answer Requirements
- Your last response must start with `\\boxed{{` and end with `}}`.
- Inside `\\boxed{{}}`, place the final answer value(s) directly.
- **DO NOT** wrap the answer in a list `[]`.
- If the question asks for multiple values, separate them with a comma `,`.

---
### Interaction Example

#### User Question
What are the latest opening prices for SF Express's A-share (sz002352) and HK-share (0270)?

#### Your Response (First Turn)
Thinking:
I need to find the latest opening price for two different stocks in two different markets. I will use `get_a_stock_daily_price` for the A-share and `get_hk_stock_daily_price` for the HK-share.

Action:
<tool>get_a_stock_daily_price({{"code": "sz002352", "column_label": "open", "query_date": "latest"}})</tool>
<tool>get_hk_stock_daily_price({{"code": "0270", "column_label": "open", "query_date": "latest"}})</tool>

#### (System returns tool results to you: A-share price is 40.2, HK-share price is 7.2)

#### Your Response (Second and FINAL Turn) Put the answer in \\boxed{{}} (value, string, date) in the end of the answer, for example:
\\boxed{{40.2, 7.2}}
---

## Tool Hierarchy
{hierarchy}

## Available Tools (JSON Schema)
{tools_json}
"""
    system_prompt = template.format(hierarchy=tool_hierarchy_text, tools_json=tools_json_str)
    return system_prompt

def _numpy_converter(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

# ==========================================================
# 5. 核心 Generate 过程与评估辅助函数
# ==========================================================
def execute_simplified_code(code_string: str, function_mapper_dict: dict) -> Tuple[bool, Any, Optional[str]]:
    """
    执行代码字符串
    """
    indented_code = "\n".join(["    " + line for line in code_string.strip().splitlines()])
    temp_func_code = f"def temp_solve():\n{indented_code}"
    try:
        execution_context = globals().copy()
        execution_context.update(function_mapper_dict)
        exec(temp_func_code, execution_context)
        solve_function = execution_context.get('temp_solve')
        if not callable(solve_function):
            return False, None, "Failed to create a callable temporary function from the code string."
        result = solve_function()
        return True, result, None
    except Exception as e:
        return False, None, f"Execution failed: {traceback.format_exc()}"


def _truncate_messages(messages: List[Dict[str, str]], tokenizer: Any, max_len: int) -> List[Dict[str, str]]:
    PROMPT_OVERHEAD_ESTIMATE = 50
    try:
        current_tokens = len(tokenizer.apply_chat_template(messages, add_generation_prompt=False))
    except Exception:
        current_tokens = sum(len(tokenizer.encode(m.get("content", ""))) for m in messages)
    if current_tokens + PROMPT_OVERHEAD_ESTIMATE <= max_len:
        return messages
    print(f"    [Agent 警告] 对话历史过长 (估算 {current_tokens} tokens)，正在进行截断...")
    system_prompt = messages[0]
    CONVERSATION_TO_KEEP = 4 
    truncated_history = messages[-(CONVERSATION_TO_KEEP * 2):]
    new_messages = [system_prompt] + truncated_history
    try:
        new_tokens = len(tokenizer.apply_chat_template(new_messages, add_generation_prompt=False))
        print(f"    [Agent 信息] 截断后，对话历史长度从 {current_tokens} 减少到 {new_tokens} tokens。")
    except Exception:
        pass
    return new_messages

def run_agent_loop(user_question: str, model: LLM, tokenizer: Any, sampling_params: SamplingParams, system_prompt: str) -> Tuple[str, str]:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}]
    final_response = ""
    full_trajectory = f"--- USER ---\n{user_question}\n\n"
    try:
        model_max_len = model.llm_engine.model_config.max_model_len
    except AttributeError:
        model_max_len = 32768
        print(f"    [Agent 警告] 无法自动获取模型最大长度，使用默认值: {model_max_len}")
    for turn in range(MAX_TURNS):
        print(f"    [Agent] 正在进行第 {turn + 1}/{MAX_TURNS} 推理...")
        messages = _truncate_messages(messages, tokenizer, model_max_len)
        try:
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            error_msg = f"    [Agent 错误] apply_chat_template 失败: {e}"
            print(error_msg)
            return "", full_trajectory + error_msg
        if len(tokenizer.encode(prompt_str)) >= model_max_len:
            error_msg = f"    [Agent 错误] 即使截断后，最终的 prompt 长度仍然超出模型限制。请考虑减少 CONVERSATION_TO_KEEP 的值。"
            print(error_msg)
            return "", full_trajectory + error_msg
        outputs = model.generate([prompt_str], sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text.strip()
        messages.append({"role": "assistant", "content": response_text})
        final_response = response_text
        full_trajectory += f"--- ASSISTANT (思考与工具调用) ---\n{response_text}\n\n"
        print(f"    [Agent 模型回复] {response_text[:150]}...")
        tool_call_matches = re.findall(r"<tool>(.*?)</tool>", response_text, re.DOTALL)
        if not tool_call_matches:
            print("    [Agent] 流程结束 (模型未调用工具，准备提供最终答案)。")
            return final_response, full_trajectory
        tool_output_content = ""
        for match_str in tool_call_matches:
            try:
                func_name = match_str.split('(', 1)[0].strip()
                args_str = match_str.rsplit(')', 1)[0].split('(', 1)[1]
                arguments = json.loads(args_str)
                function_to_call = function_mapper.get(func_name)
                if function_to_call:
                    function_output = function_to_call(**arguments)
                    output_str = json.dumps(function_output, ensure_ascii=False) if isinstance(function_output, (dict, list)) else str(function_output)
                    tool_output_content += output_str
                else:
                    tool_output_content += f"Error: Tool '{func_name}' not found."
            except Exception as e:
                tool_output_content += f"工具执行失败: {e}"
        if len(tool_output_content) > 4000:
            print(f"    [Agent 警告] 工具输出过长 ({len(tool_output_content)} chars)，将被截断。")
            tool_output_content = tool_output_content[:4000] + "\n... [TRUNCATED] ..."
        messages.append({"role": "tool", "content": tool_output_content})
        full_trajectory += f"--- TOOL OUTPUT ---\n{tool_output_content}\n\n"
    print(f"    [Agent 警告] 达到最大 {MAX_TURNS} 轮次，强行终止。")
    return final_response, full_trajectory

def extract_boxed_answer(model_content: str) -> Tuple[Any, Optional[str]]:
    boxed_match = re.search(r"\\boxed{([^}]+)}", model_content)
    if not boxed_match:
        return None, "Error: No \\boxed{} found in the final response."
    boxed_answer_str = boxed_match.group(1).strip()
    eval_str = f"({boxed_answer_str},)" if ',' not in boxed_answer_str else f"({boxed_answer_str})"
    try:
        import ast
        result = ast.literal_eval(eval_str)
        return result[0] if len(result) == 1 else result, None
    except Exception as e:
        return None, f"Error: Failed to parse content inside \\boxed{{}}: {e}"

def compare_answers(model_answer: Any, ground_truth: Any, rel_tol=1e-5) -> bool:
    # 1. 标准化为元组以便统一比较
    model_tuple = model_answer if isinstance(model_answer, tuple) else (model_answer,)
    truth_tuple = ground_truth if isinstance(ground_truth, tuple) else (ground_truth,)
    # 2. 检查长度
    if len(model_tuple) != len(truth_tuple):
        return False
    # 3. 逐个元素比较
    for m_item, t_item in zip(model_tuple, truth_tuple):
        if isinstance(m_item, (int, float)) and isinstance(t_item, (int, float)):
            if not math.isclose(m_item, t_item, rel_tol=rel_tol, abs_tol=1e-9):
                return False
        elif isinstance(m_item, float) and isinstance(t_item, float) and math.isnan(m_item) and math.isnan(t_item):
            continue
        elif m_item != t_item:
            return False
    return True

# ==========================================================
# 6. 主评估循环 (已修复 vLLM 加载问题)
# ==========================================================
def main_evaluation(model: LLM, tokenizer: Any, sampling_params: SamplingParams):
    
    system_prompt = build_final_answer_prompt(tool_tree, all_tools_description)
    print("--- 启动评估 (Agent流程 + 直接值Boxed Answer评估) ---")
    
    try:
        with open(DATASET_FILE_PATH, 'r', encoding='utf-8') as f: data_list = json.load(f)
        total_count = len(data_list)
        print(f"成功加载 {total_count} 条数据。")
    except Exception as e: 
        print(f"【致命错误】: 加载数据集失败: {e}")
        return
        
    processed_prompts = set()
    if os.path.exists(OUTPUT_FILE_PATH):
        try:
            with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "prompt" in data:
                                processed_prompts.add(data["prompt"])
                        except json.JSONDecodeError:
                            print(f"  [警告] 在读取输出文件时发现无效JSON行，已跳过: {line.strip()}")
                            pass
            print(f"--- 已加载 {len(processed_prompts)} 个已处理的 prompts，将在本次运行中跳过它们。---\n")
        except Exception as e:
            print(f"--- 读取现有输出文件失败: {e}. 将从头开始处理。 ---")
            
    results_log = [] 
    success_count_this_run = 0
    
    with open(OUTPUT_FILE_PATH, 'a', encoding='utf-8') as f_out:
        for i, item in enumerate(data_list):
            prompt = item.get("question")
            ground_truth_code = item.get("code")
            
            if prompt in processed_prompts:
                continue
                
            print(f"\n--- 正在评估第 {i+1}/{total_count} 项 (新项目) ---")
            
            if not prompt or not ground_truth_code: 
                print("    [跳过] 数据缺失 ('question' or 'code')。")
                continue
                
            print(f"    [Prompt] {prompt[:100]}...")
            
            log_entry = { 
                        "index": i + 1, 
                        "prompt": prompt, 
                        "is_correct": False, 
                        "eval_score": 0 
                        }
            
            success, ground_truth_answer, err = execute_simplified_code(ground_truth_code, function_mapper)
            
            if not success:
                print(f"    ❌ 基准代码执行失败: {err}")
                log_entry["ground_truth_error"] = str(err)
                f_out.write(json.dumps(log_entry, ensure_ascii=False, default=_numpy_converter) + '\n'); f_out.flush()
                processed_prompts.add(prompt)
                continue
                
            log_entry["ground_truth_answer"] = ground_truth_answer
            print(f"    ✅ 基准答案: {ground_truth_answer}")
            
            # [修复] (tom)，'model', 'tokenizer', 'sampling_params' 现在是传入的参数
            final_response, trajectory = run_agent_loop(prompt, model, tokenizer, sampling_params, system_prompt)
            log_entry["model_trajectory"] = trajectory
            
            model_answer, err = extract_boxed_answer(final_response)
            
            if err:
                print(f"    ❌ 模型答案提取失败: {err}")
                log_entry["model_error"] = err
                f_out.write(json.dumps(log_entry, ensure_ascii=False, default=_numpy_converter) + '\n'); f_out.flush()
                processed_prompts.add(prompt)
                continue
                
            log_entry["model_answer"] = model_answer
            print(f"    🤖 模型答案: {model_answer}")
            
            is_correct = compare_answers(model_answer, ground_truth_answer)
            log_entry["is_correct"] = is_correct
            
            if is_correct:
                log_entry["eval_score"] = 1
                success_count_this_run += 1
                print("    👍 结果: 正确")
            else:
                log_entry["eval_score"] = 0
                print("    👎 结果: 错误")
                
            json_str = json.dumps(log_entry, ensure_ascii=False, default=_numpy_converter)
            f_out.write(json_str + '\n')
            f_out.flush()
            processed_prompts.add(prompt)
            results_log.append(log_entry) 
            
    total_processed_in_file = len(processed_prompts)
    processed_this_run = len(results_log)
    total_correct_in_file = 0
    
    if os.path.exists(OUTPUT_FILE_PATH):
        with open(OUTPUT_FILE_PATH, 'r', encoding='utf-8') as f_final:
            for line in f_final:
                if line.strip():
                    try:
                        final_data = json.loads(line)
                        if final_data.get("is_correct") is True:
                            total_correct_in_file += 1
                    except: pass
                    
    print("\n" + "="*50)
    print("--- 评估总结 ---")
    print(f"本次运行处理条目:  {processed_this_run}")
    print(f"  - 本次运行正确:  {success_count_this_run}")
    print("-" * 20)
    print(f"输入文件总条目:      {total_count}")
    print(f"输出文件中总已处理:  {total_processed_in_file}")
    print(f"输出文件中总正确:    {total_correct_in_file}")
    
    if total_processed_in_file > 0:
        overall_accuracy = (total_correct_in_file / total_processed_in_file) * 100
        print(f"基于已处理条目的总成功率: {overall_accuracy:.2f}%")
        
    print(f"详细日志已追加到: {OUTPUT_FILE_PATH}")
    print("="*50)

# ==========================================================
# 7. 主入口点
# ==========================================================
if __name__ == "__main__":
    try:
        print(f"--- 正在加载 VLLM 模型: {MODEL_PATH} ---")
        model = LLM(
            model=MODEL_PATH, 
            tensor_parallel_size=TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=GPU_MEM_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            disable_log_stats=True 
        )
        tokenizer = model.get_tokenizer()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS) 
        print("--- VLLM 模型和 Tokenizer 加载成功。")
    except Exception as e:
        print(f"加载 vLLM 模型失败: {e}"); 
        print(traceback.format_exc()) 
        sys.exit(1)
    main_evaluation(model, tokenizer, sampling_params)