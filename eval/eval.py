#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
FinBoom 统一评估脚本

支持两种模式：
1. 本地部署模型（使用 vLLM）
2. API模型（OpenAI, Anthropic, Google, Qwen等）

使用方法：
  # 本地模型
  python eval/eval.py --model-type local --model-path /path/to/model ...

  # API模型
  python eval/eval.py --model-type api --provider openai --model-name gpt-4 ...
"""

import json
import os
import re
import sys
import math
import time
import argparse
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# ==========================================================
# 0. 导入依赖
# ==========================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__))
finboom_root = os.path.join(current_script_dir, '..')
sys.path.insert(0, finboom_root) 

try:
    from data.tool_library import *
except ImportError:
    print("找不到 tool_library, 请确保您在 'finboom' 根目录下运行")
    sys.exit(1)

# 可选导入 vLLM（仅本地模型需要）
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# 可选导入 API 库（仅API模型需要）
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


# ==========================================================
# 1. API模型包装器
# ==========================================================
class APIModelWrapper:
    """API模型包装器，统一接口"""
    
    def __init__(self, provider: str, model_name: str, api_key: str = None, base_url: str = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.base_url = base_url
        
        if self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("需要安装 openai: pip install openai")
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("需要安装 anthropic: pip install anthropic")
            self.client = Anthropic(api_key=self.api_key)
        elif self.provider == "google":
            if not GOOGLE_AVAILABLE:
                raise ImportError("需要安装 google-generativeai: pip install google-generativeai")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(model_name)
        elif self.provider == "qwen":
            if not OPENAI_AVAILABLE:
                raise ImportError("需要安装 openai: pip install openai")
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            raise ValueError(f"不支持的provider: {provider}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """统一的聊天接口"""
        if self.provider == "openai" or self.provider == "qwen":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
        )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            system_message = None
            conversation = []
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    conversation.append(msg)
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", 2048),
                system=system_message,
                messages=conversation
            )
            return response.content[0].text
        elif self.provider == "google":
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            response = self.client.generate_content(prompt)
            return response.text
        else:
            raise ValueError(f"不支持的provider: {self.provider}")


# ==========================================================
# 2. 工具函数
# ==========================================================
def format_tool_tree_for_prompt(tool_tree_dict, indent_level=0):
    """格式化工具树为Markdown"""
    markdown_str = ""
    indent = "  " * indent_level
    for key, value in tool_tree_dict.items():
        if key == "_description":
            continue
        if isinstance(value, dict):
            description = value.get("_description", "")
            markdown_str += f"{indent}- **{key}**: {description}\n"
            if isinstance(value.get("children"), dict):
                markdown_str += format_tool_tree_for_prompt(value["children"], indent_level + 1)
        else:
            markdown_str += f"{indent}- **{key}**\n"
    return markdown_str


def build_final_answer_prompt(tool_tree_dict: dict, all_tools_description: list) -> str:
    """构建系统提示词"""
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
    """JSON序列化numpy类型"""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def execute_simplified_code(code_string: str, function_mapper_dict: dict) -> Tuple[bool, Any, Optional[str]]:
    """执行代码字符串"""
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


def extract_boxed_answer(model_content: str) -> Tuple[Any, Optional[str]]:
    """从响应中提取\\boxed{}答案"""
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
    """比较模型答案和基准答案"""
    model_tuple = model_answer if isinstance(model_answer, tuple) else (model_answer,)
    truth_tuple = ground_truth if isinstance(ground_truth, tuple) else (ground_truth,)
    if len(model_tuple) != len(truth_tuple):
        return False
    for m_item, t_item in zip(model_tuple, truth_tuple):
        if isinstance(m_item, (int, float)) and isinstance(t_item, (int, float)):
            if not math.isclose(m_item, t_item, rel_tol=rel_tol, abs_tol=1e-9):
                return False
        elif isinstance(m_item, float) and isinstance(t_item, float) and math.isnan(m_item) and math.isnan(t_item):
            continue
        elif m_item != t_item:
            return False
    return True


def extract_tool_calls(response: str) -> List[Dict[str, Any]]:
    """从响应中提取工具调用"""
    tool_calls = []
    pattern = r"<tool>(.*?)</tool>"
    matches = re.findall(pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            func_name = match.split('(', 1)[0].strip()
            args_str = match.rsplit(')', 1)[0].split('(', 1)[1]
            arguments = json.loads(args_str)
            tool_calls.append({
                "name": func_name,
                "args": arguments
            })
        except Exception as e:
            print(f"    工具调用解析失败: {e}")
    
    return tool_calls


# ==========================================================
# 3. 本地模型 Agent 循环
# ==========================================================
def _truncate_messages(messages: List[Dict[str, str]], tokenizer: Any, max_len: int) -> List[Dict[str, str]]:
    """截断过长的对话历史"""
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


def run_agent_loop_local(
    user_question: str,
    model: LLM,
    tokenizer: Any,
    sampling_params: SamplingParams,
    system_prompt: str,
    function_mapper: dict,
    max_turns: int
) -> Tuple[str, str, List[Dict], int]:
    """运行本地模型的Agent循环"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}]
    final_response = ""
    full_trajectory = f"--- USER ---\n{user_question}\n\n"
    tool_call_history = []
    error_count = 0
    
    try:
        model_max_len = model.llm_engine.model_config.max_model_len
    except AttributeError:
        model_max_len = 32768
        print(f"    [Agent 警告] 无法自动获取模型最大长度，使用默认值: {model_max_len}")
    
    for turn in range(max_turns):
        print(f"    [Agent] 正在进行第 {turn + 1}/{max_turns} 推理...")
        messages = _truncate_messages(messages, tokenizer, model_max_len)
        try:
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            error_msg = f"    [Agent 错误] apply_chat_template 失败: {e}"
            print(error_msg)
            return "", full_trajectory + error_msg, tool_call_history, error_count
        if len(tokenizer.encode(prompt_str)) >= model_max_len:
            error_msg = f"    [Agent 错误] 即使截断后，最终的 prompt 长度仍然超出模型限制。"
            print(error_msg)
            return "", full_trajectory + error_msg, tool_call_history, error_count
        
        outputs = model.generate([prompt_str], sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text.strip()
        messages.append({"role": "assistant", "content": response_text})
        final_response = response_text
        full_trajectory += f"--- ASSISTANT (思考与工具调用) ---\n{response_text}\n\n"
        print(f"    [Agent 模型回复] {response_text[:150]}...")
        
        tool_call_matches = re.findall(r"<tool>(.*?)</tool>", response_text, re.DOTALL)
        if not tool_call_matches:
            print("    [Agent] 流程结束 (模型未调用工具，准备提供最终答案)。")
            return final_response, full_trajectory, tool_call_history, error_count
        
        tool_output_content = ""
        for match_str in tool_call_matches:
            tool_call_start = time.time()
            func_name = "unknown"
            arguments = {}
            try:
                func_name = match_str.split('(', 1)[0].strip()
                args_str = match_str.rsplit(')', 1)[0].split('(', 1)[1]
                arguments = json.loads(args_str)
                function_to_call = function_mapper.get(func_name)
                
                if function_to_call:
                    function_output = function_to_call(**arguments)
                    tool_call_end = time.time()
                    output_str = json.dumps(function_output, ensure_ascii=False) if isinstance(function_output, (dict, list)) else str(function_output)
                    tool_output_content += output_str
                    
                    tool_call_history.append({
                        "turn": turn,
                        "tool_name": func_name,
                        "tool_args": arguments,
                        "result": function_output,
                        "success": True,
                        "latency": tool_call_end - tool_call_start
                    })
                else:
                    error_count += 1
                    tool_call_end = time.time()
                    tool_output_content += f"Error: Tool '{func_name}' not found."
                    tool_call_history.append({
                        "turn": turn,
                        "tool_name": func_name,
                        "tool_args": arguments,
                        "result": {"error": f"工具 '{func_name}' 不存在"},
                        "success": False,
                        "latency": tool_call_end - tool_call_start
                    })
            except json.JSONDecodeError as e:
                error_count += 1
                tool_call_end = time.time()
                error_msg = f"工具参数解析失败: {str(e)}"
                tool_output_content += error_msg
                tool_call_history.append({
                    "turn": turn,
                    "tool_name": func_name,
                    "tool_args": arguments,
                    "result": {"error": f"参数解析失败: {str(e)}"},
                    "success": False,
                    "latency": tool_call_end - tool_call_start
                })
            except Exception as e:
                error_count += 1
                tool_call_end = time.time()
                error_msg = f"工具执行失败: {str(e)}"
                tool_output_content += error_msg
                tool_call_history.append({
                    "turn": turn,
                    "tool_name": func_name,
                    "tool_args": arguments,
                    "result": {"error": str(e)},
                    "success": False,
                    "latency": tool_call_end - tool_call_start
                })
        
        if len(tool_output_content) > 4000:
            print(f"    [Agent 警告] 工具输出过长 ({len(tool_output_content)} chars)，将被截断。")
            tool_output_content = tool_output_content[:4000] + "\n... [TRUNCATED] ..."
        messages.append({"role": "tool", "content": tool_output_content})
        full_trajectory += f"--- TOOL OUTPUT ---\n{tool_output_content}\n\n"
    
    print(f"    [Agent 警告] 达到最大 {max_turns} 轮次，强行终止。")
    return final_response, full_trajectory, tool_call_history, error_count


# ==========================================================
# 4. API模型 Agent 循环
# ==========================================================
def run_agent_loop_api(
    user_question: str,
    model: APIModelWrapper,
    system_prompt: str,
    function_mapper: dict,
    max_turns: int,
    max_tokens: int,
    temperature: float
) -> Tuple[str, str, List[Dict], int]:
    """运行API模型的Agent循环"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}]
    final_response = ""
    full_trajectory = f"--- USER ---\n{user_question}\n\n"
    tool_call_history = []
    error_count = 0
    
    for turn in range(max_turns):
        print(f"    [Agent] 正在进行第 {turn + 1}/{max_turns} 推理...")
        try:
            response = model.chat(
                messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            messages.append({"role": "assistant", "content": response})
            final_response = response
            full_trajectory += f"--- ASSISTANT (思考与工具调用) ---\n{response}\n\n"
            print(f"    [Agent 模型回复] {response[:150]}...")
            
            tool_calls = extract_tool_calls(response)
            
            if not tool_calls:
                final_answer = extract_boxed_answer(response)[0]
                if final_answer:
                    print("    [Agent] 流程结束 (已提取最终答案)。")
                    break
            continue
            
            tool_outputs = []
            for tool_call in tool_calls:
                tool_call_start = time.time()
                try:
                    func = function_mapper.get(tool_call["name"])
                    if func:
                        result = func(**tool_call["args"])
                        tool_call_end = time.time()
                        tool_call_history.append({
                            "turn": turn,
                            "tool_name": tool_call["name"],
                            "tool_args": tool_call["args"],
                            "result": result,
                            "success": True,
                            "latency": tool_call_end - tool_call_start
                        })
                        tool_outputs.append(json.dumps(result, ensure_ascii=False))
                    else:
                        error_count += 1
                        tool_call_history.append({
                            "turn": turn,
                            "tool_name": tool_call["name"],
                            "tool_args": tool_call["args"],
                            "result": {"error": f"工具 '{tool_call['name']}' 不存在"},
                            "success": False,
                            "latency": 0
                        })
                        tool_outputs.append(f"错误: 工具 '{tool_call['name']}' 不存在")
                except Exception as e:
                    error_count += 1
                    tool_call_end = time.time()
                    tool_call_history.append({
                        "turn": turn,
                        "tool_name": tool_call["name"],
                        "tool_args": tool_call["args"],
                        "result": {"error": str(e)},
                        "success": False,
                        "latency": tool_call_end - tool_call_start
                    })
                    tool_outputs.append(f"工具执行失败: {str(e)}")
            
            if tool_outputs:
                tool_output_str = "\n".join(tool_outputs)
                if len(tool_output_str) > 4000:
                    tool_output_str = tool_output_str[:4000] + "...(已截断)"
                messages.append({
                    "role": "user",
                    "content": f"工具调用结果:\n{tool_output_str}\n\n请根据工具调用结果继续处理任务，或给出最终答案。"
                })
                full_trajectory += f"--- TOOL OUTPUT ---\n{tool_output_str}\n\n"
            
            final_answer = extract_boxed_answer(response)[0]
            if final_answer:
                break
        
        except Exception as e:
            print(f"    轮次 {turn + 1} 发生错误: {e}")
            error_count += 1
            break
    
    return final_response, full_trajectory, tool_call_history, error_count


# ==========================================================
# 5. 主评估函数
# ==========================================================
def main_evaluation(args):
    """主评估函数"""
    # 加载工具定义
    try:
        print("--- 正在加载工具定义和树状结构 ---")
        if not os.path.exists(args.tool_tree_path):
            raise FileNotFoundError(f"Missing: {args.tool_tree_path}")
        if not os.path.exists(args.tool_desc_path):
            raise FileNotFoundError(f"Missing: {args.tool_desc_path}")
        with open(args.tool_tree_path, 'r', encoding='utf-8') as f:
            tool_tree = json.load(f)
        with open(args.tool_desc_path, 'r', encoding='utf-8') as f:
            all_tools_description = list(json.load(f).values())
        print(f"--- 工具定义 ({len(all_tools_description)}个) 和树状结构加载成功 ---")
    except Exception as e:
        print(f"加载工具文件时发生意外错误: {e}")
        sys.exit(1)
    
    # 构建工具映射器
    with open(args.tool_desc_path, 'r', encoding='utf-8') as f:
        tool_desc = json.load(f)
    function_mapper = {
        name: globals()[name]
        for name in tool_desc.keys()
        if name in globals() and callable(globals()[name])
    }
    print(f"--- function_mapper (共 {len(function_mapper)} 个) ---")
    
    # 构建系统提示词
    system_prompt = build_final_answer_prompt(tool_tree, all_tools_description)
    
    # 初始化模型
    if args.model_type == "local":
        if not VLLM_AVAILABLE:
            print("错误: 需要安装 vLLM 来使用本地模型")
            print("请安装: pip install vllm")
            sys.exit(1)
        print(f"--- 正在加载 VLLM 模型: {args.model_path} ---")
        try:
            model = LLM(
                model=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                trust_remote_code=True,
                disable_log_stats=True
            )
            tokenizer = model.get_tokenizer()
            sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
            print("--- VLLM 模型和 Tokenizer 加载成功。")
        except Exception as e:
            print(f"加载 vLLM 模型失败: {e}")
            print(traceback.format_exc())
            sys.exit(1)
    else:  # API模型
        print(f"--- 初始化 API 模型: {args.provider}/{args.model_name} ---")
        try:
            model = APIModelWrapper(
                provider=args.provider,
                model_name=args.model_name,
                api_key=args.api_key,
                base_url=args.base_url
            )
            print("--- API 模型初始化成功。")
        except Exception as e:
            print(f"初始化 API 模型失败: {e}")
            print(traceback.format_exc())
            sys.exit(1)
    
    # 加载数据集
    print("--- 启动评估 (Agent流程 + 直接值Boxed Answer评估) ---")
    try:
        with open(args.dataset_file_path, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
        total_count = len(data_list)
        print(f"成功加载 {total_count} 条数据。")
    except Exception as e: 
        print(f"【致命错误】: 加载数据集失败: {e}")
        return
        
    # 处理已完成的条目
    processed_prompts = set()
    if os.path.exists(args.output_file_path):
        try:
            with open(args.output_file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "question" in data:
                                processed_prompts.add(data["question"])
                        except json.JSONDecodeError:
                            pass
            print(f"--- 已加载 {len(processed_prompts)} 个已处理的 prompts，将在本次运行中跳过它们。---\n")
        except Exception as e:
            print(f"--- 读取现有输出文件失败: {e}. 将从头开始处理。 ---")
            
    results_log = [] 
    success_count_this_run = 0
    
    with open(args.output_file_path, 'a', encoding='utf-8') as f_out:
        for i, item in enumerate(data_list):
            prompt = item.get("question")
            ground_truth_code = item.get("code")
            task_start_time = time.time()
            
            if prompt in processed_prompts:
                continue
                
            print(f"\n--- 正在评估第 {i+1}/{total_count} 项 (新项目) ---")
            
            if not prompt or not ground_truth_code: 
                print("    [跳过] 数据缺失 ('question' or 'code')。")
                continue
                
            print(f"    [Prompt] {prompt[:100]}...")
            
            log_entry = { 
                "task_id": i,
                "question": prompt,
                "chain_category": item.get("chain_category", "unknown"),
                "tool_count": item.get("tool_count", 0),
                        "is_correct": False, 
                        "eval_score": 0 
                        }
            
            # 执行基准代码
            success, ground_truth_answer, err = execute_simplified_code(ground_truth_code, function_mapper)
            
            if not success:
                print(f"    ❌ 基准代码执行失败: {err}")
                log_entry["ground_truth_error"] = str(err)
                f_out.write(json.dumps(log_entry, ensure_ascii=False, default=_numpy_converter) + '\n')
                f_out.flush()
                processed_prompts.add(prompt)
                continue
                
            log_entry["ground_truth"] = ground_truth_answer
            print(f"    ✅ 基准答案: {ground_truth_answer}")
            
            # 运行Agent循环
            if args.model_type == "local":
                final_response, trajectory, tool_call_history, error_count = run_agent_loop_local(
                    prompt, model, tokenizer, sampling_params, system_prompt, function_mapper, args.max_turns
                )
            else:
                final_response, trajectory, tool_call_history, error_count = run_agent_loop_api(
                    prompt, model, system_prompt, function_mapper, args.max_turns, args.max_tokens, args.temperature
                )
            
            task_end_time = time.time()
            latency = task_end_time - task_start_time
            
            log_entry["model_trajectory"] = trajectory
            log_entry["tool_call_history"] = tool_call_history
            log_entry["error_count"] = error_count
            log_entry["total_turns"] = trajectory.count("--- ASSISTANT")
            log_entry["latency"] = latency
            log_entry["conversation_history"] = []
            
            # 提取模型答案
            model_answer, err = extract_boxed_answer(final_response)
            
            if err:
                print(f"    ❌ 模型答案提取失败: {err}")
                log_entry["model_error"] = err
                log_entry["final_answer"] = None
                log_entry["abandoned"] = True
                f_out.write(json.dumps(log_entry, ensure_ascii=False, default=_numpy_converter) + '\n')
                f_out.flush()
                processed_prompts.add(prompt)
                continue
                
            log_entry["final_answer"] = model_answer
            log_entry["abandoned"] = False
            print(f"    🤖 模型答案: {model_answer}")
            
            # 比较答案
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
            
            # API模型限流
            if args.model_type == "api":
                time.sleep(0.5)
    
    # 统计结果
    total_processed_in_file = len(processed_prompts)
    processed_this_run = len(results_log)
    total_correct_in_file = 0
    
    if os.path.exists(args.output_file_path):
        with open(args.output_file_path, 'r', encoding='utf-8') as f_final:
            for line in f_final:
                if line.strip():
                    try:
                        final_data = json.loads(line)
                        if final_data.get("is_correct") is True:
                            total_correct_in_file += 1
                    except:
                        pass
                    
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
        
    print(f"详细日志已追加到: {args.output_file_path}")
    print("="*50)


# ==========================================================
# 6. 评估指标计算（来自 calculate_metrics.py）
# ==========================================================
def normalize_string(s: str) -> str:
    """规范化字符串（移除空格、标点等）"""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = s.replace(',', '')
    return s


def try_convert_type(value: Any, target_type: type) -> Optional[Any]:
    """尝试类型转换"""
    try:
        if target_type == float:
            return float(value)
        elif target_type == int:
            return int(float(value))
        elif target_type == str:
            return str(value)
    except:
        return None
    return None


def is_answer_match(model_answer: Any, ground_truth_answer: Any, tolerance: float = 1e-6) -> bool:
    """判断答案是否匹配（完全基于规则，不使用LLM）"""
    if model_answer is None:
        return False
    
    if type(model_answer) != type(ground_truth_answer):
        converted = try_convert_type(model_answer, type(ground_truth_answer))
        if converted is None:
            return False
        model_answer = converted
    
    if isinstance(ground_truth_answer, (int, float)):
        if isinstance(model_answer, (int, float)):
            if abs(ground_truth_answer) < 1e-10:
                error = abs(model_answer - ground_truth_answer)
                return error < tolerance
            else:
                relative_error = abs(model_answer - ground_truth_answer) / abs(ground_truth_answer)
                return relative_error < tolerance
    
    elif isinstance(ground_truth_answer, str):
        if isinstance(model_answer, str):
            normalized_model = normalize_string(model_answer)
            normalized_truth = normalize_string(ground_truth_answer)
            return normalized_model == normalized_truth
    
    elif isinstance(ground_truth_answer, (list, tuple)):
        if isinstance(model_answer, (list, tuple)):
            if len(model_answer) != len(ground_truth_answer):
                return False
            model_sorted = sorted(model_answer, key=str)
            truth_sorted = sorted(ground_truth_answer, key=str)
            return all(is_answer_match(m, t, tolerance) for m, t in zip(model_sorted, truth_sorted))
    
    elif isinstance(ground_truth_answer, dict):
        if isinstance(model_answer, dict):
            if set(model_answer.keys()) != set(ground_truth_answer.keys()):
                return False
            return all(is_answer_match(model_answer[k], ground_truth_answer[k], tolerance) 
                      for k in ground_truth_answer.keys())
    
    else:
        return model_answer == ground_truth_answer
    
    return False


def has_tool_call(tool_call_history: List[Dict]) -> bool:
    """判断是否至少调用了一个工具"""
    return len(tool_call_history) > 0


def are_all_tool_calls_successful(tool_call_history: List[Dict]) -> bool:
    """判断所有工具调用是否成功"""
    if len(tool_call_history) == 0:
        return False
    
    for call in tool_call_history:
        if not call.get('success', False):
            return False
        
        result = call.get('result')
        if result is None:
            return False
        
        if isinstance(result, dict):
            error_keys = ['error', 'Error', 'ERROR', 'exception', 'Exception', 'timeout', 'Timeout']
            for key in error_keys:
                if key in result:
                    return False
    
    return True


def count_tool_call_errors(tool_call_history: List[Dict]) -> int:
    """统计工具调用失败次数"""
    error_count = 0
    for call in tool_call_history:
        if not call.get('success', False):
            error_count += 1
    return error_count


def is_abandoned(result: Dict, max_turns: int = 10) -> bool:
    """判断任务是否被提前放弃"""
    total_turns = result.get('total_turns', 0)
    if total_turns >= max_turns:
        return False
    
    final_answer = result.get('final_answer')
    if final_answer is None:
        return True
    
    return False


def calculate_metrics(task_results: List[Dict], max_turns: int = 10) -> Dict[str, float]:
    """计算所有评估指标"""
    total_tasks = len(task_results)
    
    if total_tasks == 0:
        return {
            "tsr": 0.0, "faa": 0.0, "memory_cheating_rate": 0.0, "cer": 0.0,
            "ar": 0.0, "avg_eep": 0.0, "frr": 0.0, "avg_lc": 0.0
        }
    
    tsr_count = 0
    faa_count = 0
    cer_successful_tool_calls = 0
    cer_error_count = 0
    ar_count = 0
    total_eep = 0
    frr_with_errors = 0
    frr_resolved = 0
    total_latency = 0
    
    for result in task_results:
        final_answer = result.get('final_answer')
        ground_truth = result.get('ground_truth')
        tool_call_history = result.get('tool_call_history', [])
        error_count = result.get('error_count', 0)
        latency = result.get('latency', 0.0)
        
        answer_match = is_answer_match(final_answer, ground_truth)
        has_tool = has_tool_call(tool_call_history)
        if answer_match and has_tool:
            tsr_count += 1
        
        if answer_match:
            faa_count += 1
        
        all_tools_successful = are_all_tool_calls_successful(tool_call_history)
        if all_tools_successful and len(tool_call_history) > 0:
            cer_successful_tool_calls += 1
            if not answer_match:
                cer_error_count += 1
        
        if is_abandoned(result, max_turns):
            ar_count += 1
        
        if error_count > 0:
            total_eep += error_count
        else:
            total_eep += count_tool_call_errors(tool_call_history)
        
        task_error_count = error_count if error_count > 0 else count_tool_call_errors(tool_call_history)
        if task_error_count > 0:
            frr_with_errors += 1
            if answer_match and has_tool:
                frr_resolved += 1
        
        total_latency += latency
    
    metrics = {
        "tsr": tsr_count / total_tasks,
        "faa": faa_count / total_tasks,
        "memory_cheating_rate": (faa_count - tsr_count) / total_tasks,
        "cer": cer_error_count / cer_successful_tool_calls if cer_successful_tool_calls > 0 else 0.0,
        "ar": ar_count / total_tasks,
        "avg_eep": total_eep / total_tasks,
        "frr": frr_resolved / frr_with_errors if frr_with_errors > 0 else 1.0,
        "avg_lc": total_latency / total_tasks
    }
    
    return metrics


def calculate_metrics_by_category(task_results: List[Dict], max_turns: int = 10) -> Dict[str, Dict[str, float]]:
    """按任务复杂度分类计算指标"""
    from collections import defaultdict
    by_category = defaultdict(list)
    for result in task_results:
        category = result.get('chain_category', 'unknown')
        by_category[category].append(result)
    
    metrics_by_category = {}
    for category, results in by_category.items():
        metrics_by_category[category] = calculate_metrics(results, max_turns)
    
    return metrics_by_category


def calculate_metrics_by_tool_count(task_results: List[Dict], max_turns: int = 10) -> Dict[str, Dict[str, float]]:
    """按工具链长度分类计算指标"""
    from collections import defaultdict
    by_tool_count = defaultdict(list)
    for result in task_results:
        tool_count = result.get('tool_count', 0)
        by_tool_count[tool_count].append(result)
    
    metrics_by_tool_count = {}
    for tool_count, results in sorted(by_tool_count.items()):
        metrics_by_tool_count[f"tool_count_{tool_count}"] = calculate_metrics(results, max_turns)
        metrics_by_tool_count[f"tool_count_{tool_count}"]["task_count"] = len(results)
    
    return metrics_by_tool_count


def generate_metrics_report(task_results: List[Dict], max_turns: int = 10) -> Dict[str, Any]:
    """生成完整的评估报告"""
    overall_metrics = calculate_metrics(task_results, max_turns)
    metrics_by_category = calculate_metrics_by_category(task_results, max_turns)
    metrics_by_tool_count = calculate_metrics_by_tool_count(task_results, max_turns)
    
    from collections import defaultdict
    by_category = defaultdict(int)
    by_tool_count = defaultdict(int)
    for result in task_results:
        by_category[result.get('chain_category', 'unknown')] += 1
        by_tool_count[result.get('tool_count', 0)] += 1
    
    report = {
        "summary": {
            "total_tasks": len(task_results),
            "max_turns": max_turns
        },
        "overall_metrics": overall_metrics,
        "metrics_by_category": metrics_by_category,
        "metrics_by_tool_count": metrics_by_tool_count,
        "detailed_breakdown": {
            "category_statistics": dict(by_category),
            "tool_count_statistics": dict(by_tool_count)
        }
    }
    
    return report


def calculate_metrics_main(args):
    """计算指标的主函数"""
    task_results = []
    with open(args.results_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                task_results.append(json.loads(line))
    
    report = generate_metrics_report(task_results, args.max_turns)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        if args.pretty:
            json.dump(report, f, ensure_ascii=False, indent=2)
        else:
            json.dump(report, f, ensure_ascii=False)
    
    print("=" * 60)
    print("FinBoom 评估指标报告")
    print("=" * 60)
    print(f"\n总任务数: {report['summary']['total_tasks']}")
    print(f"最大轮次: {report['summary']['max_turns']}")
    print("\n总体指标:")
    print("-" * 60)
    for key, value in report['overall_metrics'].items():
        if isinstance(value, float):
            print(f"  {key.upper():25s}: {value:.4f}")
        else:
            print(f"  {key.upper():25s}: {value}")
    
    print("\n按任务复杂度分类:")
    print("-" * 60)
    for category, metrics in report['metrics_by_category'].items():
        print(f"\n  {category.upper()}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"    {key.upper():25s}: {value:.4f}")
            else:
                print(f"    {key.upper():25s}: {value}")
    
    print("\n" + "=" * 60)
    print(f"详细报告已保存到: {args.output_file}")
    print("=" * 60)


# ==========================================================
# 7. 结果对比（来自 compare_results.py）
# ==========================================================
def compare_results_main(args):
    """对比多个模型结果的主函数"""
    try:
        import pandas as pd
    except ImportError:
        print("错误: 需要安装 pandas: pip install pandas")
        sys.exit(1)
    
    from pathlib import Path
    
    all_metrics = []
    for metrics_file in args.metrics_files:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_name = Path(metrics_file).stem.replace('metrics_', '')
        metrics = {
            'Model': model_name,
            **data['overall_metrics']
        }
        all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)
    
    numeric_columns = ['tsr', 'faa', 'memory_cheating_rate', 'cer', 'ar', 'avg_eep', 'frr', 'avg_lc']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)
    
    if args.format == 'markdown':
        output = generate_markdown_comparison(df, all_metrics)
    elif args.format == 'csv':
        output = df.to_csv(index=False)
    elif args.format == 'json':
        output = json.dumps(all_metrics, indent=2, ensure_ascii=False)
    else:
        output = df.to_string(index=False)
    
    print(output)
    
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"\n对比报告已保存到: {args.output_file}")


def generate_markdown_comparison(df, all_metrics: List[Dict]) -> str:
    """生成Markdown格式的对比报告"""
    report = []
    report.append("# FinBoom 模型对比报告\n")
    report.append("## 总体指标对比\n")
    report.append("| 模型 | TSR | FAA | 记忆作弊率 | CER | AR | Avg. EEP | FRR | Avg. LC |")
    report.append("|------|-----|-----|-----------|-----|-----|----------|-----|---------|")
    
    for _, row in df.iterrows():
        report.append(
            f"| {row['Model']} | {row.get('tsr', 'N/A')} | {row.get('faa', 'N/A')} | "
            f"{row.get('memory_cheating_rate', 'N/A')} | {row.get('cer', 'N/A')} | "
            f"{row.get('ar', 'N/A')} | {row.get('avg_eep', 'N/A')} | "
            f"{row.get('frr', 'N/A')} | {row.get('avg_lc', 'N/A')} |"
        )
    
    report.append("")
    return "\n".join(report)


# ==========================================================
# 8. Benchmark对比（来自 benchmark_comparison.py）
# ==========================================================
def benchmark_comparison_main(args):
    """Benchmark对比的主函数"""
    try:
        import pandas as pd
    except ImportError:
        print("错误: 需要安装 pandas: pip install pandas")
        sys.exit(1)
    
    with open(args.finboom_metrics, 'r', encoding='utf-8') as f:
        finboom_metrics = json.load(f)
    
    other_benchmarks = []
    for item in args.other_benchmarks:
        if ':' in item:
            name, file_path = item.split(':', 1)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            other_benchmarks.append({
                "benchmark_name": name,
                "metrics": data
            })
    
    finboom_overall = finboom_metrics.get("overall_metrics", {})
    
    comparison_table = [{
        "Benchmark": "FinBoom",
        "TSR/Accuracy": f"{finboom_overall.get('tsr', 0):.4f}",
        "FAA": f"{finboom_overall.get('faa', 0):.4f}",
        "Memory Cheating Rate": f"{finboom_overall.get('memory_cheating_rate', 0):.4f}",
        "CER": f"{finboom_overall.get('cer', 0):.4f}",
        "AR": f"{finboom_overall.get('ar', 0):.4f}",
        "Avg. EEP": f"{finboom_overall.get('avg_eep', 0):.4f}",
        "FRR": f"{finboom_overall.get('frr', 0):.4f}",
        "Avg. LC": f"{finboom_overall.get('avg_lc', 0):.4f}",
    }]
    
    for other in other_benchmarks:
        other_metrics = other["metrics"]
        comparison_table.append({
            "Benchmark": other["benchmark_name"],
            "TSR/Accuracy": f"{other_metrics.get('accuracy', other_metrics.get('acc', 'N/A'))}",
            "FAA": "N/A",
            "Memory Cheating Rate": "N/A",
            "CER": "N/A",
            "AR": "N/A",
            "Avg. EEP": "N/A",
            "FRR": "N/A",
            "Avg. LC": "N/A",
        })
    
    report = {
        "comparison_table": comparison_table,
        "finboom_metrics": finboom_overall
    }
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print("FinBoom Benchmark对比分析报告")
    print("=" * 80)
    print("\n对比表:")
    print("-" * 80)
    df = pd.DataFrame(comparison_table)
    print(df.to_string(index=False))
    print(f"\n\n详细报告已保存到: {args.output_file}")
    print("=" * 80)


# ==========================================================
# 9. 参数解析
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="FinBoom 统一评估工具（支持评估、指标计算、结果对比）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行评估（本地模型）
  python eval.py evaluate --model-type local --model-path /path/to/model ...
  
  # 运行评估（API模型）
  python eval.py evaluate --model-type api --provider openai --model-name gpt-4 ...
  
  # 计算指标
  python eval.py calculate-metrics --results-file results.jsonl --output-file metrics.json
  
  # 对比多个模型
  python eval.py compare --metrics-files metrics1.json metrics2.json --output-file comparison.md
  
  # Benchmark对比
  python eval.py benchmark-comparison --finboom-metrics metrics.json --other-benchmarks name1:file1.json name2:file2.json --output-file comparison.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='运行模型评估')
    eval_parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["local", "api"],
        help="模型类型: 'local' 使用本地vLLM模型, 'api' 使用API模型"
    )
    
    # 本地模型参数
    eval_parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="本地模型的路径（仅本地模型需要）"
    )
    eval_parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小（仅本地模型需要）"
    )
    eval_parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU内存利用率（仅本地模型需要）"
    )
    eval_parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="模型最大上下文长度（仅本地模型需要）"
    )
    
    # API模型参数
    eval_parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "google", "qwen"],
        help="API提供商（仅API模型需要）"
    )
    eval_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="API模型名称（仅API模型需要）"
    )
    eval_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API密钥（可选，也可通过环境变量设置）"
    )
    eval_parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API基础URL（可选）"
    )
    eval_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度（仅API模型，默认0.0）"
    )
    
    # 通用参数
    eval_parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="最大生成token数"
    )
    eval_parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Agent循环的最大推理轮次"
    )
    eval_parser.add_argument(
        "--output-file-path",
        type=str,
        required=True,
        help="评估结果的输出文件路径"
    )
    eval_parser.add_argument(
        "--dataset-file-path",
        type=str,
        required=True,
        help="数据集路径"
    )
    eval_parser.add_argument(
        "--tool-tree-path",
        type=str,
        required=True,
        help="工具树路径"
    )
    eval_parser.add_argument(
        "--tool-desc-path",
        type=str,
        required=True,
        help="工具定义路径"
    )
    
    # 计算指标命令
    metrics_parser = subparsers.add_parser('calculate-metrics', help='计算评估指标')
    metrics_parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="任务结果文件路径（JSONL格式）"
    )
    metrics_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="评估报告输出文件路径（JSON格式）"
    )
    metrics_parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="最大推理轮次（默认：10）"
    )
    metrics_parser.add_argument(
        "--pretty",
        action="store_true",
        help="美化JSON输出"
    )
    
    # 对比结果命令
    compare_parser = subparsers.add_parser('compare', help='对比多个模型的评估结果')
    compare_parser.add_argument(
        "--metrics-files",
        type=str,
        nargs='+',
        required=True,
        help="指标文件路径列表（可多个）"
    )
    compare_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="输出文件路径（可选）"
    )
    compare_parser.add_argument(
        "--format",
        type=str,
        choices=['markdown', 'csv', 'json', 'table'],
        default='markdown',
        help="输出格式（默认：markdown）"
    )
    
    # Benchmark对比命令
    benchmark_parser = subparsers.add_parser('benchmark-comparison', help='对比FinBoom与其他benchmark')
    benchmark_parser.add_argument(
        "--finboom-metrics",
        type=str,
        required=True,
        help="FinBoom指标文件路径"
    )
    benchmark_parser.add_argument(
        "--other-benchmarks",
        type=str,
        nargs='+',
        required=True,
        help="其他benchmark指标文件路径（格式：name:file_path）"
    )
    benchmark_parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    # 验证评估命令的参数
    if args.command == 'evaluate':
        if args.model_type == "local":
            if not args.model_path:
                eval_parser.error("--model-path 是必需的（当使用本地模型时）")
        else:  # API模型
            if not args.provider:
                eval_parser.error("--provider 是必需的（当使用API模型时）")
            if not args.model_name:
                eval_parser.error("--model-name 是必需的（当使用API模型时）")
    
    return args


# ==========================================================
# 10. 主入口
# ==========================================================
if __name__ == "__main__":
    args = parse_args()
    
    if args.command == 'evaluate':
        main_evaluation(args)
    elif args.command == 'calculate-metrics':
        calculate_metrics_main(args)
    elif args.command == 'compare':
        compare_results_main(args)
    elif args.command == 'benchmark-comparison':
        benchmark_comparison_main(args)
    else:
        print("错误: 请指定一个命令 (evaluate, calculate-metrics, compare, benchmark-comparison)")
        print("使用 --help 查看帮助信息")
        sys.exit(1)
