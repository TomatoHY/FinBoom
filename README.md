# FinBoom 💰

FinBoom 是一个专注于金融工具调用的基准测试数据集，用于评估大语言模型在金融领域的工具调用能力。

## 📋 目录

- [数据集](#-数据集-dataset)
- [工具系统](#️-工具系统-tools)
- [评估系统](#-评估系统-evaluation)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [API 密钥配置](#-关键设置api-密钥)

---

## 📊 数据集 (Dataset)

### 数据集概览

- **数据集名称**: FinBoom (Financial Benchmark)
- **任务类型**: 金融问答与工具调用（Financial QA and Tool Calling）
- **总任务数**: 624 条
- **工具链分类**:
  - **Simple**: 273道题 (43.8%) - 1个工具，单工具调用任务
  - **Moderate**: 293道题 (47.0%) - 2-4个工具，多工具组合任务
  - **Complex**: 58道题 (9.3%) - 5+个工具，长工具链任务
- **数据来源**: 自建金融工具调用基准测试数据集

### 数据集文件

数据集文件位于 `data/` 目录下：

* **`data/finboom_dataset.json`**
    * 这是项目的核心数据集文件
    * 包含：问题、用于评估的可执行代码、题型分类、工具使用信息
    * 每个任务包含以下字段：
      - `question`: 问题文本
      - `code`: 可执行的Python代码（包含工具调用）
      - `tool_count`: 工具调用次数
      - `tools_used`: 使用的工具列表
      - `chain_category`: 工具链复杂度分类（simple/moderate/complex）
      - `ground_truth`: 正确答案

* **`data/local_data_archive.pkl`**
    * 这是一个可直接使用的数据归档文件
    * 包含：股票/指数的名称和代码，以及指数的发布时间
    * 用于加速工具执行，避免重复查询

### 使用数据集

在运行评估时，需要指定数据集路径：

```bash
--dataset-file-path data/finboom_dataset.json
```

---

## 🛠️ 工具系统 (Tools)

### 工具概览

FinBoom 配备了**81个专业金融工具**，涵盖以下领域：

- **市场数据**: 股票价格、行情、K线数据
- **财务数据**: 财报、基本面数据、财务指标
- **宏观经济**: GDP、CPI、PMI等宏观经济指标
- **外汇数据**: 汇率、货币对数据
- **指数数据**: 各类股票指数、行业指数
- **分析工具**: 技术分析、情绪分析、预测工具

### 工具文件

工具系统由以下文件组成：

* **`data/tool_description.json`**
    * 包含所有工具的详细分类与描述
    * 每个工具包含：名称、描述、参数、返回值、示例

* **`data/tool_tree.json`**
    * 描述工具的三级分类树结构
    * 一级分类: 市场/宏观、个体实体、外部工具
    * 二级分类: 股票、宏观经济、外汇、指数等
    * 三级分类: 价格/行情、财报/基本面、分析/情绪/预测等

* **`data/tool_library.py`**
    * 工具的实际实现代码
    * 基于 **Akshare**、**Tushare** 及 **Ashare** 等开源财经数据接口库
    * 数据源包括：腾讯财经、东方财富、新浪财经、同花顺等

### 工具调用格式

数据集中的 `code` 字段包含可直接执行的Python代码，格式如下：

```python
from tool_library import get_stock_price

# 获取股票价格
price = get_stock_price(symbol='000001', start_date='2023-01-01', end_date='2023-12-31')
result = price['收盘'].iloc[-1]
```

---

## 📈 评估系统 (Evaluation)

FinBoom 评估系统基于**7个核心评估指标**，完全基于规则，不使用LLM-as-a-Judge，确保评估结果的客观性、可复现性和一致性。

### 评估指标

1. **TSR (Task Success Rate)**: 任务成功率 - 答案匹配 + 至少调用一个工具
2. **FAA (Final Answer Accuracy)**: 最终答案准确率 - 只评估最终答案的正确性
3. **CER (Calculation Error Rate)**: 计算推理错误率 - 工具调用成功但计算错误的比例
4. **AR (Abandonment Rate)**: 放弃率 - 未达到最大轮次就提前放弃的比例
5. **EEP (Execution Error Cost)**: 平均执行错误成本 - 工具调用阶段的平均失败次数
6. **FRR (Failure Resolution Rate)**: 失败解决率 - Agent的纠错能力
7. **LC (Latency Cost)**: 平均时间成本 - 解决每个任务所需的平均端到端时间

### 评估脚本

评估系统使用统一的 `eval/eval.py` 脚本，支持以下功能：

- **evaluate**: 运行模型评估（支持本地模型和API模型）
- **calculate-metrics**: 计算评估指标
- **compare**: 对比多个模型的评估结果
- **benchmark-comparison**: 对比FinBoom与其他benchmark

### 详细文档

**完整的评估系统使用说明请参考**: [`eval/README.md`](eval/README.md)

该文档包含：
- 本地部署模型评估指南
- API模型评估指南（OpenAI、Anthropic、Google、Qwen）
- 批量评估脚本使用
- 结果分析和对比
- 常见问题解答

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install vllm numpy pandas akshare tushare

# 确保有足够的GPU内存（本地模型）
nvidia-smi
```

### 2. 配置API密钥

在 `data/tool_library.py` 文件中配置以下API密钥：

1. **`TUSHARE_API_KEY`**
   - 获取方式: 需要在 [Tushare 官网](https://tushare.pro/) 注册获取
   - 注意: 请确保您的 Tushare 账户积分**至少有 120 分**

2. **`CURRENCY_API_KEY`**
   - 获取方式: 需要在 [currencybeacon.com](https://currencybeacon.com/) 注册获取

### 3. 运行评估

#### 本地模型评估

```bash
python eval/eval.py evaluate \
    --model-type local \
    --model-path /path/to/model \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --max-tokens 2048 \
    --max-turns 10 \
    --dataset-file-path data/finboom_dataset.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_model_name.jsonl
```

#### API模型评估

```bash
export OPENAI_API_KEY="your-api-key"

python eval/eval.py evaluate \
    --model-type api \
    --provider openai \
    --model-name gpt-4 \
    --api-key $OPENAI_API_KEY \
    --max-tokens 2048 \
    --max-turns 10 \
    --temperature 0.0 \
    --dataset-file-path data/finboom_dataset.json \
    --tool-tree-path data/tool_tree.json \
    --tool-desc-path data/tool_description.json \
    --output-file-path eval/output/results_gpt4.jsonl
```

#### 批量评估（多模型并行）

```bash
# 编辑 eval/eval.sh，配置模型列表
vim eval/eval.sh

# 运行批量评估
bash eval/eval.sh
```

### 4. 计算评估指标

```bash
python eval/eval.py calculate-metrics \
    --results-file eval/output/results_model_name.jsonl \
    --output-file eval/output/metrics_model_name.json \
    --max-turns 10 \
    --pretty
```

---

## 📁 项目结构

```
FinBoom/
├── data/                          # 工具系统文件
│   ├── tool_description.json      # 工具描述文件
│   ├── tool_tree.json             # 工具分类树
│   └── tool_library.py            # 工具实现代码
├── data/                          # 数据集和工具系统文件
│   ├── finboom_dataset.json       # 核心数据集
│   ├── local_data_archive.pkl     # 本地数据归档
│   └── ...                        # 其他备份文件
├── eval/                          # 评估系统
│   ├── eval.py                    # 统一评估脚本
│   ├── eval.sh                    # 批量评估脚本
│   ├── README.md                  # 评估系统详细文档
│   ├── output/                    # 评估结果输出
│   └── log/                       # 评估日志
├── readme.md                      # 本文件
├── FinBoom_工具分析报告.md       # 工具分析报告
├── 实验设计.md                    # 实验设计文档
└── 质量控制说明.md                # 质量控制说明
```

---

## 🔑 关键设置：API 密钥

> **重要提示**：在运行 `tool_library.py` 之前，您必须先配置好以下 API 密钥。

请在 `data/tool_library.py` 文件的顶部找到并替换以下变量：

1.  **`TUSHARE_API_KEY`**
    * **获取方式**: 需要在 [Tushare 官网](https://tushare.pro/) 注册获取
    * **注意**: 请确保您的 Tushare 账户积分**至少有 120 分**，这是使用所需接口的门槛

2.  **`CURRENCY_API_KEY`**
    * **获取方式**: 需要在 [currencybeacon.com](https://currencybeacon.com/) 注册获取

---

## 📚 相关文档

- [评估系统详细文档](eval/README.md) - 完整的评估系统使用指南
- [工具分析报告](FinBoom_工具分析报告.md) - 详细的工具使用统计和分析
- [实验设计](实验设计.md) - 实验设计和研究问题
- [质量控制说明](质量控制说明.md) - 数据集质量控制流程
