# FinBench 💰

这是一个 FinBench 项目的 README。本文档包含了关于数据集、工具和评估方法的详细说明。

## 📊 数据集 (Dataset)

本项目使用的数据主要包含以下两部分：

* `data/finbench_dataset.json`
    * 这是项目的核心数据集。
    * 包含：「问题」、「用于评估的可执行代码」以及「题型分类」。
* `data/local_data_archive.pkl`
    * 这是一个可直接使用的数据归档文件。
    * 包含：「股票 / 指数」的名称和代码，以及指数的发布时间。

---

## 🛠️ 工具介绍 (Tools)

本项目的工具集（`tool_library.py`）旨在提供丰富的财经数据访问能力。

* **数据源**：工具主要依据 **Akshare**, **Tushare** 及 **Ashare** 等开源财经数据接口库设计，数据源包括腾讯财经、东方财富、新浪财经、同花顺等。
* **工具定义**:
    * `data/tool_description.json`: 包含了所有工具的详细分类与描述。
    * `data/tool_tree.json`: 描述了工具的分类树结构。

---

## 🔑 关键设置：API 密钥

> **重要提示**：在运行 `tool_library.py` 之前，您必须先配置好以下 API 密钥。

请在 `tool_library.py` 文件的顶部找到并替换以下变量：

1.  **`TUSHARE_API_KEY`**
    * **获取方式**: 需要在 [Tushare 官网](https://tushare.pro/) 注册获取。
    * **注意**: 请确保您的 Tushare 账户积分**至少有 120 分**，这是使用所需接口的门槛。

2.  **`CURRENCY_API_KEY`**
    * **获取方式**: 需要在 [currencybeacon.com](https://currencybeacon.com/) 注册获取。

---

## 📈 运行评估 (Evaluation)

如需在本地评估模型，请执行以下脚本：

```bash
bash eval/eval.sh