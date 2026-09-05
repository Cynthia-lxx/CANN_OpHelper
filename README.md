# CANN_OpHelper

Windows 10 本地运行的 Python CLI 工具，用于**辅助生成 CANN Ascend C 算子工程模板代码**。

> 本项目不编译、不运行任何 C++ 代码；本地无 C/C++ 编译器、无 CANN 套件/NPU 驱动。所有编译与运行验证均在**云端 CANN Lab** 完成。

## 工作流

```
用户输入算子描述 ──► 生成 msopgen 命令 ──► 用户复制到云端执行，得到初始工程
                                                     │
                                                     ▼
           输出修改后完整工程 ◄── Jinja2 填充/修改 ◄── 用户将初始工程复制到本地
```

1. 用 `new-op` 交互向导（或 `--from add` 预设）快速生成算子描述 YAML，或手工编写 YAML（算子名、表达式、输入输出、dtype、形状、soc 等）。
2. 工具生成一条完整的 `msopgen` 命令并在终端展示，同时记录算子元信息。
3. 用户在云端 CANN Lab 执行该命令，获得 msopgen 生成的初始工程，复制到本地。
4. 工具读取该工程，依据元信息与模板用 Jinja2 填充 Kernel 侧 `Compute` 核心逻辑、完善 Host 侧 Tiling（当前支持核间/核内均分）、按需注入调试语句。
5. 输出修改后的**完整工程**目录，供用户上传云端编译验证。

## 功能边界

- 不做 C++ 语法/语义分析（无本地编译器）、不模拟 NPU。
- 不生成"从零开始"的完整工程——初始框架依赖官方 `msopgen`。
- 原型 JSON 生成边界（规则 6.4）：允许**基于已校验算子描述（OpSpec/YAML）的机械导出**为 msopgen `-i` 原型文件；**不臆造**无官方依据的字段（如 attr 布局，在官方样例确认前一律不导出）。
- 可选的文本级检查：必要头文件是否齐全、计算 API 名称是否正确（仅正则/包含匹配）。

## 安装（本地，Windows）

嵌入式 Python 3.14.5 位于 `P:\Software\python-3.14.5-embed-amd64`（快捷入口 `P:\python.bat` / `P:\python.ps1`）。

项目**复用工作区根下已配置好的虚拟环境 `penv`**（无点号，位于 `p:\Dev\CANN_Learning_Refs\penv`，嵌入式 Python 3.14.5 + pip 已就绪）。**不创建新的 `.penv`**。

```powershell
# 在项目根目录（CANN_OpHelper）执行——脚本定位既有 penv 并执行 pip install -e ".[dev]"：
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_venv.ps1
# 或直接运行
.\scripts\bootstrap_venv.cmd
# 也可手工在 penv 中安装（venv python 位置）：
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m pip install -e ".[dev]"
```

激活环境（该 venv 位于工作区根，非项目内）：

```powershell
p:\Dev\CANN_Learning_Refs\penv\Scripts\Activate.ps1
```

## 使用示例

CLI 提供三个子命令（`cann-ophelper` 已注册为 console script，激活 `penv` 后可直呼）：

| 子命令 | 作用 |
| --- | --- |
| `new-op [--from add] [--yes] [--out <yaml>]` | 交互逐问收集算子需求：算子类型、SoC、描述、输入/输出张量及每个张量的 name/param_type/dtype×format（按官方并行数组一次输入逗号分隔列表）、可选 shape，边问边校验。`--from add` 用内置 Add 预设预填每项、回车即确认。确认后落盘 OpSpec YAML。 |
| `gen-msopgen <yaml> [--proto-out <json>]` | 校验并预览算子元信息，拼装一条完整的 `msopgen` 命令与云端执行说明（本身无副作用）。加 `--proto-out` 会把 spec **无损导出**为官方原型 JSON 落盘，供上传云端作 `-i`。省略 `--proto`/`--out` 时沿用官方示例的相对路径默认值。 |
| `render <yaml> --out <目录>` | 按模板渲染 `op_kernel`/`op_host` 三产物并写入 `--out` 目录（覆盖该目录下已有同名文件，常用于覆盖云端拷回的 msopgen 工程）；省略 `--out` 或加 `--dry-run` 时仅预览不落盘。 |

```powershell
# 帮助与版本
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper --help

# 交互向导生成自己的算子 YAML（默认 zh 输出；--lang en 切换英文）
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper new-op

# 从 Add 预设起步：回车一路确认预填值，--yes 跳过最终确认
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper new-op --from add --yes --out examples\my_add.yaml

# 用 YAML 生成 msopgen 命令
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper gen-msopgen examples\my_add.yaml
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper --lang en gen-msopgen examples\my_add.yaml

# 导出官方原型 JSON（上传云端后作为 msopgen -i 的原型文件）
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper gen-msopgen examples\my_add.yaml --proto-out add_custom.json

# 渲染三产物：预览（不写盘）或写入指定工程目录（覆盖 op_host/op_kernel）
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper render examples\my_add.yaml
p:\Dev\CANN_Learning_Refs\penv\Scripts\python.exe -m cann_ophelper render examples\my_add.yaml --out out\AddCustomTemplate
```

工作流第 1~4 步的操作入口：

1. `new-op` 生成算子 YAML（也可手工编写）。
2. `gen-msopgen <yaml> --proto-out <json>` 导出官方原型 JSON 并展示命令与云端步骤。
3. 复制命令到云端 CANN Lab 执行（`-i` 指向刚导出的 JSON），获得初始工程并拷回本地。
4. `render ... --out <本地工程目录>` 将三产物写回该工程，随后整体上传云端编译验证。

## 目录结构

- `src/cann_ophelper/`：工具源码（CLI / 模型 / msopgen 生成 / 模板引擎 / tiling 策略 / apply 流水线 / 文本检查）。
- `templates/elementwise/`：Element-Wise 算子的 Kernel / Host 模板（内容源自官方示例的固定模式）。
- `examples/`：算子描述 YAML 示例。
- `docs/official-patterns.md`：官方模式基准（msopgen 命令格式、文件布局、固定代码写法）。
- `scripts/`：环境引导脚本。

## 参考（只读，不修改）

- 官方文档：`../Documentation_for_Developers/`
- 官方示例：`../CANN_Learning_Hub_(for_dev)/`
