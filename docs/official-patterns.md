# CANN_OpHelper —— 官方模式基准（Official Patterns Baseline）

> 本文档是本工具生成/修改算子工程代码时的**事实基准**：所有命令、目录结构、代码模式均逐条提取自本地官方文档与官方示例（只读参考目录），并标注出处供复核。
> 若本文档与任何外部知识冲突，以本文档所引的本地官方原文为准；若官方原文变更，请先更新本文档再修改工具。
>
> 更新日期：2026-09-04

## 0. 出处（Sources）

| 代号 | 相对路径（以 `p:\Dev\CANN_Learning_Refs\` 为根） | 内容 |
|---|---|---|
| [D1] | `Documentation_for_Developers\ascendc_operator_development\03_intermediate_vector_operator_development\03.02_operator_engineering_intro.ipynb` | 《工程化算子开发介绍》：msopgen 定位、命令与参数、生成工程目录结构 |
| [D2] | 同章 `src\add_custom.json` | msopgen 输入：AddCustomTemplate 算子原型定义（官方真实样例） |
| [S1] | 同章 `src\custom_op\op_kernel\add_custom_template.cpp` | msopgen 生成的 Kernel 侧算子实现（含调试 printf） |
| [S2] | 同章 `src\custom_op\op_kernel\add_custom_template_tiling.h` | 生成的 Tiling 结构体头文件 |
| [S3] | 同章 `src\custom_op\op_host\add_custom_template.cpp` | 生成的 Host 侧：原型注册 / TilingFunc / InferShape / InferDataType |
| [S4] | 同章 `src\custom_op\test\main.cpp` | 生成工程自带的调用测试（ACL/aclnn 高层接口） |
| [S5] | 同章 `src\custom_op\build.sh`、`test\run.sh` | 编译/运行脚本模式 |
| [D3] | `Documentation_for_Developers\ascendc_operator_development\03_intermediate_vector_operator_development\03.03_acl_pybind_call.ipynb` | 工程化算子的调用与验证（ACL/pyacl）上下文 |
| [D4] | `Documentation_for_Developers\ascendc_operator_development\09_course_practice\09.01_vector_ops_practice.ipynb` | msopgen `-c ai_core-ascend910b1` 的另一处一致用法 |

> 工具产出代码以官方 AddCustomTemplate 工程（S1–S5）为**推荐基准形态**；文档 D1/D4 为命令与流程基准。

---

## 1. msopgen 工具

### 1.1 定位

- CANN 开发套件提供自定义算子工程生成工具 **msOpGen**，可根据**算子原型定义（JSON）**生成基础算子工程，包括 Host 侧代码、Kernel 侧代码及工程编译配置文件；开发者在其上继续实现与编译。`[D1 章节：什么是算子工程]`
- **CANN 9.0.0 自带 msOpGen**，无需源码编译安装；可直接验证 `msopgen -h`。`[D1]`

### 1.2 标准命令（官方原文）

```
msopgen gen -i Sources/03.02/add_custom.json -c ai_core-ascend910b1 -lan cpp -out Sources/03.02/custom_op
```

出处：`[D1 章节：创建算子工程]`；`-c ai_core-ascend910b1` 亦见于 `[D4]`。

参数含义（官方原文整理）：`[D1 命令参数含义]`

| 参数 | 含义 | 取值要点 |
|---|---|---|
| `-i` | 指定算子原型定义文件（JSON）所在路径 | 按实际路径修改 |
| `-c` | 指定算子对应执行的昇腾 AI 处理器 | 格式 `ai_core-<soc>`，如 `ai_core-ascend910b1` |
| `-lan` | 开发语言框架 | `cpp`＝基于 Ascend C、使用 C/C++ 开发 |
| `-out` | 生成算子工程所在路径 | 绝对或相对路径均可 |

- 本资料集中 msopgen 调用**仅出现 `-i / -c / -lan / -out`**，未发现 `-t`、`ccec` 等其它选项（不代表工具没有，仅本套教材未使用）。`[D1][D4]`
- soc 大小写：msopgen 的 `-c` 在本套教材统一小写 `ai_core-ascend910b1`。大写（如 `Ascend910B4`）仅见于 CPU 仿真等其它工具参数，**不属于 msopgen**，勿混用。`[D1][D4]`

### 1.3 算子原型 JSON（msopgen 的 `-i` 输入）—— 只引用、不生成

官方真实样例 `add_custom.json` 全文：`[D2]`

```json
[{
    "op": "AddCustomTemplate",
    "input_desc": [{
            "name": "x",
            "param_type": "required",
            "format": ["ND", "ND"],
            "type": ["float16", "float"]
        },
        {
            "name": "y",
            "param_type": "required",
            "format": ["ND", "ND"],
            "type": ["float16", "float"]
        }
    ],
    "output_desc": [{
        "name": "z",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["float16", "float"]
    }]
}]
```

要点（由样例归纳）：
- 顶层是数组，每元素描述一个算子（含 `op`、`input_desc`、`output_desc`，可扩展 `attr_desc`）。
- `param_type`：`required` / `optional`（描述参数是否必选，非方向；方向由 input/output_desc 区分）。
- `format` / `type` 为**数组**，表示该输入支持的多组 format 与 dtype（按位一一对应，如 `["float16","float"]` 支持 fp16 与 fp32）。
- **本工具规约（6.4）：不生成、不修改 JSON 原型；只引用用户提供的 JSON 路径。**

### 1.4 msopgen 生成的工程目录结构

官方目录树（`[D1 算子工程目录结构]`，与实测 `src\custom_op\` 一致并补充实测项）：

```
custom_op/
├── framework/                    # 框架适配（如 tf_plugin 子模块；实测存在）[实测]
├── op_host/
│   ├── add_custom_template.cpp   # Host 侧：算子原型注册 + Tiling 实现 + Shape/Dtype 推导
│   └── CMakeLists.txt            # Host 侧构建文件，一般不用改
├── op_kernel/
│   ├── add_custom_template_tiling.h   # 算子 Tiling 结构体定义
│   ├── add_custom_template.cpp        # Kernel 侧算子代码实现
│   └── CMakeLists.txt                 # Kernel 侧构建文件，一般不用改
├── test/                         # 调用验证样例（main.cpp/run.sh/CMakeLists，实测存在）
├── test_sub/
├── CMakeLists.txt                # 根构建文件
├── CMakePresets.json             # CMake 编译配置（一般只需改 ASCEND_CANN_PACKAGE_PATH）
└── build.sh                      # 编译脚本
```

- 文件名根据 JSON 内算子名生成（`add_custom.json` 中 `op: "AddCustomTemplate"` → `add_custom_template.*`，转小写下划线）。`[D1][S1][S3]`
- `op_host` 与 `op_kernel` 含算子核心实现；`CMakePresets.json` 一般只需修改 `ASCEND_CANN_PACKAGE_PATH`。`[D1]`

---

## 2. Kernel 侧固定代码模式（基准：`op_kernel\add_custom_template.cpp` `[S1]`）

### 2.1 头文件与常量

```cpp
#include "kernel_operator.h"
#include "add_custom_template_tiling.h"              // 本算子 Tiling 结构体
#include "kernel_operator_dump_tensor_intf_impl.h"   // 使用 printf 调试时引入
constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue
constexpr int32_t QUEUE_DEPTH = 1;
```

### 2.2 Kernel 类骨架（模板化，按输入/输出 dtype 实例化）

```cpp
template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();          // 核间均分
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;       // 核内均分
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        AscendC::printf("Core %ld executed %d times in total\n", AscendC::GetBlockIdx(), loopCount);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);             // ← Element-Wise 计算核心
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX;   // VECIN：输入
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueZ; // VECOUT：输出
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};
```

固定四段式：`Init`（按核均分算量、绑定 GM/Local 缓冲）→ `CopyIn`（GM→Local 入队）→ `Compute`（出队→AscendC 计算 API→入队）→ `CopyOut`（出队→写回 GM）。

### 2.3 核函数入口（全局函数 + tiling 数据装配）

```cpp
__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingDataTemplate);
    GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;      // DTYPE_* 由构建宏按实际数据类型注入
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
```

- 核函数形参固定为 `GM_ADDR ... + workspace + tiling` 后缀。`[S1]`
- `DTYPE_X/DTYPE_Y/DTYPE_Z` 为构建系统注入的宏，模板据此实例化。`[S1]`

### 2.4 Tiling 结构体头（`op_kernel\*_tiling.h` `[S2]`）

```cpp
#ifndef ADD_CUSTOM_TEMPLATE_TILING_H
#define ADD_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

struct TilingDataTemplate {
    uint32_t totalLength;
    uint32_t tileNum;
};
#endif // ADD_CUSTOM_TEMPLATE_TILING_H
```

本样例只做核间均分 + `tileNum=1` 的核内单块（`tileNum` 由 Host 侧写入）。

---

## 3. Host 侧固定代码模式（基准：`op_host\add_custom_template.cpp` `[S3]`）

### 3.1 include 与 TilingFunc（Tiling 计算）

```cpp
#include "register/op_def_registry.h"
#include "../op_kernel/add_custom_template_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(8);
    TilingDataTemplate *tiling = context->GetTilingData<TilingDataTemplate>();
    tiling->totalLength = totalLength;
    tiling->tileNum = 1;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling
```

要点：从 `context->GetInputShape(0)` 取元素总数 → `SetBlockDim(核数)` → 填充 Tiling 结构体到 `GetTilingData<T>()`。`[S3]`

### 3.2 InferShape / InferDataType

```cpp
namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;               // Element-Wise：输出形状 = 输入形状
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));   // 输出 dtype = 输入 dtype
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge
```

### 3.3 算子原型注册（OpDef）

```cpp
namespace ops {
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");     // 处理器配置名（与 msopgen -c 的 soc 对应但写法不同）
    }
};
OP_ADD(AddCustomTemplate);
}  // namespace ops
```

要点：`.Input/.Output` 链式声明名称 → `ParamType(REQUIRED)` → `DataType({...})`（对应 JSON `type` 数组的 `ge::DT_*` 写法）→ `Format({...})`（对应 JSON `format` 数组）→ 挂 Infer 函数 → `AICore().SetTiling(...).AddConfig("ascend910b")` → `OP_ADD(...)` 注册。`[S3]`

> 注意 `ge::DT_FLOAT` 对应 JSON 中的 `"float"`；`AddConfig("ascend910b")` 与 msopgen `-c ai_core-ascend910b1` 写法不同（Config 少版本尾号、无前缀）。本工具在元信息中记录**基础 soc**（如 `ascend910b1`），命令生成与 Config 生成按各自格式拼装。

---

## 4. 构建与运行验证

- 编译：`custom_op/build.sh` 使用 CMake preset 构建（`cmake -S . -B <build> --preset=default` → `cmake --build ... --target binary/目标`）。`[S5]`
- 上板测试：工程自带 `test/main.cpp`，用 ACL/aclnn 高层接口（`#include "aclnn_add_custom_template.h"`、`aclInit/aclrtMalloc/aclCreateTensor/aclnn...`），经 `test/run.sh` 编译并运行（需 `source setenv.bash`、设置 `LD_LIBRARY_PATH` 含 `opp/vendors/customize/op_api/lib`）。`[S4][S5]`
- 其它调用方式：ACL pyacl（见 03.03 章 `[D3]`），属本工具后续轮的可选演示目标。

---

## 5. 官方"开发→验证"主流程与本工具插入点

综合 `[D1][D3]` 官方教材流程，本工具在两个阶段介入：

```
阶段一（生成工程）：算子描述(JSON 原型) ──msopgen──► 初始算子工程(custom_op)
                              ▲ 本工具在此输出 msopgen 命令行 + 执行说明（用户云端执行）
阶段二（实现算子）：在初始工程上补全 Kernel Compute / Host Tiling 逻辑 ──► 编译 ──► 上板验证
                              ▲ 本工具在此读取工程、按模板填充/修改（后续轮实现）
```

本工具**阶段一只产出命令文本**；阶段二为后续轮次的模板引擎/apply 流水线目标。官方样例（S1）的 Kernel `Compute` 已含 `AscendC::Add(zLocal, xLocal, yLocal, tileLength)` 与调试 `AscendC::printf`，即为后续 Element-Wise 模板填充的最小范例。

---

## 6. 复用要点速查（供 tool 开发直接引用）

1. msopgen 命令拼装格式：`msopgen gen -i <json> -c ai_core-<soc> -lan cpp -out <out_dir>`。`[D1]`
2. Kernel 固定四段式与 `REGISTER_TILING_DEFAULT / GET_TILING_DATA_WITH_STRUCT`。`[S1]`
3. Host 侧三段：`optiling::TilingFunc` + `ge::InferShape/InferDataType` + `ops::Xxx : OpDef` 注册。`[S3]`
4. Tiling 结构体文件由 Kernel/Host 共享（Host 写、Kernel 读）。`[S1][S2][S3]`
5. dtype/format 在 JSON（字符串小写）、Host（`ge::DT_*`/`ge::FORMAT_*`）、Kernel（`DTYPE_*` 宏）三处写法不同——工具需维护**对应映射表**（后续轮）。
6. 生成文件/类名规则：`OpType`（PascalCase）→ 文件与函数名 `op_type`（snake_case）。`[D1][S1]`

---

## 7. template-engine 整文件模板基准（2026-09-04 落地）

阶段二地基轮：`src/cann_ophelper/template/`。本轮只做 **AddCustomTemplate 单点 + 整文件模板**；片段拼装留待 apply-pipeline 轮。

### 7.1 基准文件与产物映射

渲染一个 OpSpec（AddCustomTemplate 形态）得到三个整文件，逐字对照官方：

| 产物（engine 输出 relpath） | 模板（包内逻辑名） | 官方对照基准 |
| --- | --- | --- |
| `op_kernel/<op_snake>.cpp` | `templates/op_kernel/kernel.cpp.j2` | [S1] `op_kernel/add_custom_template.cpp` |
| `op_kernel/<op_snake>_tiling.h` | `templates/op_kernel/tiling.h.j2` | [S2] `op_kernel/add_custom_template_tiling.h` |
| `op_host/<op_snake>.cpp` | `templates/op_host/host.cpp.j2` | [S3] `op_host/add_custom_template.cpp` |

### 7.2 命名 / AddConfig 暂定规则（有证据但非全仓通用，勿虚构推导）

已检索官方样例全部 13 处 Kernel 类、62 处 TilingData 结构体、20 处 AddConfig，**命名跨章节不一致**：

- 03 章 msopgen 模板工程路线：类名 `KernelAdd` = `Kernel` + OpType 去尾部 `CustomTemplate`/`Custom` 核心名；tiling 结构体为常量 `TilingDataTemplate`（与算子名无关）。
- 开源仓路线（示例见 devkit/昇腾社区）：`<Op>TilingData` 等不同，两套路线并存。

故本轮只落地与 03 章 AddCustomTemplate 一致的显式/单点规则（源码注释已标注暂定）：

- `naming.kernel_class`：`KERNEL_SUFFIXES = ("CustomTemplate", "Custom")` 去尾缀后加 `Kernel` 前缀。
- `naming.tiling_struct`：常量 `TilingDataTemplate`（仅对 msopgen 模板工程路线成立）。
- `maps.opdef_soc`：显式映射表 `{"ascend910b1": "ascend910b"}`；未收录 soc 抛错，**不做「去版本尾号」式通用推导**。
- `maps.ge_dtype/ge_format`：小表仅收录官方样例确认项（`float16`/`float` → `ge::DT_FLOAT16`/`ge::DT_FLOAT`；`ND` → `ge::FORMAT_ND`）；未知输入抛 `OpSpecError`（i18n hint 指引先登记 maps 并记录出处）。

### 7.3 渲染 vs 官方 S1–S3 的逐字差异清单（有记录修正，勿当作 bug）

产物与官方原文仅差下列归一化修正（`tests/test_template_engine.py::normalize_official` 同款）：

1. 官方 host 拼写笔误 `intputShape` → `inputShape`。
2. 官方 kernel `__global__` 行首多余空格清除。
3. 官方 kernel `printf(",  AscendC::GetBlockIdx(), ...)` 逗号后双空格清为单空格。
4. 统一 EOF 换行：官方 `tiling.h` 文件末尾无换行，规范化为有（其余两文件官方本就有）。

另：渲染 `.DataType({ ... })` / `.Format({ ... })` 花括号收紧为无内空格（`{{- -}}`），与官方逐字一致。

### 7.4 引擎行为与后续改造约定

- Jinja2 Environment：`keep_trailing_newline=True`、`trim_blocks=True`（仅去掉 `{% %}` 块标签后随行换行，避免 `{% set %}` 在产物顶部留空行）、`lstrip_blocks=False`（不动缩进）。
- Add 形态假设：两输入 x/y → 一输出 z；模板正文逐字官方四段式，张量 token 经 `{% set x = inputs[0] %}` 绑定。
- 逐字对齐回归双保险：golden 快照（主回归手段，自包含不依赖大目录）+ 官方 S1–S3 实样对齐（次，只读 3 个精确文件，运行期不扫目录）。
- **后续 apply-pipeline 轮拆片段时，只改 `templates/` 与 `engine.py`，不动 `context.py`/`maps.py`/`model.py`**（render context 是模板变量契约唯一来源）。

### 7.5 已实现自检

命名/映射/上下文/引擎测试与全量回归 94/94 通过；golden 快照 3 件套入库。
