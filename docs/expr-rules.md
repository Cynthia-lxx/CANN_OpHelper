# CANN_OpHelper 表达式 → AscendC 规则库（expr-rules）

> 本文件是「逐元素数学表达式 → AscendC 内核代码生成」的**事实来源**：每个可识别符号都登记其
> AscendC 指令方案、常量处理约定与**官方出处**。写任何内核生成逻辑前先查本表；
> 未收录的符号一律报双语错误，禁止臆造 API。
>
> 所有出处相对本机两个只读参考目录（下文简称 `HUB`、`DOC`）：
> - `HUB` = `p:\Dev\CANN_Learning_Refs\CANN_Learning_Hub_(for_dev)\tutorials\ascendc_operator_development\`
> - `DOC` = `p:\Dev\CANN_Learning_Refs\Documentation_for_Developers\ascendc_operator_development\`

## 1. 支持符号与 AscendC 指令方案

| 数学符号 | 中缀/别名 | 元数 | Kernel 方案（就地合法） | 官方出处 |
| --- | --- | --- | --- | --- |
| 加法 | `+`、LaTeX `+` | 2 | `AscendC::Add(dst, a, b, count)` | 教学 Add golden：`CANN_OpHelper\tests\fixtures\golden\add_custom_template_op_kernel.cpp` L61；HUB `03_intermediate_vector_operator_development\src\custom_op\op_kernel\add_custom_template.cpp` |
| 减法 | `-` | 2 | `AscendC::Sub(dst, a, b, count)` | HUB `03_intermediate_vector_operator_development\answer\03.02_answer\op_kernel\sub_custom_template.cpp`（`Sub`）|
| 乘法 | `*`、LaTeX `\cdot` | 2 | `AscendC::Mul(dst, a, b, count)` | HUB `05_fused_operator_development\answer\05.03_answer\op_kernel\square_diff.cpp`（`Mul(xLocal,xLocal,xLocal)` 就地/同张量）|
| 除法 | `/`、LaTeX `\frac{a}{b}` | 2 | `AscendC::Div(dst, numerator, denominator, count)` | HUB `06_opensource_repo_operator_intro_and_contribution\answer\06.05_answer\op_kernel\sigmoid.h` L84、L102 |
| 取负 | 一元 `-` | 1 | `AscendC::Muls(dst, src, (T)-1, count)`（标量乘 -1） | sigmoid.h L80 |
| sigmoid | `sigmoid(x)` | 1 | `AscendC::Sigmoid(dst, src, count)`（**存在直接 API**）| HUB `03_intermediate_vector_operator_development\answer\03.06_answer\op_kernel\sigmoid_custom.cpp` L66 |
| exp | `exp(x)`、`e^x` 不做 v1 | 1 | `AscendC::Exp(dst, src, count)` | sigmoid.h L81/L99（就地链式）；HUB `05_fused_operator_development\answer\05.05_answer\op_kernel\matmul_sinh.cpp` |
| abs | `abs(x)` | 1 | `AscendC::Abs(dst, src, count)` | HUB `05_fused_operator_development\answer\05.04_answer\op_kernel\matmul_abs.cpp`（就地）|
| 常量填充 | 二元中任一操作数为字面量 | 1 条隐式 | `AscendC::Duplicate<T>(buf, (T)c, count)` 把常量铺成张量，再走二元 API | sigmoid.h L83、L101 |
| 张量+常量 | `x + c`（c 在右）| 折叠 | `AscendC::Adds(dst, src, (T)c, count)` | sigmoid.h L82（就地）|
| 张量×常量 | `x * c`（c 在右）| 折叠 | `AscendC::Muls(dst, src, (T)c, count)` | sigmoid.h L80 |
| 张量−常量 | `x - c` | 不折叠 | `Duplicate` + `Sub` | 同上推理（无 scalar Sub 依据）|
| 常量−张量 / 常量÷张量 / 张量÷常量 | `c - x`、`c / x`、`x / c` | 不折叠 | `Duplicate` 常量到缓冲 + `Sub`/`Div` | sigmoid.h L83→L84 即「分子填 1 → Div」的组合模式 |
| 幂 `^`/`**`、log、relu、sqrt、min/max 等 | — | — | **v1 不收录**（无逐元素官方样例依据）| — |

### 规则要点

- **就地合法**：`AscendC` 数学 API 允许 `dst` 与某个 `src` 相同（官方到处就地链式，如 sigmoid.h L80→L84、square_diff 的 `Mul(x,x,x)`）。
- **分子在前的除法**：`Div(dst, numerator, denominator, count)`——sigmoid.h L84 是 `Div(yLocal, xLocal/*1*/, yLocal/*den*/)`。
- 常量折叠只对 **`x+const`、`x*const`**（右操作数、`Adds`/`Muls`）；其它含常量二元一律 `Duplicate` 成缓冲张量后走张量二元 API。折叠与否是生成细节，与 Python 期望值无关（两条路径共用同一 AST，数值误差在验证容差内）。
- dtype：v1 为官方教学口径 `float16`/`float` 同构，模板参数经构建框架 `DTYPE_X` 多实例化；生成代码写法与官方教学 kernel 完全一致。

## 2. Kernel 代码生成基线（与教学 Add 同构）

参照 golden 成品（`tests\fixtures\golden\add_custom_template_op_kernel.cpp`，BUFFER_NUM=1 / QUEUE_DEPTH=1）：

1. `Init(GM_ADDR..., uint32_t totalLength, uint32_t tileNum)`：
   - `blockLength = totalLength / GetBlockNum()`；`tileLength = blockLength / tileNum / BUFFER_NUM`；
   - 每输入一队列 `pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength*sizeof(T))`；输出同；**每个表达式内部节点一个** `AscendC::TBuf<VECCALC> tK; pipe.InitBuffer(tK, tileLength*sizeof(float))`（教学 sigmoid.h L54-55 同款临时缓冲，Compute 内串行取用）。
2. `Process`：`loopCount = tileNum*BUFFER_NUM`，循环 CopyIn/Compute/CopyOut。
3. `CopyIn`：`AllocTensor` → `DataCopy(local, xGm[progress*tileLength], tileLength)` → `EnQue`。
4. `Compute`：`DeQue` 输入 → `AllocTensor` 输出 `out` → 执行表达式语句序列（最后一条的 `dst` = 输出缓冲，中间 `dst` = 各 scratch）→ `EnQue(out)` → `FreeTensor` 输入。
5. `CopyOut`：`DeQue` → `DataCopy(zGm[...], local, tileLength)` → `FreeTensor`。
6. 入口：模板类 + `REGISTER_TILING_DEFAULT`/`GET_TILING_DATA_WITH_STRUCT`，末尾参数表固定 `GM_ADDR workspace, GM_ADDR tiling`（golden L87-94）。

tiling 结构与 host：教学口径 `struct XxxTilingData { uint32_t totalLength; uint32_t tileNum; };`
host `TilingFunc`：`totalLength = GetInputShape(0)->GetOriginShape().GetShapeSize()`、`SetBlockDim(8)`、`tileNum=1`；
`InferShape` 输出复制输入 shape；`InferDataType` 输出取输入类型（golden host 全文）。

**v1 假设（与官方 Add 教学口径一致，非实现缺陷）**：总元素可被 `GetBlockNum()` 整除；数据 `ND`、输入输出同 shape、单输出；非整数情形由上层给出双语警告/校验提示。

## 3. 云端一键「编译 → 运行 → 数值比对」链路（官方最小模式）

出处：DOC `09_course_practice\src\09.01_testcase\testcase_1..6\run.sh`（六份逐字一致）；部署/编译单行见 HUB `05_fused_operator_development\05.04_cv_fused_operator_development.ipynb` cell（源码行 L1324/1327/1329）。

```bash
# (1) 构建 + 部署算子到 $HOME/vendors/customize
cd <msopgen 工程根>; bash build.sh
./build_out/custom_opp_*.run --install-path=${HOME}/

# (2) 编译 aclnn 单算子调用程序（host 侧 C++，CANN 自动为自定义算子生成 aclnn_<snake>.h 与 libcust_opapi）
g++ -I$ASCEND_TOOLKIT_HOME/include -I${HOME}/vendors/customize/op_api/include \
    -L$ASCEND_TOOLKIT_HOME/lib64 -L${HOME}/vendors/customize/op_api/lib \
    <run>/aclnn_test.cpp -lcust_opapi -lnnopbase -lacl_rt -o <run>/execute_op

# (3) 运行并把 device 输出落盘 output.bin
source ${HOME}/vendors/customize/bin/set_env.bash
<run>/execute_op .      # 程序从 <run> 读 input.bin、写 output.bin

# (4) 与 helper 预生成的 golden.bin 比对
python3 <run>/verify_result.py
```

> 注：本工具生成的 `verify/run_verify.sh` 在第 (1) 步之后额外内置 **soc 目录别名步骤**——CANN 9.0.0 的 910B 系云 Lab 设备 runtime 报 socVersion `ascend910_93`，而算子按 `ascend910b` 编译安装（落 `kernel/{ascend910b, config/ascend910b}`）；NNOP 只按设备 soc 查 `kernel/config/ascend910_93/binary_info_config.json`，目录缺失会 `GetWorkspaceSize → 161001 regInfo failed`。别名步骤在目标目录不存在时 `cp -r` 镜像一份（同 DAV_2201 arch，.o 可互换；build+重装会清掉，故每次运行都重做）。完整案例见 `.codebuddy/memory/ascendc_07_custom_op_project_practice.md` §8。

### aclnn 调用形态（生成器参照）

- 接口名 = `aclnn<OpCamel>`，三件套：`aclnn<Op>GetWorkspaceSize(...)` / `aclnn<Op>(workspace, workspaceSize, executor, stream)`，头文件 `aclnn_<op_snake>.h`；见 DOC 09.01 testcase_6 `aclnn_test.cpp` L143-157（示例为 `LogSigmoidCustom`）。
- 数据通路：`aclrtMalloc` → `aclrtMemcpy(H2D)` → `aclCreateTensor(..., ACL_FORMAT_ND, dtype)`；执行后 `aclrtMemcpy(D2H)` → 写 `output.bin`。样例见 DOC 09.01 testcase_6 `aclnn_test.cpp` 全文与 HUB 03.06 `test/aclnn_test.cpp`。
- 官方样例把 shape/dtype/host 数据类型收敛进 `case_config.h`（`HostDataType`/`CASE_SHAPE`/`CASE_ACL_DTYPE`，DOC 09.01 testcase_6 `case_config.h`），gen/verify 与之配套：`gen_data.py`（云端 numpy 造 input/golden）→ `execute_op` → `verify_result.py`。
- 比对口径（官方 verify_result.py）：`np.isclose(rtol=1e-3, atol=1e-3, equal_nan=True)`。
- 本工具策略差异：**input.bin 与 golden.bin 由 helper 在本地生成**（确定性随机 + 自研 AST 求值器，不依赖云端 numpy）；verify_result.py 改为纯标准库实现等价口径，保证云端 `python3` 免装包即可比对。

## 4. 出处文件清单

- Golden 教学三件：`CANN_OpHelper\tests\fixtures\golden\add_custom_template_op_kernel.cpp` / `..._op_kernel_tiling.h` / `..._op_host.cpp`
- sigmoid 组合实现 + 临时 TBuf + 常量 Duplicate：HUB `...06.05_answer\op_kernel\sigmoid.h` L44-107
- Sigmoid 直接 API 教学 kernel/host：HUB `...03.06_answer\op_kernel\sigmoid_custom.cpp` L61-100、`op_host\sigmoid_custom.cpp`
- 双输入队列 + Sub 教学：HUB `...03.02_answer\op_kernel\sub_custom_template.cpp`
- Abs 就地：HUB `05.04_answer\op_kernel\matmul_abs.cpp`；Exp 就地：HUB `05.05_answer\op_kernel\matmul_sinh.cpp`
- 运行链路：DOC `09_course_practice\src\09.01_testcase\testcase_6\{run.sh, aclnn_test.cpp, case_config.h, gen_data.py, verify_result.py}`
- 部署单行：HUB `05_fused_operator_development\05.04_cv_fused_operator_development.ipynb`

## 5. v1 边界（诚实声明）

- 符号集：`+ - * /` 二元、一元 `-`、`sigmoid/exp/abs`；常量（含 `\frac{c}{x}` 形态）；括号；LaTeX 子集。幂/对数/relu/min/max/多输出/广播/规约不做 v1（架构留扩展点）。
- dtype：按 spec 声明，全部张量要求同一 dtype（float 或 float16），ND、单输出、同 shape。
- **整除声明仅约束 float 默认路径**（官方 Add 教学口径）；float16 路径使用 32B 块尾块 tiling，不再要求总元素被核数整除——见 §6。

## 6. P1：float16 与 32B 块尾块契约（fillgen B 系统，2026-09-05 落地）

- **dtype 域**：`{float, float16}` 单一且跨张量一致；float→kernel `float`/host `DT_FLOAT`，float16→kernel `half`/host `DT_FLOAT16`；int8/int32/bf16 拒（i18n `fillgen.err.*`）。常量折叠 cast 随 dtype（`(half)2` / `(float)2`）。
- **shape hint（可选）**：`TensorSpec.shape` 只驱动 verify 元素数 = 全部显式 shape 的乘积；所有显式 shape 必须一致（否则 `shape_conflict`）、不得含动态维 `-1`（否则 `shape_dynamic`）；无 hint 时验证回退默认 8×2048。msopgen 命令与原型 JSON 忽略 shape（官方 proto 本就不含 shape）。
- **float16 tail tiling**（生成的三件套均为该结构）：
  - TilingData：`totalLength / bigDataNum / smallDataNum / tailBlockNum`（uint32×4）。
  - host：`totalBlocks=totalLength/16`；`perCoreBlocks=totalBlocks/8`；`tailBlockNum=totalBlocks%8`；`bigDataNum=(perCoreBlocks+(tailBlockNum>0?1:0))*16`；`smallDataNum=perCoreBlocks*16`。
  - kernel：`blockIdx < tailBlockNum` 为大核（`dataNum=bigDataNum`），GM 偏移大核 `blockIdx*bigDataNum`、小核 `tailBlockNum*bigDataNum+(blockIdx-tailBlockNum)*smallDataNum`；每核 32B 对齐；`Process` 单趟 CopyIn→Compute→CopyOut（无核内二次切片）。
- **边界（诚实声明）**：尾块粒度 = 32B 块（fp16 为 16 元素/块）；生成器假定总元素是 32B 块整数倍——DataCopy 最小搬运粒度 32B，真正的「<32B 残尾」需要 `DataCopyPad`，P1 不生成，留待后续阶段。用例 shape hint 应满足该前提（如 416=26×16）。
- **fp16 验证**：输入/golden 以 `struct '<e'`（RNE）打包与舍入；golden 解释器每语句 round 到 fp16（与 kernel 指令同精度）；runner host 类型 `uint16_t` + `ACL_FLOAT16`；`verify_result.py` 以 `<e` 解包，判据沿用 rtol=atol=1e-3。
- **回归防线**：float 默认路径文本与旧版逐字节一致（229→232 既有用例全绿）；新能力仅由 dtype=float16 或非整除 32B 块 shape 触发。
