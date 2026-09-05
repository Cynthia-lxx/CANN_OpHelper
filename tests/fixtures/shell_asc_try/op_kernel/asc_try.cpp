#include "kernel_operator.h"
#include "asc_try_tiling.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelAscTry {
public:
    __aicore__ inline KernelAscTry() {}
    __aicore__ inline void Init() {}
    __aicore__ inline void Process() {}
private:
    __aicore__ inline void CopyIn(int32_t progress) {}
    __aicore__ inline void Compute(int32_t progress) {}
    __aicore__ inline void CopyOut(int32_t progress) {}
};

 __global__ __aicore__ void asc_try(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingDataTemplate);
    GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, tiling_data, tiling);

}
