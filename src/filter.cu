#include "filter.h"
#include "CellManager.h"
__global__ void _fast_memset_zero(int* dptr) {
    //if (threadIdx.x >= sz)return;
    dptr[threadIdx.x] = 0;
}

__global__ void _filter_by_pair(const CellAttr* cat, int* fltc, CellIndex* out, size_t memb_sz, size_t sz) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;

    if (index >= sz)return;
    if (cat[index].pair >= 0) {
        int mask = __ballot(1);
        int leader = __ffs(mask) - 1;
        int lane_id = threadIdx.x%WARP_SZ;
        int res;
        if (lane_id == leader) {
            res = atomicAdd(fltc, __popc(mask)); //globally added
        }
        res = __shfl(res, leader);
        int head = res + __popc(mask & ((1 << lane_id) - 1));
        out[head] = index + memb_sz;
    }
}

void _fastmz(int*ptr) {
    _fast_memset_zero << <1, 32 >> > (ptr);
}
void _fbp(size_t aa, const CellAttr* cat, int* fltc, CellIndex* out, size_t memb_sz, size_t sz) {
    _filter_by_pair << <aa / 128 + 1, 128 >> > (cat, fltc, out, memb_sz, sz);
}
