#include "define.h"
#include <thrust/device_vector.h>
#define WARP_SZ (32)
class CellAttr;
template<CELL_STATE...state_Or>
__global__ void _filter_by_state(const CELL_STATE* cst, int* fltc, CellIndex* out, size_t memb_sz, size_t sz) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;

    if (index >= sz)return;
    using expand_type = int[];
    bool sor = false;
    (void)expand_type {
        0, (sor = sor || (cst[index] == state_Or), void(), 0)...
    };
    if (sor) {
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
void _fbp(size_t aa, const CellAttr* cat, int* fltc, CellIndex* out, size_t memb_sz, size_t sz);




void _fastmz(int*ptr);
template<unsigned N>
class CellIndexFilter {

    int* flt_count_head;
    thrust::device_vector<int> vec[N];
    size_t _size;
    int last_num[N];
public:
    
    CellIndex* get_ptr(int idx) {
        return thrust::raw_pointer_cast(vec[idx].data());
    }
    int get_last_num(int idx)const {
        return last_num[idx];
    }

    const int* get_last_num_ptr(int idx)const {
        return &flt_count_head[idx];
    }
    void resize(size_t size) {
        _size = size;
        for (int i = 0; i < N; i++) {
            vec[i].resize(size);
        }
    }
    void init(size_t size) {
        resize(size);
        
    }
    CellIndexFilter(size_t size) {
        init(size);
        cudaMalloc((void**)&flt_count_head, sizeof(int)*32);
    }
    CellIndexFilter() {
        cudaMalloc((void**)&flt_count_head, sizeof(int)*32);
    }

    void initialize_filtering() {
        _fastmz(flt_count_head);
    }
    template<CELL_STATE...state_Or>
    void filter_by_state(int idx,const CELL_STATE* cst, size_t nm_start) {
        
        _filter_by_state<state_Or...> << <_size / 128 + 1, 128 >> >(cst, &flt_count_head[idx], 
            get_ptr(idx), nm_start, _size);

        
    }

    void filter_by_pair(int idx, const CellAttr*cat, size_t nm_start) {

        _fbp(_size,cat, &flt_count_head[idx],
            get_ptr(idx), nm_start, _size);


    }



    void finish_filtering() {
        //cudaMemcpy(&last_num[0], flt_count_head, sizeof(int)*N, cudaMemcpyDeviceToHost);
    }
};

