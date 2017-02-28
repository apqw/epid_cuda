#include "define.h"
#include "CubicDynArr.h"
#include "CellManager.h"
#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"
#include "math_helper.h"
#include "map.h"
static constexpr int irx_nm = (int)(NON_MEMB_RAD*NX / LX);
static constexpr int iry_nm = (int)(NON_MEMB_RAD*NY / LY);
static constexpr int irz_nm = (int)(NON_MEMB_RAD*NZ / LZ);

static constexpr int irx_m = (int)(MEMB_RAD*NX / LX);
static constexpr int iry_m = (int)(MEMB_RAD*NY / LY);
static constexpr int irz_m = (int)(MEMB_RAD*NZ / LZ);

#define NMAFM_THREAD_NUM (128)
#define NMAFM_MULTI (32)


__global__ void map_gen_memb_non_afm(const cudaTextureObject_t pos_tex, const CellIndex* m_nonafm_filtered,CubicDynArrAccessor<int> cmap1, CubicDynArrAccessor<CMask_t> cmap2,size_t memb_sz,size_t sz) {

    const int th_id = threadIdx.x%NMAFM_MULTI;
    const int b_id = threadIdx.x / NMAFM_MULTI;
    const int _index = threadIdx.x / NMAFM_MULTI + blockIdx.x*blockDim.x / NMAFM_MULTI;

    __shared__ real4 mycp[NMAFM_THREAD_NUM / NMAFM_MULTI];
    __shared__ int myidx[NMAFM_THREAD_NUM / NMAFM_MULTI];
    __shared__ real myradsq[NMAFM_THREAD_NUM / NMAFM_MULTI];
    if (_index >= sz)return;
    static_assert(irx_m == 2 && iry_m == 2, "ir*_num must be 2");
    static constexpr int width = irx_m * 2 + 1;
    if (th_id > 24)return;
    if (th_id == 0) {
        myidx[b_id] = _index<memb_sz?_index:m_nonafm_filtered[_index-memb_sz];
        mycp[b_id] = tex1Dfetch_real4(pos_tex, myidx[b_id]);
        myradsq[b_id]= _index < memb_sz ? MEMB_RAD*MEMB_RAD: NON_MEMB_RAD*NON_MEMB_RAD ;
    }
    __syncthreads();
    const int index = myidx[b_id];
    const real4 mypos = mycp[b_id];
    const int4 lat = {
        int(mypos.x*inv_dx),
        int(mypos.y*inv_dy),
        int(mypos.z*inv_dz),
        0
    };
    const real radsq = myradsq[b_id];
    const int zmin = _m_max(lat.z - irz_m, 0);
    const int zmax = _m_min(lat.z + irz_m, NZ - 1);
    const int i = lat.x - irx_m + th_id%width;
    const int j = lat.y - iry_m + th_id / width;
    //for (int i = lat.x - irx_nm; i <= lat.x + irx_nm; i++) {
    const int rlx = (i + NX) % NX;
    //for (int j = lat.y - iry_nm; j <= lat.y + iry_nm; j++) {
    const int rly = (j + NY) % NY;
    for (int k = zmin; k <= zmax; k++) {
        const int rlz = k;
        const real dist_sq = _m_norm_sq(i*dx - mypos.x, j*dy - mypos.y, k*dz - mypos.z);
        if (dist_sq < radsq) {
            cmap1.at(rlx, rly, rlz) = index;
        }
    }

}
#define AFM_THREAD_NUM (128)
#define AFM_MULTI (32)
__global__ void map_gen_afm(const cudaTextureObject_t pos_tex,const CellIndex* afm_filtered, CubicDynArrAccessor<int> cmap1, CubicDynArrAccessor<CMask_t> cmap2, size_t sz) {
    const int th_id = threadIdx.x%AFM_MULTI;
    const int b_id = threadIdx.x / AFM_MULTI;
    const int _index = threadIdx.x/AFM_MULTI + blockIdx.x*blockDim.x/AFM_MULTI;
    
    __shared__ real4 mycp[AFM_THREAD_NUM / AFM_MULTI];
    __shared__ int myidx[AFM_THREAD_NUM / AFM_MULTI];
    if (_index >= sz)return;
    static_assert(irx_nm == 2 && iry_nm == 2, "ir*_num must be 2");
    static constexpr int width = irx_nm * 2 + 1;
    if (th_id > 24)return;
    if (th_id == 0) {
        myidx[b_id]= afm_filtered[_index];
        mycp[b_id] = tex1Dfetch_real4(pos_tex, myidx[b_id]);
    }
    __syncthreads();
    const int index = myidx[b_id];
    const real4 mypos = mycp[b_id];
    const int4 lat = {
        int(mypos.x*inv_dx),
        int(mypos.y*inv_dy),
        int(mypos.z*inv_dz),
        0
    };
    const int zmin = _m_max(lat.z - irz_nm, 0);
    const int zmax = _m_min(lat.z + irz_nm, NZ - 1);
    const int i = lat.x - irx_nm + th_id%width;
    const int j = lat.y - iry_nm + th_id/width;
    //for (int i = lat.x - irx_nm; i <= lat.x + irx_nm; i++) {
        const int rlx = (i + NX) % NX;
        //for (int j = lat.y - iry_nm; j <= lat.y + iry_nm; j++) {
            const int rly = (j + NY) % NY;
            for (int k = zmin; k <= zmax; k++) {
                const int rlz = k;
                const real dist_sq = _m_norm_sq(i*dx - mypos.x, j*dy - mypos.y, k*dz - mypos.z);
                if (dist_sq < FAC_MAP*FAC_MAP*NON_MEMB_RAD*NON_MEMB_RAD) {
                    cmap2.at(rlx, rly, rlz) = 1;
                }
                if (dist_sq < NON_MEMB_RAD*NON_MEMB_RAD) {
                    cmap1.at(rlx, rly, rlz) = index;
                }
            }
        
    
}

void map_gen(CellManager&cm, CubicDynArrAccessor<int> cmap1, CubicDynArrAccessor<CMask_t> cmap2) {
    int flt_num;
    const CellIndex* afms=cm.nm_filter.filter_by_state<ALIVE, FIX, MUSUME>(&flt_num);
    map_gen_afm<<<AFM_MULTI*flt_num/AFM_THREAD_NUM+1,AFM_THREAD_NUM>>>(cm.get_pos_tex(), afms, cmap1, cmap2, flt_num);

    afms = cm.nm_filter.filter_by_state<DEAD,DER,AIR>(&flt_num);
    const size_t msz = cm.memb_size();
    map_gen_memb_non_afm << <NMAFM_MULTI*(msz+flt_num) / NMAFM_THREAD_NUM + 1, NMAFM_THREAD_NUM >> >(cm.get_pos_tex(), afms, cmap1, cmap2,msz,msz+ flt_num);
}
