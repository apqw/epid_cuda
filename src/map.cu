#include "define.h"
#include "CubicDynArr.h"
#include "CellManager.h"
#include "device_launch_parameters.h"
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


__global__ void map_gen_memb_non_afm(const cudaTextureObject_t pos_tex, const CellIterateRange_device cir, CubicDynArrAccessor<int> cmap1,int memb_sz) {

    const int th_id = threadIdx.x%NMAFM_MULTI;
    const int b_id = threadIdx.x / NMAFM_MULTI;
    const int _index = threadIdx.x / NMAFM_MULTI + blockIdx.x*blockDim.x / NMAFM_MULTI;

    __shared__ real4 mycp[NMAFM_THREAD_NUM / NMAFM_MULTI];
    __shared__ real myradsq[NMAFM_THREAD_NUM / NMAFM_MULTI];
    if (_index >= cir.size<CI_DER, CI_AIR, CI_DEAD, CI_MEMB>())return;
    static_assert(irx_m == 2 && iry_m == 2, "ir*_num must be 2");
    static constexpr int width = irx_m * 2 + 1;
    const int index = cir.idx<CI_DER, CI_AIR, CI_DEAD, CI_MEMB>(_index);
    if (th_id == 0) {
            mycp[b_id] = tex1Dfetch_real4(pos_tex, index);
            myradsq[b_id]= index < memb_sz ? MEMB_RAD*MEMB_RAD: NON_MEMB_RAD*NON_MEMB_RAD ;
        }
        __syncthreads();
    if (th_id > 25)return;
    


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
    const int rlx = (i + NX) % NX;
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
__global__ void map_gen_afm(const cudaTextureObject_t pos_tex, const CellIterateRange_device cir, CubicDynArrAccessor<int> cmap1, CubicDynArrAccessor<CMask_t> cmap2) {
    const int th_id = threadIdx.x%AFM_MULTI;
    const int b_id = threadIdx.x / AFM_MULTI;
    const int _index = threadIdx.x/AFM_MULTI + blockIdx.x*blockDim.x/AFM_MULTI;
    
    __shared__ real4 mycp[AFM_THREAD_NUM / AFM_MULTI];
    if (_index >= cir.size<CI_ALIVE, CI_MUSUME, CI_FIX>())return;
    static_assert(irx_nm == 2 && iry_nm == 2, "ir*_num must be 2");
    static constexpr int width = irx_nm * 2 + 1;
    const int index = cir.idx<CI_ALIVE, CI_MUSUME, CI_FIX>(_index);
    if (th_id == 0) {
            mycp[b_id] = tex1Dfetch_real4(pos_tex,index);
        }
        __syncthreads();//pls do this after dropping threads
    if (th_id > 25)return;
    


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
        const int rlx = i<0?i+NX:i>=NX?i-NX:i;
            const int rly = j<0?j+NY:j>=NY?j-NY:j;
            if(rlx<0||rlx>=NX||rly<0||rly>=NY||zmin<0||zmin>=NZ||zmax<0||zmax>=NZ){
            	printf("broken coord in map: %d %d (%d,%d)\n",rlx,rly,zmin,zmax);
            	assert(false);
            }
            for (int k = zmin; k <= zmax; k++) {
                const int rlz = k;
                const real dist_sq = _m_norm_sq(i*dx - mypos.x, j*dy - mypos.y, k*dz - mypos.z);
                if (dist_sq < FAC_MAP*FAC_MAP*NON_MEMB_RAD*NON_MEMB_RAD) {
                    cmap2.at(rlx, rly, rlz) = -1;
                }
                if (dist_sq < NON_MEMB_RAD*NON_MEMB_RAD) {
                    cmap1.at(rlx, rly, rlz) = index;
                }
            }
        
    
}
__global__ void cmap1_clear(CubicDynArrAccessor<int> cmap1) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cmap1.size())return;
    cmap1.raw_at(index) = -1;
}

__global__ void cmap2_clear(CubicDynArrAccessor<CMask_t> cmap2) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cmap2.size())return;
    cmap2.raw_at(index) = 0;
}
void map_gen(CellManager&cm, CubicDynArrAccessor<int> cmap1, CubicDynArrAccessor<CMask_t> cmap2) {
    cmap1_clear << <cmap1.size() / 1024 + 1, 1024 >> > (cmap1);
    cmap2_clear << <cmap2.size() / 1024 + 1, 1024 >> > (cmap2);
   // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const size_t nmsz = cm.non_memb_size();
    CellIterateRange_device cird = cm.get_cell_iterate_range_d();
    map_gen_afm<<<AFM_MULTI*nmsz/AFM_THREAD_NUM+1,AFM_THREAD_NUM>>>(cm.get_pos_tex(), cird, cmap1, cmap2);
   // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
    const size_t msz = cm.memb_size();
    map_gen_memb_non_afm << <NMAFM_MULTI*cm.all_size() / NMAFM_THREAD_NUM + 1, NMAFM_THREAD_NUM >> >(cm.get_pos_tex(), cird, cmap1,msz);
   // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
}
