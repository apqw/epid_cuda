#include "define.h"
#include "CubicDynArr.h"
#include "LFStack.h"
#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"
#include "math_helper.h"
#include "cell_connection.h"
#include "CellManager.h"
#include <cfloat>
#include <thrust/count.h>

/** グリッドサイズ*/
static constexpr real AREA_GRID_ORIGINAL = 2.0;//ok

                                                 /** グリッドサイズ (?)*/
static constexpr real AREA_GRID = AREA_GRID_ORIGINAL + real(1e-7);//ok
static constexpr real AREA_GRID_INV = real(1.0)/AREA_GRID;//ok

                                                              /** X方向のグリッド数*/
static constexpr int	ANX = (int)((real)LX / AREA_GRID_ORIGINAL + 0.5);//ok
                                                                                  /** Y方向のグリッド数*/
static constexpr int	ANY = (int)((real)LY / AREA_GRID_ORIGINAL + 0.5);//ok
                                                                                  /** Z方向のグリッド数*/
static constexpr int	ANZ = (int)((real)LZ / AREA_GRID_ORIGINAL);//ok
                                                                            /** グリッド1つ当たりの細胞格納数上限 */
static constexpr int	N3 = 200; //max grid cell num //ok
                                  /** 1細胞の接続最大数 */
//static constexpr int	N2 = 400; //max conn num //ok

__global__ void grid_init(cudaTextureObject_t pos_tex,CubicDynArrAccessor<LFStack<int, N3>> darr,size_t sz) {
    //need memset
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sz) {
        CellPos cp = tex1Dfetch_real4(pos_tex, index);
        //printf("%f %f %f\n", cp.x, cp.y, cp.z);
        
        int aix=  (int)((real(0.5)*LX - p_diff_x(real(0.5)*LX, cp.x)) * AREA_GRID_INV);
        int aiy = (int)((real(0.5)*LY - p_diff_y(real(0.5)*LY, cp.y)) * AREA_GRID_INV);
        int aiz = (int)((min0(cp.z)) * AREA_GRID_INV);
        
        assert(!(aix >= ANX || aiy >= ANY || aiz >= ANZ || aix < 0 || aiy < 0 || aiz < 0));
        
        darr.at(aix, aiy, aiz).push_back_d(index);
        
    }
}

#define _m_max(a,b) ((a)>(b)?(a):(b))
#define _m_min(a,b) ((a)<(b)?(a):(b))

#define grid_init_THREAD_NUM (64)
#define connect_proc_THREAD_NUM (128)

#define TH_MULTI (32)
__global__ void connect_proc(cudaTextureObject_t pos_tex,NonMembConn* all_nm_conn, CubicDynArrAccessor<LFStack<int, N3>> darr, int nmemb_start,size_t sz) {
    //need memset
    const int th_id = threadIdx.x%TH_MULTI;
    const int index = nmemb_start+blockIdx.x * (blockDim.x/TH_MULTI) + threadIdx.x/TH_MULTI;
    __shared__ real4 mycp[connect_proc_THREAD_NUM/TH_MULTI];
    static constexpr int srange = 2;
    static constexpr int width = srange*2+1;
    if (index < sz) {
        
        if (th_id > 24)return;
        /*
        if (th_id==0&&index%100==0)printf("PogAndChamp %d\n",index);
        if (th_id == 0 && index==sz-1)printf("End %d\n", index);
        */
        /*
        const CellPos cp = tex1Dfetch_real4(pos_tex, index);
        const int anx = (int)(cp.x * AREA_GRID_INV);
        const int any = (int)(cp.y* AREA_GRID_INV);
        const int anz = (int)(cp.z* AREA_GRID_INV);
        
        assert(!(anx >= (int)ANX || any >= (int)ANY || anz >= (int)ANZ || anx < 0 || any < 0 || anz < 0));
        //const int cw = width / TH_MULTI + th_id;
        const int xstart = anx -srange+th_id%width;
        const int xend = xstart;
        const int ystart = any - srange+th_id/width;
        const int yend = ystart;
        const int zstart = _m_max(anz - srange,0); //zstart = zstart > 0 ? zstart : 0;
        const int zend = _m_min(anz + srange, ANZ - 1); //zend = zend < ANZ - 1 ? zend : ANZ - 1;
        for (int j = xstart; j <= xend; j++) {
            const int cj = (j + ANX) % ANX;
            for (int k = ystart; k <= yend; k++) {
                const int ck = (k + ANY) % ANY;
                for (int cl = zstart; cl <= zend; cl++) {
                    //const int cl = l;
                    const LFStack<int, N3>& stref = darr.at(cj, ck, cl);
                    const size_t g_sz = stref.size();
                    for (int m = 0; m < g_sz; m++) {
                        const int oi = stref[m];
                        if (index <= oi)continue;
                        const CellPos ocp = tex1Dfetch_real4(pos_tex, oi);
                        const real rad_sum = oi >= nmemb_start ? NON_MEMB_RAD + NON_MEMB_RAD : MEMB_RAD + NON_MEMB_RAD;
                        if (p_dist_sq(cp, ocp) <= LJ_THRESH*LJ_THRESH*rad_sum*rad_sum) {
                            all_nm_conn[index].conn.push_back_d(oi);
                            all_nm_conn[oi].conn.push_back_d(index);
                        }
                    }
                }
            }
        }
        */
        if(th_id==0)mycp[threadIdx.x / TH_MULTI] = tex1Dfetch_real4(pos_tex, index);
        __syncthreads();
        const CellPos& cp = mycp[threadIdx.x / TH_MULTI];
        const int anx = (int)(cp.x * AREA_GRID_INV);
        const int any = (int)(cp.y* AREA_GRID_INV);
        const int anz = (int)(cp.z* AREA_GRID_INV);

        assert(!(anx >= (int)ANX || any >= (int)ANY || anz >= (int)ANZ || anx < 0 || any < 0 || anz < 0));
        //const int cw = width / TH_MULTI + th_id;
        const int j = anx - srange + th_id%width;
        const int k = any - srange + th_id / width;
        const int zstart = _m_max(anz - srange, 0); //zstart = zstart > 0 ? zstart : 0;
        const int zend = _m_min(anz + srange, ANZ - 1); //zend = zend < ANZ - 1 ? zend : ANZ - 1;
            const int cj = (j + ANX) % ANX;
                const int ck = (k + ANY) % ANY;
                for (int cl = zstart; cl <= zend; cl++) {
                    //const int cl = l;
                    const LFStack<int, N3>& stref = darr.at(cj, ck, cl);
                    const size_t g_sz = stref.size();
                    for (int m = 0; m < g_sz; m++) {
                        const int oi = stref[m];
                        if (index <= oi)continue;
                        const CellPos ocp = tex1Dfetch_real4(pos_tex, oi);
                        const real rad_sum = oi >= nmemb_start ? NON_MEMB_RAD + NON_MEMB_RAD : MEMB_RAD + NON_MEMB_RAD;
                        if (p_dist_sq(cp, ocp) <= LJ_THRESH*LJ_THRESH*rad_sum*rad_sum) {
                            all_nm_conn[index].conn.push_back_d(oi);
                            all_nm_conn[oi].conn.push_back_d(index);
                        }
                    }
                }
            
        
    }
}
/*
__global__ void connect_proc2(cudaTextureObject_t pos_tex, NonMembConn* all_nm_conn, int nmemb_start, size_t sz) {
    const int index = nmemb_start + blockIdx.x * blockDim.x + threadIdx.x;
    static constexpr int srange = 2;
    if (index < sz) {
        const NonMembConn& nmc = all_nm_conn[index];
        const size_t csz = nmc.conn.size();
        for (int i = 0; i < csz; i++) {
            const int oi = nmc.conn[i];
            if (oi < nmemb_start) {
                all_nm_conn[oi].conn.push_back_d(index);
            }
        }
    }
}
*/
__global__ void find_dermis(cudaTextureObject_t pos_tex, CellAttr* nm_cattr, const NonMembConn* all_nm_conn,const CellIndex* fix_musume_filtered,size_t nm_start,size_t sz) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sz) {
        //const int raw_index = index - nmemb_start;
        const int raw_idx = fix_musume_filtered[index];
            real d1sq = REAL_MAX;
            //real distsq = real{ 0.0 };
            int dermis_idx = -1;
            const NonMembConn& nmc = all_nm_conn[raw_idx];
            const int nmc_sz = nmc.conn.size();
            for (int i = 0; i < nmc_sz; i++) {
                const int raw_conn_idx = nmc.conn[i];
                if (raw_conn_idx  < nm_start) {//memb
                    const real distsq = p_dist_sq(tex1Dfetch_real4(pos_tex, raw_conn_idx), tex1Dfetch_real4(pos_tex, raw_idx));
                    if (distsq < d1sq) {
                        d1sq = distsq;
                        dermis_idx = raw_conn_idx;
                    }
                }

            }
            nm_cattr[raw_idx-nm_start].dermis = dermis_idx;
    }
}




void connect_cell(CellManager & cman)
{
    static CubicDynArrGenerator<LFStack<int, N3>> area(ANX, ANY, ANZ);
    area.memset_zero();
    cman.clear_all_non_memb_conn_both();

    size_t sz = cman.all_size();
    size_t nm_start = cman.memb_size();

    grid_init<<<((unsigned int)sz)/ grid_init_THREAD_NUM +1, grid_init_THREAD_NUM >>>(cman.get_pos_tex(), area.acc, sz);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    connect_proc << <TH_MULTI*unsigned(sz- nm_start) / connect_proc_THREAD_NUM + 1, connect_proc_THREAD_NUM >> >(cman.get_pos_tex(),cman.get_device_all_nm_conn(), area.acc, int(nm_start), sz);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
   // connect_proc2 << <(sz - nm_start) / connect_proc_THREAD_NUM + 1, connect_proc_THREAD_NUM >> >(cman.get_pos_tex(), cman.get_device_all_nm_conn(), nm_start, sz);
   // CUDA_SAFE_CALL(cudaDeviceSynchronize());

    int flt_num = 0;
    const CellIndex* ci=cman.nm_filter.filter_by_state<FIX, MUSUME>(&flt_num);
    find_dermis << <flt_num / 64 + 1, 64 >> > (cman.get_pos_tex(), cman.get_device_nmattr(), cman.get_device_all_nm_conn(), ci, nm_start, flt_num);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

}