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

/** �O���b�h�T�C�Y*/
static constexpr real AREA_GRID_ORIGINAL = 2.0;//ok

                                                 /** �O���b�h�T�C�Y (?)*/
static constexpr real AREA_GRID = AREA_GRID_ORIGINAL + real(1e-7);//ok
static constexpr real AREA_GRID_INV = real(1.0)/AREA_GRID;//ok

                                                              /** X���̃O���b�h��*/
static constexpr int	ANX = (int)((real)LX / AREA_GRID_ORIGINAL + 0.5);//ok
                                                                                  /** Y���̃O���b�h��*/
static constexpr int	ANY = (int)((real)LY / AREA_GRID_ORIGINAL + 0.5);//ok
                                                                                  /** Z���̃O���b�h��*/
static constexpr int	ANZ = (int)((real)LZ / AREA_GRID_ORIGINAL);//ok
                                                                            /** �O���b�h1������̍זE�i�[����� */
static constexpr int	N3 = 200; //max grid cell num //ok
                                  /** 1�זE�̐ڑ��ő吔 */
//static constexpr int	N2 = 400; //max conn num //ok

__global__ void grid_init(cudaTextureObject_t pos_tex,const CELL_STATE*cst,CubicDynArrAccessor<LFStack<int, N3>> darr,size_t sz,CellIterateRange_device cir) {
    //need memset
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cir.nums[CS_asz]) {
        if (cst[index] == UNUSED)return;
        CellPos cp = tex1Dfetch_real4(pos_tex, index);
        //printf("%f %f %f\n", cp.x, cp.y, cp.z);
        
        int aix=  (int)((real(0.5)*LX - p_diff_x(real(0.5)*LX, cp.x)) * AREA_GRID_INV);
        int aiy = (int)((real(0.5)*LY - p_diff_y(real(0.5)*LY, cp.y)) * AREA_GRID_INV);
        int aiz = (int)((min0(cp.z)) * AREA_GRID_INV);
        
        if (aix >= ANX || aiy >= ANY || aiz >= ANZ || aix < 0 || aiy < 0 || aiz < 0) {
            printf("wtf %d %d %d, %f %f %f\n", aix, aiy, aiz,cp.x,cp.y,cp.z);
            assert(false);
        }
        
        darr.at(aix, aiy, aiz).push_back_d(index);
        
    }
}



#define grid_init_THREAD_NUM (64)
#define connect_proc_THREAD_NUM (128)

#define TH_MULTI (32)
__global__ void connect_proc(cudaTextureObject_t pos_tex,NonMembConn* all_nm_conn, CubicDynArrAccessor<LFStack<int, N3>> darr, int nmemb_start,size_t sz,CellIterateRange_device cir) {
    //need memset
    const int th_id = threadIdx.x%TH_MULTI;
    const int index = nmemb_start+blockIdx.x * (blockDim.x/TH_MULTI) + threadIdx.x/TH_MULTI;
    __shared__ real4 mycp[connect_proc_THREAD_NUM/TH_MULTI];
    static constexpr int srange = 2;
    static constexpr int width = srange*2+1;
    if (index <  cir.nums[CS_asz]) {
        
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

        if (anx >= (int)ANX || any >= (int)ANY || anz >= (int)ANZ || anx < 0 || any < 0 || anz < 0) {
            dbgprintf("wtf2 %d %d %d, %f %f %f\n", anx, any, anz, cp.x, cp.y, cp.z);
            assert(false);
        }
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

__global__ void fix_periodic_pos(CellPos*cpos, CellIterateRange_device cir) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cir.nums[CS_asz]) {
        /*
            define valid range:
                -LX<=x<LX
                -LY<=y<LY

            if out of range toward negative,this returns negative num
        */
        cpos[index].x = fmod(cpos[index].x+LX, LX);
        cpos[index].y = fmod(cpos[index].y+LY, LY);
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
#define find_dermis_THREAD_NUM (64)
#define FD_MULTI (32)
__global__ void find_dermis(cudaTextureObject_t pos_tex, CellAttr* cattr, const NonMembConn* all_nm_conn,const CellIterateRange_device cir,int mbsz) {

    const int index = threadIdx.x / FD_MULTI + blockIdx.x*blockDim.x / FD_MULTI;
    const int th_id = threadIdx.x%FD_MULTI;
    const int b_id = threadIdx.x / FD_MULTI;
    __shared__ real4 mypos_sh[find_dermis_THREAD_NUM/ FD_MULTI];
    //__shared__ int rci_sh[find_dermis_THREAD_NUM/ FD_MULTI];
    __shared__ size_t connsz_sh[find_dermis_THREAD_NUM/ FD_MULTI];
    __shared__ int min_idx[find_dermis_THREAD_NUM];
    __shared__ real min_sq[find_dermis_THREAD_NUM];
    //__shared__ real4 out_dr[MUSUME_TH];
	//const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index >= cir.size<CI_MUSUME, CI_FIX>())return;
    const int raw_idx = cir.idx<CI_MUSUME, CI_FIX>(index);
    if(th_id==0){
    	//rci_sh[b_id]=fix_musume_filtered[index];
    	connsz_sh[b_id] = all_nm_conn[raw_idx].conn.size();
    	mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, raw_idx);
    	for(int i=0;i<find_dermis_THREAD_NUM;i++){
    		min_sq[i]=REAL_MAX;
    		min_idx[i]=-1;
    	}
    }
    __syncthreads();
    //const int raw_idx = rci_sh[b_id];
        const NonMembConn& nmc = all_nm_conn[raw_idx];
        const size_t nmc_sz = connsz_sh[b_id];
        if(nmc_sz<=th_id)return;
        const CellPos mypos = mypos_sh[b_id];
        //const int raw_index = index - nmemb_start;
        //const int raw_idx = fix_musume_filtered[index];
            //real d1sq = REAL_MAX;
            //real distsq = real{ 0.0 };
            //int dermis_idx = -1;
            //const NonMembConn& nmc = all_nm_conn[raw_idx];
            //const int nmc_sz = nmc.conn.size();
            for (int i = th_id; i < nmc_sz; i+=FD_MULTI) {
                const int raw_conn_idx = nmc.conn[i];
                if (raw_conn_idx  < mbsz) {//memb
                    const real distsq = p_dist_sq(tex1Dfetch_real4(pos_tex, raw_conn_idx), mypos);
                    if (distsq < min_sq[threadIdx.x]) {
                    	min_sq[threadIdx.x] = distsq;
                    	min_idx[threadIdx.x]= raw_conn_idx;
                    }
                }

            }
            __syncthreads();

            	for(int r=(FD_MULTI >>1);r>=1;r>>=1){
            		__syncthreads();
            		if(th_id<r){
            			real& me_sq=min_sq[th_id+b_id*FD_MULTI];
            			const real& op_sq=min_sq[th_id+r+b_id*FD_MULTI];
            			if(me_sq>op_sq){
            				me_sq=op_sq;
            				min_idx[th_id+b_id*FD_MULTI]=min_idx[th_id+r+b_id*FD_MULTI];
            			}
            		}
            	}

            if(th_id==0){
            cattr[raw_idx].dermis = min_idx[0+b_id*FD_MULTI];
            }

}

__global__ void area_reset(CubicDynArrAccessor<LFStack<int, N3>> darr) {
    if (threadIdx.x >= ANX)return;
    darr.at(threadIdx.x, blockIdx.x, blockIdx.y).head = 0;
}
__global__ void nmconn_reset(NonMembConn* nmc,CellIterateRange_device cir) {
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx >= cir.nums[CS_asz])return;
    nmc[idx].conn.head = 0;
}

void connect_cell(CellManager & cman)
{
    
    static CubicDynArrGenerator<LFStack<int, N3>> area(ANX, ANY, ANZ);
    size_t sz = cman.all_size();
    size_t nm_start = cman.memb_size();
    CellIterateRange_device cir = cman.get_cell_iterate_range_d();
    
    fix_periodic_pos << <sz * 2 / 1024 + 1, 1024 >> > (cman.get_device_pos_all(), cir);
    area_reset << <dim3(ANY,ANZ),ANX>> > (area.acc);
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
   // area.memset_zero();
    
    nmconn_reset << <(sz*2) / 64 + 1, 64 >> > (cman.get_device_all_nm_conn(),cir );
   // cman.clear_all_non_memb_conn_both();

    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));

    grid_init<<<((unsigned int)sz*2)/ grid_init_THREAD_NUM +1, grid_init_THREAD_NUM >>>(cman.get_pos_tex(),cman.get_device_cstate(), area.acc, sz,cir);
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
    connect_proc << <TH_MULTI*unsigned(sz*2- nm_start) / connect_proc_THREAD_NUM + 1, connect_proc_THREAD_NUM >> >(cman.get_pos_tex(),cman.get_device_all_nm_conn(), area.acc, int(nm_start), sz,cir);
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
   // connect_proc2 << <(sz - nm_start) / connect_proc_THREAD_NUM + 1, connect_proc_THREAD_NUM >> >(cman.get_pos_tex(), cman.get_device_all_nm_conn(), nm_start, sz);
   // CUDA_SAFE_CALL(cudaDeviceSynchronize());

    int flt_num = 0;
    //const CellIndex* ci=cman.nm_filter.filter_by_state<FIX, MUSUME>(&flt_num);
    //CellIterateRange mfcir = cman.get_cell_iterate_range<CI_MUSUME, CI_FIX>();
    
    find_dermis << <FD_MULTI*sz*2 / find_dermis_THREAD_NUM + 1, find_dermis_THREAD_NUM >> >
        (cman.get_pos_tex(), cman.get_device_attr(), cman.get_device_all_nm_conn(), cir,cman.memb_size());
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

}
