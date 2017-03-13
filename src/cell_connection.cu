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
static constexpr int	ANX = (int)((real)LX / AREA_GRID_ORIGINAL);//ok
                                                                                  /** Y���̃O���b�h��*/
static constexpr int	ANY = (int)((real)LY / AREA_GRID_ORIGINAL);//ok
                                                                                  /** Z���̃O���b�h��*/
static constexpr int	ANZ = (int)((real)LZ / AREA_GRID_ORIGINAL);//ok

static constexpr real 	rtANX = (real)ANX/LX;
static constexpr real 	rtANY = (real)ANY/LY;
static constexpr real 	rtANZ = (real)ANZ/LZ;

static constexpr int	N3 = 200; //max grid cell num //ok
                                  /** 1�זE�̐ڑ��ő吔 */
//static constexpr int	N2 = 400; //max conn num //ok

__global__ void grid_init(cudaTextureObject_t pos_tex,const CELL_STATE*cst,CubicDynArrAccessor<LFStack<int, N3>> darr,CellIterateRange_device cir) {
    //need memset
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cir.nums[CS_asz]) {
        if (cst[index] == UNUSED)return;
        CellPos cp = tex1Dfetch_real4(pos_tex, index);
        //printf("%f %f %f\n", cp.x, cp.y, cp.z);
        
        int aix=  (int)(cp.x*rtANX);
        int aiy = (int)(cp.y*rtANY);
        int aiz = (int)(cp.z*rtANZ);
        
        DBG_ONLY_B(
        if (aix >= ANX || aiy >= ANY || aiz >= ANZ || aix < 0 || aiy < 0 || aiz < 0) {
            printf("wtf %d %d %d, %f %f %f\n", aix, aiy, aiz,cp.x,cp.y,cp.z);
            assert(false);
        });
        
        darr.at(aix, aiy, aiz).push_back_d(index);
        DBG_ONLY_B(
        if(!darr.at(aix, aiy, aiz).isvalid()){
        	printf("too big grid elem size:%d at %d %d %d\n",darr.at(aix,aiy,aiz).size(),aix,aiy,aiz);
        	        	assert(false);
        });
    }
}



#define grid_init_THREAD_NUM (64)
#define connect_proc_THREAD_NUM (128)

#define TH_MULTI (32)
__global__ void connect_proc(cudaTextureObject_t pos_tex,NonMembConn* all_nm_conn, CubicDynArrAccessor<LFStack<int, N3>> darr, int nmemb_start,CellIterateRange_device cir) {
    //need memset
    const int th_id = threadIdx.x%TH_MULTI;
    const int index = nmemb_start+blockIdx.x * (blockDim.x/TH_MULTI) + threadIdx.x/TH_MULTI;
    __shared__ real4 mycp[connect_proc_THREAD_NUM/TH_MULTI];
    static constexpr int srange = 2;
    static constexpr int width = srange*2+1;
    if (index <  cir.nums[CS_asz]) {
    	if(th_id==0)mycp[threadIdx.x / TH_MULTI] = tex1Dfetch_real4(pos_tex, index);
    	__syncthreads(); //no divergence

        if (th_id > 25)return;
        
        const CellPos& cp = mycp[threadIdx.x / TH_MULTI];
        const int anx=  (int)(cp.x*rtANX);
        const int any = (int)(cp.y*rtANY);
        const int anz = (int)(cp.z*rtANZ);
        DBG_ONLY_B(
        if (anx >= (int)ANX || any >= (int)ANY || anz >= (int)ANZ || anx < 0 || any < 0 || anz < 0) {
            dbgprintf("wtf2 %d %d %d, %f %f %f\n", anx, any, anz, cp.x, cp.y, cp.z);
            assert(false);
        });
        const int j = anx - srange + th_id%width;
        const int k = any - srange + th_id / width;
        const int zstart = _m_max(anz - srange, 0); //zstart = zstart > 0 ? zstart : 0;
        const int zend = _m_min(anz + srange, ANZ - 1); //zend = zend < ANZ - 1 ? zend : ANZ - 1;
            const int cj = j<0?j+ANX:j>=ANX?j-ANX:j;
                const int ck = k<0?k+ANY:k>=ANY?k-ANY:k;
                DBG_ONLY_B(
                if(cj<0||cj>=ANX||ck<0||ck>=ANY||zstart<0||zstart>=ANZ||zend<0||zend>=ANZ){
                            	printf("broken coord in conn: %d %d (%d,%d)\n",j,k,zstart,zend);
                            	//printf("origc:%d %d %d\n",anx,any,anz);
                            	//printf("cr:%d %d %d\n",ANX,ANY,ANZ);
                            	assert(false);
                            });
                for (int cl = zstart; cl <= zend; cl++) {
                    const LFStack<int, N3>& stref = darr.at(cj, ck, cl);
                    const size_t g_sz = stref.size();
                    DBG_ONLY_B(
                    if(g_sz>N3){
                    	printf("broken conn size:%d at %d %d %d\n",(int)g_sz,cj,ck,cl);
                    });
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

__global__ void connect_proc_opt(cudaTextureObject_t pos_tex,NonMembConn* all_nm_conn, CubicDynArrAccessor<LFStack<int, N3>> darr, int nmemb_start,CellIterateRange_device cir) {
    //need memset
    //const int th_id = threadIdx.x;
    const int index = nmemb_start+blockIdx.x;
    __shared__ real4 mycp[1];
    __shared__ int3 an[1];
    __shared__ ushort rcidx[1024];
    __shared__ int rccounter[1];
    __shared__ int cnum[1];
    if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0){
    cnum[0]=cir.nums[CS_asz];
    }
    __syncthreads();
    static constexpr int srange = 2;
    static constexpr int width = srange*2+1;
    if (index <  cnum[0]) {
    	if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0){
    		mycp[0] = tex1Dfetch_real4(pos_tex, index);
    		an[0].x=  (int)(mycp[0].x*rtANX);
    		an[0].y = (int)(mycp[0].y*rtANY);
    		an[0].z = (int)(mycp[0].z*rtANZ);
    		rccounter[0]=0;
    	}
    	__syncthreads(); //no divergence



        const CellPos& cp = mycp[0];
        const int anx=an[0].x;
        const int any=an[0].y;
        const int anz=an[0].z;
        DBG_ONLY_B(
        if (anx >= (int)ANX || any >= (int)ANY || anz >= (int)ANZ || anx < 0 || any < 0 || anz < 0) {
            dbgprintf("wtf2 %d %d %d, %f %f %f\n", anx, any, anz, cp.x, cp.y, cp.z);
            assert(false);
        });
        const int j = anx - srange + threadIdx.x;
        const int k = any - srange + threadIdx.y;
        const int zstart = _m_max(anz - srange, 0); //zstart = zstart > 0 ? zstart : 0;
        const int zend = _m_min(anz + srange, ANZ - 1); //zend = zend < ANZ - 1 ? zend : ANZ - 1;
            const int cj = j<0?j+ANX:j>=ANX?j-ANX:j;
                const int ck = k<0?k+ANY:k>=ANY?k-ANY:k;
                DBG_ONLY_B(
                if(cj<0||cj>=ANX||ck<0||ck>=ANY||zstart<0||zstart>=ANZ||zend<0||zend>=ANZ){
                            	printf("broken coord in conn: %d %d (%d,%d)\n",j,k,zstart,zend);
                            	//printf("origc:%d %d %d\n",anx,any,anz);
                            	//printf("cr:%d %d %d\n",ANX,ANY,ANZ);
                            	assert(false);
                            });
                const int cl=zstart+threadIdx.z;
                if(cl<=zend){

                //for (int cl = zstart; cl <= zend; cl++) {
                    const LFStack<int, N3>& stref = darr.at(cj, ck, cl);
                    const size_t g_sz = stref.size();
                    const int rc_head=atomicAdd(&rccounter[0],(int)g_sz);
                    //if(rc_head+g_sz>200)printf("rch:%d,%d %d %d %d\n",rc_head,(int)g_sz,cj,ck,cl);
                    for(int i=0;i<g_sz;i++){
                    	rcidx[i+rc_head]=stref[i];
                    }

                }
                __syncthreads();
                const int th_id_l=threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);
                for(int i=th_id_l;i<rccounter[0];i+=blockDim.x*blockDim.y*blockDim.z){
                	const int oi = rcidx[i];
                	  if (index <= oi)continue;
                	  const CellPos ocp = tex1Dfetch_real4(pos_tex, oi);
                	 const real rad_sum = oi >= nmemb_start ? NON_MEMB_RAD + NON_MEMB_RAD : MEMB_RAD + NON_MEMB_RAD;
                	                        if (p_dist_sq(cp, ocp) <= LJ_THRESH*LJ_THRESH*rad_sum*rad_sum) {
                	                            all_nm_conn[index].conn.push_back_d(oi);
                	                            all_nm_conn[oi].conn.push_back_d(index);

                	                        }
                }

                //}


    }
}
#define CLC_TH (32)
__global__ void connect_proc_dyn_lc(const LFStack<int, N3>* connf,const int myidx,cudaTextureObject_t pos_tex,NonMembConn* all_nm_conn,int nmemb_start,const real4 cp){
	const size_t g_sz = connf->size();
	for(int m=threadIdx.x;m<g_sz;m+=CLC_TH){
		const int oi=connf->operator[](m);
		if(myidx<=oi)continue;
		const CellPos ocp = tex1Dfetch_real4(pos_tex, oi);
		const real rad_sum = oi >= nmemb_start ? NON_MEMB_RAD + NON_MEMB_RAD : MEMB_RAD + NON_MEMB_RAD;
		if (p_dist_sq(cp, ocp) <= LJ_THRESH*LJ_THRESH*rad_sum*rad_sum) {
		                            all_nm_conn[myidx].conn.push_back_d(oi);
		                            all_nm_conn[oi].conn.push_back_d(myidx);

		                        }
	}
}
__global__ void connect_proc_dyn(cudaTextureObject_t pos_tex,NonMembConn* all_nm_conn, CubicDynArrAccessor<LFStack<int, N3>> darr, int nmemb_start,size_t sz,CellIterateRange_device cir) {
    //need memset
    const int th_id = threadIdx.x;
    const int index = nmemb_start+blockIdx.x * blockDim.x + threadIdx.x;
    static constexpr int srange = 2;
    static constexpr int width = srange*2+1;
    if (index <  cir.nums[CS_asz]) {
    	const real4 cp=tex1Dfetch_real4(pos_tex, index);
    	const int anx=  (int)(cp.x*rtANX);
    	        const int any = (int)(cp.y*rtANY);
    	        const int anz = (int)(cp.z*rtANZ);
    	        const int j = anx - srange + th_id%width;
    	                const int k = any - srange + th_id / width;
    	                const int zstart = _m_max(anz - srange, 0); //zstart = zstart > 0 ? zstart : 0;
    	                const int zend = _m_min(anz + srange, ANZ - 1); //zend = zend < ANZ - 1 ? zend : ANZ - 1;
    	                    const int cj = j<0?j+ANX:j>=ANX?j-ANX:j;
    	                        const int ck = k<0?k+ANY:k>=ANY?k-ANY:k;

    	                        for (int cl = zstart; cl <= zend; cl++) {
    	                        	connect_proc_dyn_lc<<<1,CLC_TH>>>(&darr.at(cj, ck, cl),index,pos_tex,all_nm_conn,nmemb_start,cp);
    	                        }

    }
}
__global__ void conn_valid_check(NonMembConn* all_nm_conn, CellIterateRange_device cir){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	    if (index < cir.nums[CS_asz]) {
	    if(all_nm_conn[index].conn.size()>CELL_CONN_NUM){
	    	printf("too big conn num:%d at %d\n",all_nm_conn[index].conn.size(),index);
	    	assert(false);
	    }
	    }
}

__global__ void fix_periodic_pos(CellPos*cpos, CellIterateRange_device cir) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < cir.nums[CS_asz]) {
        /*
            define valid range:
                -LX<=x<2.0*LX
                -LY<=y<2.0*LY

            if out of range toward negative,this returns negative num
        */
    	if(cpos[index].x<-LX||cpos[index].x>=real(2.0)*LX||cpos[index].y<-LY||cpos[index].y>=real(2.0)*LY){
    		printf("broken periodic pos %f %f\n",cpos[index].x,cpos[index].y);
    	}
        cpos[index].x = fmod(cpos[index].x+LX, LX);
        cpos[index].y = fmod(cpos[index].y+LY, LY);
    }
}
#define find_dermis_THREAD_NUM (64)
#define FD_MULTI (32)
__global__ void find_dermis(cudaTextureObject_t pos_tex, CellAttr* cattr, const NonMembConn* all_nm_conn,const CellIterateRange_device cir,int mbsz) {

    const int index = threadIdx.x / FD_MULTI + blockIdx.x*blockDim.x / FD_MULTI;
    const int th_id = threadIdx.x%FD_MULTI;
    const int b_id = threadIdx.x / FD_MULTI;
    __shared__ real4 mypos_sh[find_dermis_THREAD_NUM/ FD_MULTI];
    __shared__ size_t connsz_sh[find_dermis_THREAD_NUM/ FD_MULTI];
    __shared__ int min_idx[find_dermis_THREAD_NUM];
    __shared__ real min_sq[find_dermis_THREAD_NUM];
    if (index >= cir.size<CI_MUSUME, CI_FIX>())return;
    const int raw_idx = cir.idx<CI_MUSUME, CI_FIX>(index);
    if(threadIdx.x==0){
    	for(int i=0;i<find_dermis_THREAD_NUM;i++){
    	    		min_sq[i]=REAL_MAX;
    	    		min_idx[i]=-1;
    	    	}
    }
    if(th_id==0){

    	connsz_sh[b_id] = all_nm_conn[raw_idx].conn.size();
    	mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, raw_idx);

    }
    __syncthreads();
        const NonMembConn& nmc = all_nm_conn[raw_idx];
        const size_t nmc_sz = connsz_sh[b_id];
        //if(nmc_sz<=th_id)return; //no need
        const CellPos mypos = mypos_sh[b_id];
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
    
    nmconn_reset << <(sz*2) / 64 + 1, 64 >> > (cman.get_device_all_nm_conn(),cir );

    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));

    grid_init<<<((unsigned int)sz*2)/ grid_init_THREAD_NUM +1, grid_init_THREAD_NUM >>>(cman.get_pos_tex(),cman.get_device_cstate(), area.acc, cir);
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
    //connect_proc << <TH_MULTI*unsigned(sz*2- nm_start) / connect_proc_THREAD_NUM + 1, connect_proc_THREAD_NUM >> >(cman.get_pos_tex(),cman.get_device_all_nm_conn(), area.acc, int(nm_start), cir);
    connect_proc_opt << <unsigned(sz*2- nm_start), dim3(5,5,5) >> >(cman.get_pos_tex(),cman.get_device_all_nm_conn(), area.acc, int(nm_start), cir);

    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
    DBG_ONLY_B(
    		conn_valid_check<<<sz * 2 / 1024 + 1, 1024 >>>(cman.get_device_all_nm_conn(),cir);
    );
    find_dermis << <FD_MULTI*sz*2 / find_dermis_THREAD_NUM + 1, find_dermis_THREAD_NUM >> >
        (cman.get_pos_tex(), cman.get_device_attr(), cman.get_device_all_nm_conn(), cir,cman.memb_size());
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));

}
