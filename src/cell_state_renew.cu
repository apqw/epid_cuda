#include "cell_state_renew.h"
#include "CellManager.h"
#include "define.h"
#include "math_helper.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <cassert>
#include <iostream>
#include "cuda_helper_misc.h"
#include "LFStack.h"
/*
#undef DBG_ONLY
#undef DBG_ONLY_B
#undef dbgprintf
#define DBG_ONLY(s) do{s;}while(0)
#define DBG_ONLY_B(...) do{__VA_ARGS__}while(0)
#define dbgprintf(x,...) printf((x),__VA_ARGS__)
*/
#ifdef USE_DOUBLE_AS_REAL
#define curand_uniform_real curand_uniform_double
#define curand_uniform2_real curand_uniform2_double
#else
#define curand_uniform_real curand_uniform
#endif
__global__ void _curand_init_ker(curandState* stt,size_t sz) {
    const int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=sz)return;
    curand_init(8181,idx, 0, &stt[idx]);
}
class rand_generator {
    curandState* _state;
    int _size;
    void init_state() {
        _curand_init_ker << <_size/512+1, 512 >> >(_state,_size);
    }
public:
    curandState* get_state() {
        return _state;
    }

    rand_generator(int size):_size(size) {
        cudaMalloc((void**)&_state, size * sizeof(curandState));
        init_state();
    }
    ~rand_generator() {
        cudaFree(_state);
        _state = nullptr;
    }
};
__device__ inline real weighted_eps_kb(bool malig) {
    
    return malig ? accel_div *eps_kb : eps_kb;
}

__device__ inline real ageb_const(real ca2p_avg,bool malig) {
    

    return weighted_eps_kb(malig)*(S2 + alpha_b*min0(ca2p_avg - ca2p_init));
}

__device__ inline real weighted_eps_ks(bool malig) {
    
    return malig ? accel_diff *eps_ks : eps_ks;
}

__device__ inline real agek_ALIVE_const(real ca2p_avg, bool malig) {
    //using namespace cont;
    

    return weighted_eps_ks(malig)*(S0 + alpha_k*min0(ca2p_avg - ca2p_init));
}

__device__ inline real agek_DEAD_AIR_const() {

    return eps_kk*S1;
}

__device__ inline real k_lipid_release(real ca2p_avg,real agek) {
   // using namespace cont;

    
    return real(0.25)*lipid_rel*
        (real(1.0) + tanh((ca2p_avg - ubar) *delta_sig_r1_inv))*(real(1.0) + tanh((agek - THRESH_SP) * delta_lipid_inv));
}
__device__ inline real k_lipid(real ca2p_avg, real agek) {

    
    return real(0.25)*lipid*
        (real(1.0) + tanh((ca2p_avg - ubar) *delta_sig_r1_inv))*(real(1.0) + tanh((agek - THRESH_SP) * delta_lipid_inv));
}

__device__ bool stochastic_div_test(curandState* cur,real div_age_thresh,bool malig) {

    if (!STOCHASTIC) {
        return true;
    }
    else {
    	return
#ifdef FIXED_RANDOM_DIVNUM_UNIFORM
    	FIXED_RANDOM_DIVNUM_UNIFORM
#else
    	curand_uniform_real(cur)
#endif
    			*RANDOM_DIVNUM_COEF*(div_age_thresh*stoch_div_time_ratio) <= DT_Cell*weighted_eps_kb(malig)*S2;
}
}

__device__ bool is_divide_ready(curandState* cur, real ageb,real div_age_thresh,bool malig,bool has_pair) {

    if (!has_pair && (ageb >= div_age_thresh*(real(1.0) - stoch_div_time_ratio))) {
        return stochastic_div_test(cur,div_age_thresh,malig);
    }
    else {
        return false;
    }
}

__device__ inline real4 calc_cell_uvec(real4 c1,real4 c2) {

    return normalize3_r_dev_w_inv(vpsub(c1, c2));
    /*
    double nvx = p_diff_x(c1->x(), c2->x());
    double nvy = p_diff_y(c1->y(), c2->y());
    double nvz = c1->z() - c2->z();

    double norm = sqrt(DIST_SQ(nvx, nvy, nvz));
    *outx = nvx / norm;
    *outy = nvy / norm;
    *outz = nvz / norm;
    */

}

__device__ inline real4  div_direction(curandState* cs,real4 me,real4 dermis) {


    const real4 nvec=calc_cell_uvec(me, dermis);
    real4 ov;
    real sum;
    do {
        //const real theta=;
        //double rand_theta = M_PI*genrand_real();
        real cr1, sr1;
        sincos(M_PI*curand_uniform_real(cs), &sr1, &cr1);
        real cr2, sr2;
        sincos(real(2.0)*M_PI*curand_uniform_real(cs), &sr2, &cr2);
        /*
        double cr1 = cos(rand_theta);
        double sr1 = sin(rand_theta);
        double rand_phi = 2 * M_PI*genrand_real();
        double cr2 = cos(rand_phi);
        double sr2 = sin(rand_phi);
        */
        const real4 rvec = { sr1*cr2,sr1*sr2,cr1,real(0.0) };
        /*
        double rvx = sr1*cr2;
        double rvy = sr1*sr2;
        double rvz = cr1;
        */
        ov = cross3(nvec, rvec);
        /*
        ox = ny*rvz - nz*rvy;
        oy = nz*rvx - nx*rvz;
        oz = nx*rvy - ny*rvx;
        */
    } while ((sum = get_norm3sq(ov)) < real(1.0e-14));
    sum = rsqrt(sum);
    return cvmul(sum, ov);
    /*
    *outx = ox / sum;
    *outy = oy / sum;
    *outz = oz / sum;
    */
}

//sort
//remove
//add

struct SwapStruct {
    CellPos* cpos;
    CellAttr* cat;
    CELL_STATE* cst;
    NonMembConn* nmconn;

    int* swp_data;
    int* arr_avail;
    /*
    __device__ void record_pending_swap_by_state(int src, int dst,CELL_STATE stt) {
        if (src == dst) {
            cst[src] = stt;
            return;
        }
        cpos_tmp[src] = cpos[src];
        cat_tmp[src] = cat[src];
        cst_tmp[src] = stt;
        nmconn_tmp[src].conn.copy_fast(&nmconn[src].conn);
        swp_data[src] = dst;

        cpos_tmp[dst] = cpos[dst];
        cat_tmp[dst] = cat[dst];
        cst_tmp[dst] = cst[dst];
        nmconn_tmp[dst].conn.copy_fast(&nmconn[dst].conn);
        swp_data[dst] = src;
    }*/

    __device__ void raw_swap(int s1, int s2) {
        CellPos ctmp = cpos[s1];
        CellAttr cattr = cat[s1];
        CELL_STATE sttt = cst[s1];
        NonMembConn nmc; nmc.conn.copy_fast(&nmconn[s1].conn);
        cpos[s1] = cpos[s2];
        cat[s1] = cat[s2];
        cst[s1] = cst[s2];
        nmconn[s1].conn.copy_fast(&nmconn[s2].conn);
        cpos[s2] = ctmp;
        cat[s2] = cattr;
        cst[s2] = sttt;
        nmconn[s2].conn.copy_fast(&nmc.conn);
    }


    /*
    __device__ void record_pending_swap_remove(int src, int dst_removed) {
        cpos_tmp[src] = cpos[src];
        cat_tmp[src] = cat[src];
        cst_tmp[src] = cst[src];
        nmconn_tmp[src].conn.copy_fast(&nmconn[src].conn);
        swp_data[src] = dst_removed;
    }
    __device__ void write_data_from_tmp(int idx) {
        int dst = swp_data[idx];
        cpos[dst] = cpos_tmp[idx];
        cat[dst] = cat_tmp[idx];
        cst[dst] = cst_tmp[idx];
        nmconn[dst].conn.copy_fast(&nmconn_tmp[idx].conn);
    }
    */
    __device__ void set_from(int dst,int src) {
        //int dst = swp_data[idx];
        cpos[dst] = cpos[src];
        cat[dst] = cat[src];
        cst[dst] = cst[src];
        nmconn[dst].conn.copy_fast(&nmconn[src].conn);
    }


};

__global__ void prepare_state_change_MUSUME_NOPAIR(SwapStruct sstr,CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_MUSUME_NOPAIR>())return;
    const int raw_idx = cir.idx<CI_MUSUME_NOPAIR>(index);
    if (sstr.cat[raw_idx].dermis < 0 && sstr.cat[raw_idx].pair < 0) {
        dbgprintf("MU->ALIVE %d\n", raw_idx);
        atomicAdd(&cir.nums[CS_count_musume_nopair],1);
        sstr.cst[raw_idx]=ALIVE;
        sstr.swp_data[raw_idx] = -2;
        //sstr.record_pending_swap_by_state(raw_idx, swp_idx,ALIVE);
    }
}

__global__ void prepare_state_change_MUSUME_NOPAIR2(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.nums[CS_count_musume_nopair])return;
    const int raw_idx = cir.idx<CI_MUSUME_NOPAIR>(index);

    if (sstr.swp_data[raw_idx] == -1) {
        const int av= atomicAdd(&cir.nums[CS_count_available], 1);
        sstr.arr_avail[av] = raw_idx;//order will be broken but doesnt matter
    }
    else if (sstr.swp_data[raw_idx] == -2) {
        sstr.swp_data[raw_idx] = -1;
    }
}

__global__ void exec_state_change_MUSUME_NOPAIR(SwapStruct sstr, CellIterateRange_device cir) {

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_MUSUME_NOPAIR>())return;
    const int raw_idx = cir.idx<CI_MUSUME_NOPAIR>(index);
    if (sstr.swp_data[raw_idx] == -2) {
        const int st = atomicAdd(&cir.nums[CS_count_store], 1);
        sstr.raw_swap(sstr.arr_avail[st], raw_idx);
    }
}

__global__ void prepare_state_change_ALIVE(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_ALIVE>())return;
    const int raw_idx = cir.idx<CI_ALIVE>(index);
    if (sstr.cat[raw_idx].agek >= THRESH_DEAD) {

        atomicAdd(&cir.nums[CS_count_alive], 1);
        atomicAdd(&cir.nums[CS_count_sw], 1);
        sstr.swp_data[raw_idx] = -2;
        //sstr.record_pending_swap_by_state(raw_idx, swp_idx, DEAD);
        dbgprintf("ALIVE->DEAD %d sw:%d\n", raw_idx,cir.nums[CS_count_sw]);
        sstr.cst[raw_idx]=DEAD;
    }
    else {
        CellAttr& mycat = sstr.cat[raw_idx];
        const real myca2p_avg = mycat.ca2p_avg;
        const real myagek = mycat.agek;
        const real myinfat = mycat.in_fat;
        
        const double tmp = k_lipid_release(myca2p_avg, myagek)*myinfat;
        mycat.in_fat += DT_Cell*(k_lipid(myca2p_avg, myagek)*(real(1.0) - myinfat) - tmp);
        mycat.ex_fat += DT_Cell*tmp;
        mycat.agek += DT_Cell*agek_ALIVE_const(myca2p_avg, mycat.is_malignant); //update last
    }
}

__global__ void prepare_state_change_ALIVE2(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.nums[CS_count_alive])return;
    const int raw_idx = cir.idx<CI_ALIVE>(index);

    if (sstr.swp_data[raw_idx] == -1) {
        const int av = atomicAdd(&cir.nums[CS_count_available], 1);
        sstr.arr_avail[av] = raw_idx; //order will be broken but doesnt matter
    }
    else if (sstr.swp_data[raw_idx] == -2) {
        sstr.swp_data[raw_idx] = -1;
    }
}
__global__ void exec_state_change_ALIVE(SwapStruct sstr, CellIterateRange_device cir) {

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_ALIVE>())return;
    const int raw_idx = cir.idx<CI_ALIVE>(index);
    if (sstr.swp_data[raw_idx] ==-2) {
        const int st = atomicAdd(&cir.nums[CS_count_store], 1);
        sstr.raw_swap(sstr.arr_avail[st], raw_idx);
    }
}


__global__ void prepare_state_change_PAIR(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_PAIR>())return;
    const int raw_idx = cir.idx<CI_PAIR>(index);
    constexpr real unpair_th = unpair_dist_coef*NON_MEMB_RAD_SUM;
    CellAttr& mycat = sstr.cat[raw_idx];
    if (raw_idx < mycat.pair)return;
    real distsq;
    mycat.ageb += DT_Cell*ageb_const(mycat.ca2p_avg, mycat.is_malignant);
    if (mycat.spr_nat_len < real(2.0)*NON_MEMB_RAD) {

        mycat.spr_nat_len += DT_Cell*eps_L;
        sstr.cat[mycat.pair].spr_nat_len = mycat.spr_nat_len;
    }
    else if ((distsq=p_dist_sq(sstr.cpos[raw_idx], sstr.cpos[mycat.pair]))>unpair_th*unpair_th) {
        if (sstr.cst[mycat.pair] == FIX) {
            atomicAdd(&cir.nums[CS_count_pair], 1);
            sstr.swp_data[raw_idx] = -2;
            dbgprintf("PAIR->MU(1) %d\n", raw_idx);
            //sstr.swp_data[mycat.pair] = -2;
            //printf("\n1 unpaired %d %d\n", raw_idx, mycat.pair);
        }
        else {
            atomicAdd(&cir.nums[CS_count_pair], 2);
            sstr.swp_data[raw_idx] = -2;
            sstr.swp_data[mycat.pair] = -2;
            dbgprintf("PAIR->MU(2) %d %d\n", raw_idx,mycat.pair);
            //printf("\n2 unpaired %d %d\n", raw_idx, mycat.pair);
        }

        dbgprintf("unpaired distsq:%f\n",distsq);
        mycat.spr_nat_len = 0.0;
        sstr.cat[mycat.pair].spr_nat_len = 0.0;
        sstr.cat[mycat.pair].pair = -1;
        mycat.pair = -1;
        
        //sstr.record_pending_swap_by_state(raw_idx, swp_idx, DEAD);
    }
}


__global__ void prepare_state_change_PAIR2(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;

    if (index >= cir.nums[CS_count_pair])return;
    const int raw_idx = cir.idx<CI_PAIR>(index);

    if (sstr.swp_data[raw_idx] == -1) {
    	DBG_ONLY(printf("pair non overlap %d, pair:%d\n",raw_idx,sstr.cat[raw_idx].pair));
        const int av = atomicAdd(&cir.nums[CS_count_available], 1);
        sstr.arr_avail[av] = raw_idx; //order will be broken but doesnt matter

        //if(sstr.cat[raw_idx].pair)
    }
    else if (sstr.swp_data[raw_idx] == -2) {
    	DBG_ONLY(printf("pair overlapped %d\n",raw_idx));
        sstr.swp_data[raw_idx] = -1;
    }
}

__global__ void exec_state_change_PAIR(SwapStruct sstr, CellIterateRange_device cir) {

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_PAIR>())return;
    const int raw_idx = cir.idx<CI_PAIR>(index);

    if (sstr.swp_data[raw_idx] == -2) {
        const int st = atomicAdd(&cir.nums[CS_count_store], 1);
        const int dest = sstr.arr_avail[st];
        const int pair_reserve=sstr.cat[dest].pair;
        __syncthreads();//race occurs when pairs are in swap area
        sstr.cat[pair_reserve].pair=raw_idx;
        __syncthreads();
        sstr.raw_swap(sstr.arr_avail[st], raw_idx);
    }
}

__device__ void _swp_init(int*swp_data,CellIterateRange_device cir) {
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cir.nums[CS_count_alive] = 0;
        cir.nums[CS_count_musume] = 0;
        cir.nums[CS_count_dead] = 0;
        cir.nums[CS_count_alive_act] = 0;
        cir.nums[CS_count_store] = 0;
        cir.nums[CS_count_available] = 0;
        cir.nums[CS_count_air] = 0;
        cir.nums[CS_count_removed] = 0;
        cir.nums[CS_count_removed_air] = 0;
        cir.nums[CS_count_removed_dead] = 0;
        cir.nums[CS_count_musume_divided] = 0;

        cir.nums[CS_count_pair] = 0;
        cir.nums[CS_count_musume_nopair] = 0;
    }
    const int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx >= cir.nums[CS_asz])return;
    swp_data[idx] = -1;
}

__global__ void _state_change_count_apply_and_init(int*swp_data,  CellIterateRange_device cir) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
            cir.nums[CS_alive_hd] += cir.nums[CS_count_alive];
        cir.nums[CS_musume_hd] += cir.nums[CS_count_musume_nopair];
        cir.nums[CS_pair_hd] += cir.nums[CS_count_pair];
    }
    _swp_init(swp_data,  cir);
}

//exec order
//MUSUME prep1 prep2,ALIVE prep1 prep2
//available willbe:[MUSUME available...(broken order)],[MUSUME available...(broken order)]
//then MUSUME exec,ALIVE exec (extract available idx ordered)



__global__ void prepare_remove_AIR(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_AIR>())return;
    //printf("something wrong\n");
    const int raw_idx = cir.idx<CI_AIR>(index);
    if (sstr.cat[raw_idx].agek >= ADHE_CONST&&sstr.nmconn[raw_idx].conn.size() <= DISA_conn_num_thresh) {
        atomicAdd(&cir.nums[CS_count_air], 1);
        sstr.swp_data[raw_idx] = -2;
        dbgprintf("AIR->REMOVED %d\n", raw_idx);
    }
    else {
        sstr.cat[raw_idx].agek += DT_Cell*agek_DEAD_AIR_const();
    }
}

__global__ void prepare_remove_AIR2(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int rest = cir.size<CI_AIR>() - index;
    if (rest<=0||rest>cir.nums[CS_count_air])return;
    const int raw_idx = cir.idx<CI_AIR>(index);

    if (sstr.swp_data[raw_idx] == -1) {
        const int av = atomicAdd(&cir.nums[CS_count_available], 1);
        sstr.arr_avail[av] = raw_idx; //order will be broken but doesnt matter
    }
    else if (sstr.swp_data[raw_idx] == -2) {
        sstr.swp_data[raw_idx] = -1;
    }
}

__global__ void prepare_remove_DEAD(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_DEAD>())return;
    const int raw_idx = cir.idx<CI_DEAD>(index);
    if (sstr.cat[raw_idx].agek >= ADHE_CONST&&sstr.nmconn[raw_idx].conn.size() <= DISA_conn_num_thresh) {
        atomicAdd(&cir.nums[CS_count_dead], 1);
        sstr.swp_data[raw_idx] = -2;
        dbgprintf("DEAD->REMOVED %d\n", raw_idx);
        //printf("2 removed %d\n", raw_idx);
    }
    else {
        sstr.cat[raw_idx].agek += DT_Cell*agek_DEAD_AIR_const();
    }
}

__global__ void prepare_remove_DEAD2(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int rest = cir.size<CI_DEAD>() - index;
    if (rest <= 0 || rest>cir.nums[CS_count_dead])return;
    const int raw_idx = cir.idx<CI_DEAD>(index);

    if (sstr.swp_data[raw_idx] == -1) {
        const int av = atomicAdd(&cir.nums[CS_count_available], 1);
        //printf("is this available? %d alivehead:%d\n",raw_idx,cir.nums[CS_alive_hd]);
        sstr.arr_avail[av] = raw_idx; //order will be broken but doesnt matter
    }
    else if (sstr.swp_data[raw_idx] == -2) {
        sstr.swp_data[raw_idx] = -1;
    }
}

__global__ void exec_remove_DEAD(SwapStruct sstr, CellIterateRange_device cir) {

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_DEAD>())return;
    const int raw_idx = cir.idx<CI_DEAD>(index);
    if (sstr.swp_data[raw_idx] == -2) {
        //atomicAdd(&cir.nums[CS_count_removed_dead], 1);
        const int st = atomicAdd(&cir.nums[CS_count_removed], 1);
        //printf("set removed dead %d from %d\n",raw_idx,sstr.arr_avail[st]);
        sstr.set_from(raw_idx,sstr.arr_avail[st]);
    }
}

__global__ void exec_remove_AIR(SwapStruct sstr, CellIterateRange_device cir) {

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_AIR>())return;
    const int raw_idx = cir.idx<CI_AIR>(index);
    if (sstr.swp_data[raw_idx] == -2) {
        atomicAdd(&cir.nums[CS_count_removed_air], 1);
        const int st = atomicAdd(&cir.nums[CS_count_removed], 1);
        sstr.set_from(raw_idx, sstr.arr_avail[st]);
    }
}

__global__ void finalize_remove_DEAD(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int rest = cir.size<CI_DEAD>() - index;
    if (rest <= 0 || rest>cir.nums[CS_count_removed_air])return;

    const int raw_idx = cir.idx<CI_DEAD>(index);
    sstr.set_from(cir.nums[CS_dead_hd]- cir.nums[CS_count_removed_air]+rest-1, raw_idx);
    //fill from head. if enough elements is there,completely filled.if not,whole of them are moved to head
}

__global__ void finalize_remove_ALIVE(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int rest = cir.size<CI_ALIVE>() - index;
    if (rest <= 0 || rest>cir.nums[CS_count_removed])return;
    const int raw_idx = cir.idx<CI_ALIVE>(index);
    //printf("executed %d %d alh:%d\n",cir.nums[CS_alive_hd] - cir.nums[CS_count_removed]+rest-1, raw_idx,cir.nums[CS_alive_hd]);

    sstr.set_from(cir.nums[CS_alive_hd] - cir.nums[CS_count_removed]+rest-1, raw_idx);
}

__global__ void finalize_remove_MUSUME_NOPAIR(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int rest = cir.size<CI_MUSUME_NOPAIR>() - index;
    if (rest <= 0 || rest>cir.nums[CS_count_removed])return;
    const int raw_idx = cir.idx<CI_MUSUME_NOPAIR>(index);
    sstr.set_from(cir.nums[CS_musume_hd] - cir.nums[CS_count_removed] + rest-1, raw_idx);
}

__global__ void finalize_remove_PAIR(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int rest = cir.size<CI_PAIR>() - index;
    if (rest <= 0 || rest>cir.nums[CS_count_removed])return;
    const int raw_idx = cir.idx<CI_PAIR>(index);
    const int dest = cir.nums[CS_pair_hd] - cir.nums[CS_count_removed] + rest-1;
    sstr.set_from(dest, raw_idx);
    sstr.cat[sstr.cat[dest].pair].pair = dest;
  //  printf("\nremoved  term %d %d %d %d\n",rest, cir.nums[CS_count_removed],dest,raw_idx);
}

__global__ void _remove_count_apply_and_init(int*swp_data,  CellIterateRange_device cir) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cir.nums[CS_dead_hd] -= cir.nums[CS_count_removed_air];
        cir.nums[CS_alive_hd] -= cir.nums[CS_count_removed];
        cir.nums[CS_musume_hd] -= cir.nums[CS_count_removed];
        cir.nums[CS_pair_hd] -= cir.nums[CS_count_removed];
       // if (cir.nums[CS_count_removed] != 0)printf("\npairhd %d m:%d\n", cir.nums[CS_pair_hd], cir.nums[CS_count_removed]);
        cir.nums[CS_asz] -= cir.nums[CS_count_removed];
    }
    
    __syncthreads();
    _swp_init(swp_data,  cir);
}



__global__ void divide_cell_MUSUME(curandState* cur, SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    //const int rest = cir.size<CI_MUSUME_NOPAIR>() - index;
    if (index >= cir.size<CI_MUSUME_NOPAIR>())return;
    const int raw_idx = cir.idx<CI_MUSUME_NOPAIR>(index);
    CellAttr& mycat = sstr.cat[raw_idx];
    if (sstr.cat[raw_idx].rest_div_times > 0
        && is_divide_ready(&cur[index], sstr.cat[raw_idx].ageb, agki_max, sstr.cat[raw_idx].is_malignant,false)) {
        
        CellAttr& mycat = sstr.cat[raw_idx];
        if (mycat.dermis < 0) {
            printf("no dermis on divide  ???????\n");
            assert(false);
        }
        const int newc = atomicAdd(&cir.nums[CS_asz], 1);
        
        CellAttr& ncat = sstr.cat[newc];
        ncat.ageb = 0.0;
        ncat.agek = 0.0;
        ncat.ca2p_avg = mycat.ca2p_avg;
        ncat.ex_fat = 0.0;
        ncat.ex_inert = mycat.ex_inert;
        ncat.fix_origin = mycat.fix_origin;
        ncat.in_fat = 0.0;
        ncat.IP3 = mycat.IP3;
        ncat.is_malignant = mycat.is_malignant;
        ncat.radius = mycat.radius;
        ncat.rest_div_times = mycat.rest_div_times = (mycat.rest_div_times-1);

        ncat.spr_nat_len = mycat.spr_nat_len = delta_L;
        mycat.ageb = 0.0;
        mycat.pair = newc;
        ncat.pair = raw_idx;//fix later
        sstr.cst[newc]=MUSUME;
        real4& mypos = sstr.cpos[raw_idx];
        real4& npos = sstr.cpos[newc];
        const real4 div_vec = cvmul(real(0.5)*delta_L,div_direction(&cur[index], mypos, sstr.cpos[mycat.dermis]));


        npos = vsub(mypos, div_vec);
        DBG_ONLY(printf("%f %f %f %d\n",npos.x,npos.y,npos.z,newc));
        vadda(&mypos, div_vec);


        sstr.swp_data[raw_idx] = -2;
        atomicAdd(&cir.nums[CS_count_musume_divided], 1);
        DBG_ONLY(printf("NEW MUSUME %d\n", newc));
        DBG_ONLY(printf("MUSUME->PAIR %d\n", raw_idx));
    }
    else {
        mycat.ageb += DT_Cell*ageb_const(mycat.ca2p_avg, mycat.is_malignant);
    }
}

__global__ void prepare_state_change_MUSUME_to_pair2(SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    const int rest = cir.size<CI_MUSUME_NOPAIR>() - index;
    if (rest <= 0 || rest>cir.nums[CS_count_musume_divided])return;
    const int raw_idx = cir.idx<CI_MUSUME_NOPAIR>(index);

    if (sstr.swp_data[raw_idx] == -1) {
        const int av = atomicAdd(&cir.nums[CS_count_available], 1);
        DBG_ONLY(printf("muno:%d\n",av));
        sstr.arr_avail[av] = raw_idx; //order will be broken but doesnt matter
    }
    else if (sstr.swp_data[raw_idx] == -2) {
        sstr.swp_data[raw_idx] = -1;
    }
}

__global__ void exec_state_change_MUSUME_to_pair(SwapStruct sstr, CellIterateRange_device cir) {

    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_MUSUME_NOPAIR>())return;
    const int raw_idx = cir.idx<CI_MUSUME_NOPAIR>(index);
    if (sstr.swp_data[raw_idx] == -2) {
        const int st = atomicAdd(&cir.nums[CS_count_store], 1);
        DBG_ONLY(printf("muno2:%d\n",st));
        const int dest = sstr.arr_avail[st];
        const int pair_reserved = sstr.cat[raw_idx].pair;
        __syncthreads();
        sstr.cat[pair_reserved].pair=dest;
        __syncthreads();
        sstr.raw_swap(dest, raw_idx);
        //sstr.cat[sstr.cat[dest].pair].pair = dest;
        DBG_ONLY(printf("pair check: max:%d %d %d\n",cir.nums[CS_asz],sstr.cat[dest].pair,sstr.cst[sstr.cat[dest].pair]));
    }
}

__global__ void _divide_count_apply_and_init(int*swp_data, CellIterateRange_device cir) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cir.nums[CS_pair_hd] -= cir.nums[CS_count_musume_divided];
    }
    //__syncthreads();
    _swp_init(swp_data, cir);
}

__global__ void divide_cell_FIX(curandState* cur, SwapStruct sstr, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    //const int rest = cir.size<CI_MUSUME_NOPAIR>() - index;
    if (index >= cir.size<CI_FIX>())return;
    const int raw_idx = cir.idx<CI_FIX>(index);
    CellAttr& mycat = sstr.cat[raw_idx];
    if (is_divide_ready(&cur[index], sstr.cat[raw_idx].ageb, agki_max_fix, sstr.cat[raw_idx].is_malignant, false)) {

        
        if (mycat.dermis < 0) {
            printf("no dermis on divide  ???????\n");
            assert(false);
        }
        const int newc = atomicAdd(&cir.nums[CS_asz], 1);

        CellAttr& ncat = sstr.cat[newc];
        ncat.ageb = 0.0;
        ncat.agek = 0.0;
        ncat.ca2p_avg = mycat.ca2p_avg;
        ncat.ex_fat = 0.0;
        ncat.ex_inert = mycat.ex_inert;
        ncat.fix_origin = mycat.fix_origin;
        ncat.in_fat = 0.0;
        ncat.IP3 = mycat.IP3;
        ncat.is_malignant = mycat.is_malignant;
        ncat.radius = mycat.radius;

        ncat.rest_div_times = mycat.rest_div_times;//

        ncat.spr_nat_len = mycat.spr_nat_len = delta_L;
        mycat.ageb = 0.0;
        mycat.pair = newc;
        ncat.pair = raw_idx;//fix later
        sstr.cst[newc]=MUSUME;
        real4& mypos = sstr.cpos[raw_idx];
        real4& npos = sstr.cpos[newc];
        const real4 div_vec = cvmul(real(0.5)*delta_L, div_direction(&cur[index], mypos, sstr.cpos[mycat.dermis]));

        npos = vsub(mypos, div_vec);
        DBG_ONLY(printf("%f %f %f %d\n",npos.x,npos.y,npos.z,newc));
                vadda(&mypos, div_vec);

        DBG_ONLY(printf("NEW PAIR %d\n", newc));
        DBG_ONLY(printf("FIX->*PAIR %d\n", raw_idx));
    }
    else {
        mycat.ageb += DT_Cell*ageb_const(mycat.ca2p_avg, mycat.is_malignant);
    }
}
__device__ static int nummap(CELL_STATE st){
    switch (st) {
    case MEMB:
        return 0;
    case FIX:case DUMMY_FIX:
        return 1;
    case DER:
        return 2;
    case AIR:
        return 3;
    case DEAD:
        return 4;
    case ALIVE:
        return 5;
    case MUSUME:
        return 6;
    case UNUSED:
    case DISA:
    case BLANK:
        return 1023;
    default:
        return -1;
    	//throw std::runtime_error("Undefined cell state found.");
    }
}
__global__ void check_order(SwapStruct sstr, CellIterateRange_device cir){
	if(threadIdx.x!=0||blockIdx.x!=0)return;
	int last_state=nummap(MEMB);
	bool last_mu_pair=false;
	for(int i=0;i<cir.nums[CS_asz];i++){
		int current_state=nummap(sstr.cst[i]);
		bool current_mu_pair=sstr.cat[i].pair>=0 && i>=cir.nums[CS_musume_hd];
		if(last_state!=current_state){
		if(last_state>=current_state){
			printf("wrong order:%d %d %d\n",i,current_state,last_state);
			break;
		}else{
			int head=-1;
			switch(sstr.cst[i]){
		    case MEMB:
		        head=0;
		        break;
		    case FIX:case DUMMY_FIX:
		        head=cir.nums[CS_fix_hd];
		        break;
		    case DER:
		    	head=cir.nums[CS_der_hd];
		    			        break;
		    case AIR:
		    	head=cir.nums[CS_air_hd];
		    			        break;
		    case DEAD:
		    	head=cir.nums[CS_dead_hd];
		    			        break;
		    case ALIVE:
		    	head=cir.nums[CS_alive_hd];
		    			        break;
		    case MUSUME:
		    	head=cir.nums[CS_musume_hd];
		    			        break;
		    			        default:
		    			        break;
			}
			if(i!=head){
				printf("wrong head %d %d %d\n",i,head,current_state);
				break;
			}
		}
		}

		if(last_mu_pair&&!current_mu_pair){
			printf("pair wrong order:%d\n",i);
						break;
		}
		if(sstr.cat[i].pair>=0){
			if(sstr.cat[sstr.cat[i].pair].pair!=i){
				printf("wrong pairing:%d pair:%d return:%d\n",i,sstr.cat[i].pair,sstr.cat[sstr.cat[i].pair].pair);
		}
		}
		last_mu_pair=current_mu_pair;
		last_state=current_state;
	}
}
__global__ void check_order_parallel(SwapStruct sstr, CellIterateRange_device cir){
	const int index = threadIdx.x + blockIdx.x*blockDim.x;
	if(index>=cir.nums[CS_asz]-1)return;
	if(nummap(sstr.cst[index+1])<nummap(sstr.cst[index])){
		printf("wrong order:%d %d %d\n",index,nummap(sstr.cst[index]),nummap(sstr.cst[index+1]));
		assert(false);
	}
}
void exec_renew(CellManager&cm) {
    static rand_generator rng(2048*10);
    SwapStruct sstr; CellIterateRange_device cir = cm.get_cell_iterate_range_d();
    sstr.arr_avail = cm.swap_idx_store_ptr();
    sstr.swp_data = cm.swap_data_ptr();
    sstr.cat = cm.get_device_attr();
    sstr.cpos = cm.get_device_pos_all();
    sstr.cst = cm.get_device_cstate();
    sstr.nmconn = cm.get_device_all_nm_conn();
    size_t asz = cm.all_size();
#define _KERM(a,b) <<<a,b>>>
#define _NTH(fn) fn _KERM((asz)/64+1,64)(sstr,cir)
    //do in this order
    
    _NTH(prepare_state_change_ALIVE);
    _NTH(prepare_state_change_ALIVE2);
    _NTH(prepare_state_change_MUSUME_NOPAIR);
    _NTH(prepare_state_change_MUSUME_NOPAIR2);
    _NTH(prepare_state_change_PAIR);
    _NTH(prepare_state_change_PAIR2);
    _NTH(exec_state_change_ALIVE);
    _NTH(exec_state_change_MUSUME_NOPAIR);
    _NTH(exec_state_change_PAIR);
    _state_change_count_apply_and_init _KERM(asz / 64 + 1, 64) (sstr.swp_data, cir);
   
    _NTH(prepare_remove_AIR);
    _NTH(prepare_remove_AIR2);
    _NTH(prepare_remove_DEAD);
    _NTH(prepare_remove_DEAD2);
    _NTH(exec_remove_AIR);
    _NTH(exec_remove_DEAD);
    _NTH(finalize_remove_DEAD);
    _NTH(finalize_remove_ALIVE);
    _NTH(finalize_remove_MUSUME_NOPAIR);
    _NTH(finalize_remove_PAIR);
    _remove_count_apply_and_init _KERM(asz / 64 + 1, 64) (sstr.swp_data, cir);
    
    divide_cell_MUSUME _KERM(asz / 64 + 1, 64) (rng.get_state(), sstr, cir);
    divide_cell_FIX _KERM(asz / 64 + 1, 64) (rng.get_state(), sstr, cir);
    _NTH(prepare_state_change_MUSUME_to_pair2);
    _NTH(exec_state_change_MUSUME_to_pair);
    _divide_count_apply_and_init _KERM(asz / 64 + 1, 64) (sstr.swp_data, cir);

/*
   cudaDeviceSynchronize();
   check_order<<<1,1>>>(sstr,cir);
   cudaDeviceSynchronize();
*/
    /*
    cudaDeviceSynchronize();
check_order_parallel<<<(asz*2)/64+1,64>>>(sstr,cir);
cudaDeviceSynchronize();
*/
}
