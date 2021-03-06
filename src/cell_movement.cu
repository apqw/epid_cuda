#include "define.h"
#include "cell_movement.h"
#include "math_helper.h"
#include "CellManager.h"

#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"
//#include "filter.h"

template<class Fn>
__device__ void cell_movement_calc_dbg(real4* accum_out, const real4 c1, const real4 c2, Fn calc_fn,int idx1,int idx2,int st1,int st2) {
	const real tmp=calc_fn(c1, c2);
if(fabs(tmp)>100){
		printf("too strong interaction:%f cell:%d %d state:%d %d\n",tmp,idx1,idx2,st1,st2);
		assert(false);
	}else if (fabs(tmp)>10){
		printf("warn: strong interaction:%f cell:%d %d state:%d %d\n",tmp,idx1,idx2,st1,st2);
	}
    vadda(accum_out, cvmul(calc_fn(c1, c2), vpsub(c1, c2)));
}

template<class Fn>
__device__ real cell_movement_calc(real4* accum_out, const real4 c1, const real4 c2, Fn calc_fn) {
	const real tmp=calc_fn(c1, c2);
    vadda(accum_out, cvmul(tmp, vpsub(c1, c2)));
    return tmp;
}
template<class Fn>
__device__ real cell_movement_calc_eachAdd(real4* accum_out, real4* op_add, const real4 c1, const real4 c2, Fn calc_fn) {
	const real tmp=calc_fn(c1, c2);
    const real4 diff = cvmul(calc_fn(c1, c2), vpsub(c1, c2));
    atomicvsuba(op_add, cvmul(DT_Cell, diff));
    vadda(accum_out, diff);
    return tmp;
}
/////////calc funcs
__device__ inline bool no_double_count(const CellIndex c1, const CellIndex c2) {
    return c1 < c2;
}

__device__ inline bool is_der_near(const real4 d1, const real4 d2) {
    constexpr real rdiff = NON_MEMB_RAD_SUM - delta_R;
    return p_dist_sq(d1, d2) < rdiff*rdiff;
}

__device__ inline bool is_near_nm(const real4 d1, const real4 d2) {
    return p_dist_sq(d1, d2) < NON_MEMB_RAD_SUM*NON_MEMB_RAD_SUM;
}

__device__ inline bool is_near_nm_memb(const real4 d1, const real4 d2) {
    return p_dist_sq(d1, d2) < (NON_MEMB_RAD + MEMB_RAD)*(NON_MEMB_RAD + MEMB_RAD);
}

__device__ inline real ljmain_der_near(const real4 d1, const real4 d2) {
    const real dist = sqrt(p_dist_sq(d1, d2));
    const real dist2 = dist + delta_R;
    const real LJ6 = POW6(NON_MEMB_RAD_SUM) / POW6(dist2);
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (dist*dist2) + para_ljp2;
}

__device__ inline real ljmain_der_far(const real4 d1, const real4 d2) {
    const real dist = sqrt(p_dist_sq(d1, d2));
    const real dist2 = dist + delta_R;
    const real LJ6 = POW6(NON_MEMB_RAD_SUM) / POW6(dist2);
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (dist*dist) + para_ljp2;
}

__device__ inline real ljmain_nm(const real4 d1, const real4 d2) {

    const real dist_sq = p_dist_sq(d1, d2);
    const real LJ6 = POW6(NON_MEMB_RAD_SUM) / POW3(dist_sq);
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / dist_sq;

}

__device__ inline real ljmain_nm_memb(const real4 d1, const real4 d2) {

    const real dist_sq = p_dist_sq(d1, d2);
    const real LJ6 = POW6(NON_MEMB_RAD + MEMB_RAD) / POW3(dist_sq);
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / dist_sq;

}

__device__ inline real adhesion_nm(const real4 c1, const real4 c2, const real spring_const) {
    const real distlj = sqrt(p_dist_sq(c1, c2));
    constexpr real rad_sum = NON_MEMB_RAD_SUM;

    const real LJ2_m1 = (distlj / rad_sum) - real(1.0);
    return (LJ2_m1 + real(1.0) > LJ_THRESH ?
        real(0.0) :
        -(spring_const / distlj)
        * LJ2_m1*(real(1.0) - LJ2_m1*LJ2_m1 / ((LJ_THRESH - real(1.0))*(LJ_THRESH - real(1.0)))));
}

__device__ inline real adhesion_nm_memb(const real4 c1, const real4 c2, const real spring_const) {
    const real distlj = sqrt(p_dist_sq(c1, c2));
    constexpr real rad_sum = NON_MEMB_RAD + MEMB_RAD;

    const real LJ2_m1 = (distlj / rad_sum) - real(1.0);
    return (LJ2_m1 + real(1.0) > LJ_THRESH ?
        real(0.0) :
        -(spring_const / distlj)
        * LJ2_m1*(real(1.0) - LJ2_m1*LJ2_m1 / ((LJ_THRESH - real(1.0))*(LJ_THRESH - real(1.0)))));
}


/////////
struct c_der_to_der {
    __device__ real operator()(const real4 d1, const real4 d2)const {

        if (is_der_near(d1, d2)) {
            return ljmain_der_near(d1, d2);
        }
        else if (is_near_nm(d1, d2)) {
            return para_ljp2;
        }
        else {
            return ljmain_der_far(d1, d2);
        }
    }
};

struct c_other_nm {
    __device__ real operator()(const real4 d1, const real4 d2)const {
        return ljmain_nm(d1, d2);
    }
};

struct c_other_nm_memb {
    __device__ real operator()(const real4 d1, const real4 d2)const {
        return ljmain_nm_memb(d1, d2);
    }
};

struct c_memb_to_memb {
    __device__ real operator()(const real4 c1, const real4 c2)const {
        constexpr real cr_dist = MEMB_RAD_SUM*P_MEMB;
        constexpr real cr_dist_inv = 1.0 / cr_dist;
        constexpr real cr_dist_sq = cr_dist*cr_dist;


        const real dist_sq = p_dist_sq(c1, c2);
        if (dist_sq < cr_dist_sq) {

            const real LJ6 = POW3(cr_dist_sq) / POW3(dist_sq);
            return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / dist_sq;
        }
        else if (dist_sq < MEMB_RAD_SUM_SQ) {

            return -DER_DER_CONST*(cr_dist_inv - rsqrt(dist_sq));
            //-(DER_DER_CONST / distlj) * (distlj / cr_dist - 1.0);
        }
        else {
        	printf("3rd calc memb-memb\n");
            const real distlj = sqrt(dist_sq);
            constexpr real lambda_dist = (real(1.0) + P_MEMB)*MEMB_RAD_SUM;
            const real LJ6 = POW6(cr_dist) / POW6(lambda_dist - distlj);
            return -(DER_DER_CONST / MEMB_RAD_SUM)*((real(1.0) - P_MEMB) / P_MEMB)
                - real(4.0) * eps_m*(LJ6*(LJ6 - real(1.0))) / ((lambda_dist - distlj)*distlj);
        }

    }
};
struct c_al_air_de_to_al_air_de_fix_mu {
    const real spf;
    __device__ c_al_air_de_to_al_air_de_fix_mu(real agek1, real agek2) :
        spf(
            agek1 > THRESH_SP&&agek2 > THRESH_SP ?
            K_TOTAL :
            K_DESMOSOME) {}
    __device__ real operator()(const real4 c1, const real4 c2)const {
        if (is_near_nm(c1, c2)) {
            return ljmain_nm(c1, c2);
        }
        else {
            return adhesion_nm(c1, c2, spf);
        }
    }
};
struct c_fix_mu_to_fix_mu {
    __device__ real operator()(const real4 c1, const real4 c2)const {
        if (is_near_nm(c1, c2)) {
            return ljmain_nm(c1, c2);
        }
        else {
            return adhesion_nm(c1, c2, K_DESMOSOME);
        }
    }
};

struct c_fix_to_memb {
    __device__ real operator()(const real4 c1, const real4 c2)const {
        if (is_near_nm_memb(c1, c2)) {
            return ljmain_nm_memb(c1, c2);
        }
        else {
            return -Kspring*(real(1.0) / (NON_MEMB_RAD + MEMB_RAD) - rsqrt(p_dist_sq(c1, c2)));
        }
    }
};

struct c_mu_to_memb {
    const real spr;
    __device__ c_mu_to_memb(real _spr) :spr(_spr) {}
    __device__ real operator()(const real4 c1, const real4 c2)const {
        if (is_near_nm_memb(c1, c2)) {
            return ljmain_nm_memb(c1, c2);
        }
        else {
            return adhesion_nm_memb(c1, c2, spr);
        }
    }
};
struct c_pair {
    const real spr_nat_len;
    __device__ c_pair(real _slen) :spr_nat_len(_slen) {}
    __device__ real operator()(const real4 c1, const real4 c2)const {
        
        return -Kspring_division*(real(1.0) - spr_nat_len*rsqrt(p_dist_sq(c1, c2)));
    }
};

//////////////////// MEMB_BEND
__device__ inline real4 tri_normal(const real4 p1, const real4 p2, const real4 p3) {
    return normalize3_r_dev_w_inv(cross3(vsub(p1, p3), vsub(p2, p3)));
}

__device__ inline real4 tri_rho(const real4 n1, const real4 b1) {
    return vsub(b1, cvmul(dot3(n1, b1), n1));
}

__device__ inline real _memb_bend_diff_factor(const real cdot) {
    if (cdot >= real(0.0)) {
        return (real(1.0) - cdot);
    }
    else {
        return real(1.0) / ((real(1.0) + cdot)*(real(1.0) + cdot));
    }

}

__device__ real4 tri_bend_1(const real4 r1, const real4 a1, const real4 b1, const real4 c1) {
    real4 norm1_pre = cross3(vpsub(r1, b1), vpsub(a1, b1));
    real norm1_pre_norm_inv;
    real4 norm1 = normalize3_r_dev_w_inv(norm1_pre, &norm1_pre_norm_inv), norm2 = normalize3_r_dev_w_inv(cross3(vpsub(a1, b1), vpsub(c1, b1)));
    real dot1 = dot3(norm1, norm2);
    return cvmul(_memb_bend_diff_factor(dot1), cvmul(norm1_pre_norm_inv, cross3(vpsub(a1, b1), tri_rho(norm1, norm2))));
}

__device__ real4 tri_bend_2(const real4 r1, const real4 a1, const real4 b1, const real4 c1) {
    real4 norm1_pre = cross3(vpsub(r1, c1), vpsub(b1, c1));
    real4 norm2_pre = cross3(vpsub(r1, b1), vpsub(a1, b1));

    real norm1_pre_norm_inv, norm2_pre_norm_inv;
    real4 norm1 = normalize3_r_dev_w_inv(norm1_pre, &norm1_pre_norm_inv), norm2 = normalize3_r_dev_w_inv(norm2_pre, &norm2_pre_norm_inv);


    real dot1 = dot3(norm1, norm2);

    return cvmul(_memb_bend_diff_factor(dot1), (vadd(cvmul(norm1_pre_norm_inv, cross3(vpsub(b1, c1), tri_rho(norm1, norm2))), cvmul(norm2_pre_norm_inv, cross3(vpsub(a1, b1), tri_rho(norm2, norm1))))));
}

#define BEND_TH (64)
#define THREAD_DIV_BEND (32)
__global__ void  calc_memb_bend(cudaTextureObject_t pos_tex, const MembConn* mconn, size_t sz, CellPos* out) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;

    if (index < sz) {
        real4 dr = { real(0.0),real(0.0),real(0.0),real(0.0) };
        const CellPos mypos = tex1Dfetch_real4(pos_tex, index);

        for (int i = 0; i < 6; i++) {
            vadda(&dr, tri_bend_1(
                mypos,
                tex1Dfetch_real4(pos_tex, mconn[index].conn[i]),
                tex1Dfetch_real4(pos_tex, mconn[index].conn[(i + 1) % 6]),
                tex1Dfetch_real4(pos_tex, mconn[mconn[index].conn[(i + 1) % 6]].conn[i])
            ));

        }
        for (int i = 0; i < 6; i++) {
            vadda(&dr, tri_bend_2(
                mypos,
                tex1Dfetch_real4(pos_tex, mconn[index].conn[i]),
                tex1Dfetch_real4(pos_tex, mconn[index].conn[(i + 1) % 6]),
                tex1Dfetch_real4(pos_tex, mconn[index].conn[(i + 2) % 6])
            ));

        }
        out[index] = vadd(mypos, cvmul(DT_Cell*KBEND, dr));
    }
}

//////////////////// wall
template<CELL_STATE cst>
__device__ inline bool collide_with_wall(const real zpos) {
    return zpos < NON_MEMB_RAD;
}

template<>
__device__ inline bool collide_with_wall<MEMB>(const real zpos) {
    return zpos < MEMB_RAD;
}

__device__ inline real wall_interaction_memb(const real cz) {
    const real distlj = real(2.0)*cz;
    const real LJ6 = POW6(MEMB_RAD) / POW6(cz);
    const real ljm = real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (distlj*distlj);
    return ljm*2.0*cz;
}

__device__ inline real wall_interaction(const real cz) {
    const real distlj = real(2.0)*cz;
    const real LJ6 = POW6(NON_MEMB_RAD) / POW6(cz);
    const real ljm = real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (distlj*distlj);
    return ljm*2.0*cz;
}

////////////// Kers
#define STRENGTH_TH (real(100.0))
#define STRENGTH_TH_PRE (real(25.0))
#ifdef NDEBUG
#define _TH_CHK(tmp_name,myidx,opidx,pos1,pos2,f...) do{f;}while(0)
#else
#define _TH_CHK(tmp_name,myidx,opidx,pos1,pos2,f...) do{const real tmp_name=f;\
if(fabs(tmp)>STRENGTH_TH){\
	printf("too strong interaction ::\ncoef:%f cell:%d %d\npos1:%f %f %f\npos2:%f %f %f  > " #f "\n",tmp_name,myidx,opidx,\
pos1.x,pos1.y,pos1.z,pos2.x,pos2.y,pos2.z);\
	assert(false);\
}else if (fabs(tmp)>STRENGTH_TH_PRE){\
printf("warn:strong interaction ::\ncoef:%f cell:%d %d\npos1:%f %f %f\npos2:%f %f %f  > " #f "\n",tmp_name,myidx,opidx,\
pos1.x,pos1.y,pos1.z,pos2.x,pos2.y,pos2.z);\
}}while(0)
#endif
__global__ void MEMB_interaction(cudaTextureObject_t pos_tex, const CELL_STATE* cst, const MembConn* mconn,
    const NonMembConn* all_nm_conn, size_t sz, CellPos* out) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= sz)return;
    const CellPos mypos = tex1Dfetch_real4(pos_tex, index);
    real z_dis = real(0.0);
    real4 dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    if (collide_with_wall<MEMB>(mypos.z)) {
        z_dis += wall_interaction_memb(mypos.z);
    }

    for (int i = 0; i < MEMB_CONN_NUM; i++) {
        const real4 op = tex1Dfetch_real4(pos_tex, mconn[index].conn[i]);
        _TH_CHK(tmp,index,mconn[index].conn[i],mypos,op,cell_movement_calc(&dr, mypos, op, c_memb_to_memb()));
    }

    const NonMembConn& nmc = all_nm_conn[index];
    const size_t conn_sz = nmc.conn.size();
    for (int i = 0; i < conn_sz; i++) {
        const int op_idx = nmc.conn[i];
        const real4 op = tex1Dfetch_real4(pos_tex, op_idx);
        if (cst[op_idx] != FIX&&cst[op_idx] != MUSUME){
        	_TH_CHK(tmp,index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_other_nm_memb()));
        }
    }

    dr.z += z_dis;

    vadda(&out[index], cvmul(DT_Cell, dr));

}
#define DER_TH (64)
#define THREAD_DIV_DER (32)
__global__ void DER_interaction(const cudaTextureObject_t pos_tex, const NonMembConn* all_nm_conn,
    const CellIterateRange_device cir,int mbsz, CellPos*out) {

    const int index = threadIdx.x / THREAD_DIV_DER + blockIdx.x*blockDim.x / THREAD_DIV_DER;
    if (index >= cir.size<CI_DER>())return;
    const int th_id = threadIdx.x%THREAD_DIV_DER;
    const int b_id = threadIdx.x / THREAD_DIV_DER;
    __shared__ real4 mypos_sh[DER_TH / THREAD_DIV_DER];
    __shared__ size_t connsz_sh[DER_TH / THREAD_DIV_DER];
    __shared__ real4 out_dr[DER_TH];
    const int raw_cell_index = cir.idx<CI_DER>(index);
    if(threadIdx.x==0){
    	memset(out_dr, 0x00, sizeof(out_dr));
    }
    if (th_id == 0) {

        connsz_sh[b_id] = all_nm_conn[raw_cell_index].conn.size();
        mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, raw_cell_index);
    }
    __syncthreads();
    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = connsz_sh[b_id];
    //if (conn_sz <= th_id)return;


    const CellPos mypos = mypos_sh[b_id];
    real4& dr = out_dr[threadIdx.x];
    for (int i = th_id; i < conn_sz; i += THREAD_DIV_DER) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= mbsz) {
            if (cir.nums[CS_der_hd]<=op_idx&&op_idx<cir.nums[CS_air_hd]) {
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_der_to_der()));

            }
            else {
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_other_nm()));
            }
        }
        else {
        	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_other_nm_memb()));
        }


    }

    __syncthreads();
    //sum reduction
    for (int r = (THREAD_DIV_DER >> 1); r >= 1; r >>= 1) {
        __syncthreads();
        if (th_id < r) {
            vadda(&out_dr[th_id + b_id*THREAD_DIV_DER], out_dr[th_id + r + b_id*THREAD_DIV_DER]);
        }
    }
    if (th_id == 0) {
        out_dr[0 + b_id*THREAD_DIV_DER].z += wall_interaction(mypos.z);
        out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, out_dr[0 + b_id*THREAD_DIV_DER]));
    }
}



__global__ void AL_AIR_DE_interaction(const cudaTextureObject_t pos_tex, const CellAttr* cat,
const NonMembConn* all_nm_conn, const CellIterateRange_device cir, int mbsz, CellPos*out) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_AIR, CI_DEAD, CI_ALIVE>())return;
    const int raw_cell_index = cir.idx<CI_AIR, CI_DEAD, CI_ALIVE>(index);
    const CellPos mypos = tex1Dfetch_real4(pos_tex, raw_cell_index);
    real4 dr = { real(0.0), real(0.0), real(0.0), real(0.0) };

    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = nmc.conn.size();
    const real myagek = cat[raw_cell_index].agek;
    for (int i = 0; i < conn_sz; i++) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= mbsz || (cir.nums[CS_der_hd] <= op_idx&&op_idx<cir.nums[CS_air_hd])) {
        	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op,
                c_al_air_de_to_al_air_de_fix_mu(myagek, cat[op_idx].agek)));
        }
        else {
        	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_other_nm_memb()));
        }


    }

    out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, dr));
}



#define FIX_TH (32)
#define THREAD_DIV_FIX (32)
__global__ void FIX_interaction(const cudaTextureObject_t pos_tex, const CellAttr* cat,
    const CELL_STATE* cst, const NonMembConn* all_nm_conn, const CellIterateRange_device cir, size_t mbsz, CellPos*out) {

    const int index = threadIdx.x / THREAD_DIV_FIX + blockIdx.x*blockDim.x / THREAD_DIV_FIX;
    if (index >= cir.size<CI_FIX>())return;
    const int th_id = threadIdx.x%THREAD_DIV_FIX;
    const int b_id = threadIdx.x / THREAD_DIV_FIX;
    __shared__ real4 mypos_sh[FIX_TH / THREAD_DIV_FIX];
    __shared__ size_t connsz_sh[FIX_TH / THREAD_DIV_FIX];
    __shared__ real4 out_dr[FIX_TH];
    const int raw_cell_index = cir.idx<CI_FIX>(index);
    if(threadIdx.x==0){
        	memset(out_dr, 0x00, sizeof(out_dr));
        }
    if (th_id == 0) {

        connsz_sh[b_id] = all_nm_conn[raw_cell_index].conn.size();
        mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, raw_cell_index);
    }
    __syncthreads();
    
    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = connsz_sh[b_id];
    //if (conn_sz <= th_id)return;


    const CellPos mypos = mypos_sh[b_id];
    real4& dr = out_dr[threadIdx.x]; dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    const CellAttr& myattr = cat[raw_cell_index];




    for (int i = th_id; i < conn_sz; i += THREAD_DIV_FIX) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= mbsz) {
            const CELL_STATE st = cst[op_idx];

            switch (st) {
            case FIX:case MUSUME:
                if (myattr.pair != op_idx) {
                	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_fix_mu_to_fix_mu()));
                }
                break;
            case ALIVE:case AIR:case DEAD:

            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op,
                    c_al_air_de_to_al_air_de_fix_mu(myattr.agek, cat[op_idx].agek)));

                break;
            default:

            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_other_nm()));

                break;
            }
        }
        else {

            if (myattr.dermis == op_idx) {
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_fix_to_memb()));
            }
            else {
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_other_nm_memb()));
            }

        }


    }
    __syncthreads();
    //sum reduction
    for (int r = (THREAD_DIV_FIX >> 1); r >= 1; r >>= 1) {
        __syncthreads();
        if (th_id < r) {
            vadda(&out_dr[th_id + b_id*THREAD_DIV_FIX], out_dr[th_id + r + b_id*THREAD_DIV_FIX]);
        }
    }
    if (th_id == 0) {
        out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, out_dr[0 + b_id*THREAD_DIV_FIX]));
    }
}
#define MUSUME_TH (64)
#define THREAD_DIV_MU (32)
__global__ void MUSUME_interaction(const cudaTextureObject_t pos_tex, const CellAttr* cat,
    const CELL_STATE* cst, const NonMembConn* all_nm_conn, const CellIterateRange_device cir, size_t _mbsz, CellPos*out) {
    const int index = threadIdx.x / THREAD_DIV_MU + blockIdx.x*blockDim.x / THREAD_DIV_MU;
    if (index >= cir.size<CI_MUSUME>())return;
    const int th_id = threadIdx.x%THREAD_DIV_MU;
    const int b_id = threadIdx.x / THREAD_DIV_MU;
    __shared__ real4 mypos_sh[MUSUME_TH / THREAD_DIV_MU];
    __shared__ size_t connsz_sh[MUSUME_TH / THREAD_DIV_MU];
    __shared__ real4 out_dr[MUSUME_TH];
    const int raw_cell_index = cir.idx<CI_MUSUME>(index);
    if(threadIdx.x==0){
        	memset(out_dr, 0x00, sizeof(out_dr));
        }
    if (th_id == 0) {

        connsz_sh[b_id] = all_nm_conn[raw_cell_index].conn.size();
        mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, raw_cell_index);
    }
    __syncthreads();
    
    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = connsz_sh[b_id];
    //if (conn_sz <= th_id)return;
    const CellPos mypos = mypos_sh[b_id];
    real4& dr = out_dr[threadIdx.x]; dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    const CellAttr& myattr = cat[raw_cell_index];
    const bool paired_with_fix =
        myattr.pair >= 0 ?
        cst[myattr.pair] == FIX ? true : false
        : false;
    const real spr = paired_with_fix ? Kspring :
        myattr.rest_div_times > 0 ? Kspring_d : 0;




    for (int i = th_id; i < conn_sz; i += THREAD_DIV_MU) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= _mbsz) {
            const CELL_STATE st = cst[op_idx];

            switch (st) {
            case FIX:case MUSUME:
                if (myattr.pair != op_idx) {
                	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_fix_mu_to_fix_mu()));
                }
                break;
            case ALIVE:case AIR:case DEAD:
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op,
                    c_al_air_de_to_al_air_de_fix_mu(myattr.agek, cat[op_idx].agek)));
                break;
            default:
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc(&dr, mypos, op, c_other_nm()));
                break;
            }
        }
        else {

            if (myattr.dermis == op_idx) {
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_mu_to_memb(spr)));
            }
            else {
            	_TH_CHK(tmp,raw_cell_index,op_idx,mypos,op,cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_other_nm_memb()));
            }

        }


    }
    __syncthreads();
    //sum reduction
    for (int r = (THREAD_DIV_MU >> 1); r >= 1; r >>= 1) {
        __syncthreads();
        if (th_id < r) {
            vadda(&out_dr[th_id + b_id*THREAD_DIV_MU], out_dr[th_id + r + b_id*THREAD_DIV_MU]);
        }
    }
    if (th_id == 0) {
        out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, out_dr[0 + b_id*THREAD_DIV_MU]));
    }
}


__global__ void pair_interaction(const cudaTextureObject_t pos_tex, const CellAttr* _cat,
     const CellIterateRange_device cir, CellPos*out) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_PAIR>())return;
    const int raw_cell_index = cir.idx<CI_PAIR>(index);
    const CellAttr& cat = _cat[raw_cell_index];
    if (raw_cell_index<cat.pair)return;
    real4 dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    
    const CellPos mypos = tex1Dfetch_real4(pos_tex, raw_cell_index);
    const CellPos pairpos = tex1Dfetch_real4(pos_tex, cat.pair);
    _TH_CHK(tmp,raw_cell_index,cat.pair,mypos,pairpos,cell_movement_calc(&dr, mypos, pairpos, c_pair(cat.spr_nat_len)));
    vadda(&out[raw_cell_index], cvmul(DT_Cell, dr));
    vadda(&out[cat.pair], cvmul(-DT_Cell, dr));
}
void calc_cell_movement(CellManager&cman) {
    size_t msz = cman.memb_size();
    size_t nmsz = cman.non_memb_size();
    CellIterateRange_device cird = cman.get_cell_iterate_range_d();
    calc_memb_bend << <(unsigned int)msz / 32 + 1, 32 >> > (cman.get_pos_tex(), cman.get_device_mconn(), msz, cman.get_device_pos_all_out());

    MEMB_interaction << <(unsigned int)msz / 64 + 1, 64>> > (cman.get_pos_tex(), cman.get_device_cstate(), cman.get_device_mconn(), cman.get_device_all_nm_conn(), msz, cman.get_device_pos_all_out());
    
    DER_interaction << <THREAD_DIV_DER*nmsz / DER_TH + 1, DER_TH >> > (cman.get_pos_tex(),
        cman.get_device_all_nm_conn(), cird,msz, cman.get_device_pos_all_out());
    

    AL_AIR_DE_interaction << <nmsz / 64 + 1, 64 >> > (cman.get_pos_tex(),
        cman.get_device_attr(),
         cman.get_device_all_nm_conn(), cird, msz, cman.get_device_pos_all_out());
    

    FIX_interaction << <THREAD_DIV_FIX*nmsz / FIX_TH + 1, FIX_TH >> > (cman.get_pos_tex(),
        cman.get_device_attr(),
        cman.get_device_cstate(), cman.get_device_all_nm_conn(), cird, msz, cman.get_device_pos_all_out());
    
    

    MUSUME_interaction << <THREAD_DIV_MU* nmsz / MUSUME_TH + 1, MUSUME_TH >> > (cman.get_pos_tex(),
        cman.get_device_attr(),
        cman.get_device_cstate(), cman.get_device_all_nm_conn(), cird, msz, cman.get_device_pos_all_out());
   
    

    pair_interaction << <nmsz / 128 + 1, 128 >> > (cman.get_pos_tex(),
        cman.get_device_attr(),
        cird, cman.get_device_pos_all_out());
        

    cman.pos_swap_device();

}
