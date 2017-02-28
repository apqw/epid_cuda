#include "define.h"
#include "cell_movement.h"
#include "math_helper.h"
#include "CellManager.h"

#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"

template<class Fn>
__device__ void cell_movement_calc(real4* accum_out, const real4 c1, const real4 c2, Fn calc_fn) {
    vadda(accum_out, cvmul(calc_fn(c1, c2), vpsub(c1, c2)));
}

template<class Fn>
__device__ void cell_movement_calc_eachAdd(real4* accum_out,real4* op_add, const real4 c1, const real4 c2, Fn calc_fn) {
    const real4 diff = cvmul(calc_fn(c1, c2), vpsub(c1, c2));
    atomicvsuba(op_add, cvmul(DT_Cell,diff));
    vadda(accum_out, diff);
}
//////////////////// MEMB_BEND
__device__ inline real4 tri_normal(const real4 p1, const real4 p2, const real4 p3) {
    return normalize3_r_dev_w_inv(cross3(vsub(p1, p3), vsub(p2, p3)));
}

__device__ inline real4 tri_rho(const real4 n1,const real4 b1) {
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
    //vec3 r1v = ctov(r1),a1v=ctov(a1),b1v=ctov(b1),c1v=ctov(c1);
    real4 norm1_pre = cross3(vpsub(r1, b1), vpsub(a1, b1));
    real norm1_pre_norm_inv;
    real4 norm1 = normalize3_r_dev_w_inv(norm1_pre, &norm1_pre_norm_inv), norm2 = normalize3_r_dev_w_inv(cross3(vpsub(a1, b1), vpsub(c1, b1)));
    real dot1 = dot3(norm1, norm2);
    return cvmul(_memb_bend_diff_factor(dot1), cvmul(norm1_pre_norm_inv,cross3(vpsub(a1, b1), tri_rho(norm1, norm2))));
}

__device__ real4 tri_bend_2(const real4 r1, const real4 a1, const real4 b1, const real4 c1) {
    //vec3 r1v = ctov(r1),a1v=ctov(a1),b1v=ctov(b1),c1v=ctov(c1);
    real4 norm1_pre = cross3(vpsub(r1, c1), vpsub(b1, c1));
    real4 norm2_pre = cross3(vpsub(r1, b1), vpsub(a1, b1));

    real norm1_pre_norm_inv, norm2_pre_norm_inv;
    real4 norm1 = normalize3_r_dev_w_inv(norm1_pre, &norm1_pre_norm_inv), norm2 = normalize3_r_dev_w_inv(norm2_pre, &norm2_pre_norm_inv);


    real dot1 = dot3(norm1, norm2);

    return cvmul(_memb_bend_diff_factor(dot1), (vadd(cvmul(norm1_pre_norm_inv,cross3(vpsub(b1, c1), tri_rho(norm1, norm2))), cvmul(norm2_pre_norm_inv,cross3(vpsub(a1, b1), tri_rho(norm2, norm1))))));
}

#define BEND_TH (64)
#define THREAD_DIV_BEND (32)
__global__ void  calc_memb_bend(cudaTextureObject_t pos_tex,const MembConn* mconn,size_t sz,CellPos* out) {
    volatile int index = threadIdx.x + blockIdx.x*blockDim.x;

    if (index<sz) {
        //Memb* memb = mbptr + index;
        real4 dr = { real(0.0),real(0.0),real(0.0),real(0.0) };
        const CellPos mypos = tex1Dfetch_real4(pos_tex, index);
 
        for (int i = 0; i<6; i++) {
            //if(memb->md.memb_spr[i]>0.01 && memb->md.memb_spr[(i+1)%6]>0.01 )
            vadda(&dr, tri_bend_1(
                mypos,
                tex1Dfetch_real4(pos_tex, mconn[index].conn[i]),//mbptr + memb->adj_memb[i],
                tex1Dfetch_real4(pos_tex, mconn[index].conn[(i + 1) % 6]),
                tex1Dfetch_real4(pos_tex, mconn[mconn[index].conn[(i + 1) % 6]].conn[i])
            ));
                //mbptr + memb->adj_memb[(i + 1) % 6], mbptr + (mbptr + memb->adj_memb[(i + 1) % 6])->adj_memb[i]));

        }
        for (int i = 0; i<6; i++) {
            //if(memb->md.memb_spr[i]>0.01 && memb->md.memb_spr[(i+1)%6]>0.01 )
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
    //LJ6 = LJ6*LJ6;
    //LJ6 = LJ6*LJ6*LJ6;
    const real ljm = real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (distlj*distlj);
    return ljm*2.0*cz;
}

__device__ inline real wall_interaction(const real cz) {
    const real distlj = real(2.0)*cz;
    const real LJ6 = POW6(NON_MEMB_RAD) / POW6(cz);
    //LJ6 = LJ6*LJ6;
    //LJ6 = LJ6*LJ6*LJ6;
    const real ljm = real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (distlj*distlj);
    return ljm*2.0*cz;
}


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
    //const double rad_sum = der1->radius + der2->radius;
    const real LJ6 = POW6(NON_MEMB_RAD_SUM) / POW6(dist2);
    //LJ6 = LJ6*LJ6*LJ6;
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (dist*dist2) + para_ljp2;
}

__device__ inline real ljmain_der_far(const real4 d1, const real4 d2) {
    const real dist = sqrt(p_dist_sq(d1, d2));
    const real dist2 = dist + delta_R;
    //const double rad_sum = der1->radius + der2->radius;
    const real LJ6 = POW6(NON_MEMB_RAD_SUM) / POW6(dist2);
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / (dist*dist) + para_ljp2;
}

__device__ inline real ljmain_nm(const real4 d1, const real4 d2) {

    const real dist_sq = p_dist_sq(d1, d2);
    // const double rad_sum = c1->radius + c2->radius;
    const real LJ6 = POW6(NON_MEMB_RAD_SUM) / POW3(dist_sq);
    //LJ6 = LJ6*LJ6*LJ6;
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / dist_sq;

}

__device__ inline real ljmain_nm_memb(const real4 d1, const real4 d2) {

    const real dist_sq = p_dist_sq(d1, d2);
    // const double rad_sum = c1->radius + c2->radius;
    const real LJ6 = POW6(NON_MEMB_RAD + MEMB_RAD) / POW3(dist_sq);
    //LJ6 = LJ6*LJ6*LJ6;
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / dist_sq;

}

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

///////////MEMB
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
            const real distlj = sqrt(dist_sq);
            constexpr real lambda_dist = (real(1.0) + P_MEMB)*MEMB_RAD_SUM;
            const real LJ6 = POW6(cr_dist) / POW6(lambda_dist - distlj);
            return -(DER_DER_CONST / MEMB_RAD_SUM)*((real(1.0) - P_MEMB) / P_MEMB)
                - real(4.0) * eps_m*(LJ6*(LJ6 - real(1.0))) / ((lambda_dist - distlj)*distlj);
        }

    }
};


__global__ void MEMB_interaction(cudaTextureObject_t pos_tex,const CELL_STATE* cst,const MembConn* mconn, const NonMembConn* all_nm_conn, size_t sz, CellPos* out) {
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
        cell_movement_calc(&dr, mypos, op, c_memb_to_memb());
    }
    
    const NonMembConn& nmc = all_nm_conn[index];
    const size_t conn_sz = nmc.conn.size();
    for (int i = 0; i < conn_sz; i++) {
        const int op_idx = nmc.conn[i];
        const real4 op = tex1Dfetch_real4(pos_tex, op_idx);
       if(cst[op_idx-sz]!=FIX&&cst[op_idx-sz]!=MUSUME) cell_movement_calc(&dr, mypos, op, c_other_nm_memb());
        //fix interaction dermis
    }
    
    dr.z += z_dis;

    vadda(&out[index] , cvmul(DT_Cell, dr));

}
#define DER_TH (64)
#define THREAD_DIV_DER (32)
__global__ void DER_interaction(const cudaTextureObject_t pos_tex,const CELL_STATE* nmcst, const NonMembConn* all_nm_conn,const CellIndex* DER_filtered, size_t nm_start, size_t sz, CellPos*out) {
    
    const int index = threadIdx.x / THREAD_DIV_DER + blockIdx.x*blockDim.x / THREAD_DIV_DER;
    const int th_id = threadIdx.x%THREAD_DIV_DER;
    const int b_id = threadIdx.x / THREAD_DIV_DER;
    __shared__ real4 mypos_sh[DER_TH / THREAD_DIV_DER];
    __shared__ int rci_sh[DER_TH / THREAD_DIV_DER];
    __shared__ size_t connsz_sh[DER_TH / THREAD_DIV_DER];
    __shared__ real4 out_dr[DER_TH];
    //__shared__ real out_z[DER_TH];
    if (index >= sz)return;
    if (th_id == 0) {
        memset(out_dr, 0x00, sizeof(out_dr));
        //memset(out_z, 0x00, sizeof(out_z));
        rci_sh[b_id] = DER_filtered[index];
        connsz_sh[b_id] = all_nm_conn[rci_sh[b_id]].conn.size();
        mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, rci_sh[b_id]);
    }
    __syncthreads();
    const int raw_cell_index = rci_sh[b_id];
    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = connsz_sh[b_id];
    if (conn_sz <= th_id)return;


    const CellPos mypos = mypos_sh[b_id];
    //real z_dis = real(0.0);
    real4& dr = out_dr[threadIdx.x]; //dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    /*
    real& z_dis = out_z[threadIdx.x];
    if (collide_with_wall<DER>(mypos.z)) {
        z_dis += wall_interaction(mypos.z);
    }
    */
    for (int i = th_id; i < conn_sz; i += THREAD_DIV_DER) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= nm_start) {
            if (nmcst[op_idx - nm_start] == DER) {
               // if (no_double_count(raw_cell_index, nmc.conn[i]))
                    cell_movement_calc(&dr, mypos, op, c_der_to_der());
                
            }
            else {
                cell_movement_calc(&dr, mypos, op, c_other_nm());
            }
        }
        else {
            cell_movement_calc(&dr, mypos, op, c_other_nm_memb());
        }


    }

    __syncthreads();
    //sum reduction
    for (int r = (THREAD_DIV_DER >> 1); r >= 1; r >>= 1) {
        __syncthreads();
        if (th_id < r) {
           // out_z[th_id + b_id*THREAD_DIV_DER] += out_z[th_id +r+ b_id*THREAD_DIV_DER];
            vadda(&out_dr[th_id + b_id*THREAD_DIV_DER], out_dr[th_id + r + b_id*THREAD_DIV_DER]);
        }
    }
    if (th_id == 0) {
        out_dr[0 + b_id*THREAD_DIV_DER].z += wall_interaction(mypos.z);
        out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, out_dr[0+b_id*THREAD_DIV_DER]));
    }
}
__device__ inline real adhesion_nm(const real4 c1, const real4 c2,const real spring_const) {
   // using namespace cont;
    const real distlj = sqrt(p_dist_sq(c1, c2));
    constexpr real rad_sum = NON_MEMB_RAD_SUM;
    //double LJ2_m1;

    const real LJ2_m1 = (distlj / rad_sum) - real(1.0);
    return (LJ2_m1 + real(1.0) > LJ_THRESH ?
        real(0.0) :
        -(spring_const / distlj)
        * LJ2_m1*(real(1.0) - LJ2_m1*LJ2_m1 / ((LJ_THRESH - real(1.0))*(LJ_THRESH - real(1.0)))));
}

__device__ inline real adhesion_nm_memb(const real4 c1, const real4 c2, const real spring_const) {
    // using namespace cont;
    const real distlj = sqrt(p_dist_sq(c1, c2));
    constexpr real rad_sum = NON_MEMB_RAD + MEMB_RAD;
    //double LJ2_m1;

    const real LJ2_m1 = (distlj / rad_sum) - real(1.0);
    return (LJ2_m1 + real(1.0) > LJ_THRESH ?
        real(0.0) :
        -(spring_const / distlj)
        * LJ2_m1*(real(1.0) - LJ2_m1*LJ2_m1 / ((LJ_THRESH - real(1.0))*(LJ_THRESH - real(1.0)))));
}
struct c_al_air_de_to_al_air_de_fix_mu {
    const real spf;
    __device__ c_al_air_de_to_al_air_de_fix_mu(real agek1,real agek2):
        spf(
            agek1>THRESH_SP&&agek2>THRESH_SP?
            K_TOTAL:
            K_DESMOSOME){}
    __device__ real operator()(const real4 c1,const real4 c2)const {
        if (is_near_nm(c1, c2)) {
            return ljmain_nm(c1, c2);
        }
        else {
            /*
            double spf = K_DESMOSOME;
            if (c1->agek > cont::THRESH_SP && c2->agek > cont::THRESH_SP) {
                spf = K_TOTAL;
            }
            */
            return adhesion_nm(c1, c2, spf);
        }
    }
};

__global__ void AL_AIR_DE_interaction(const cudaTextureObject_t pos_tex,const CellAttr* nmcat, const CELL_STATE* nmcst, const NonMembConn* all_nm_conn, const CellIndex* AL_AIR_DE_filtered, size_t nm_start, size_t sz, CellPos*out) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= sz)return;
    const int raw_cell_index = AL_AIR_DE_filtered[index];
    const CellPos mypos = tex1Dfetch_real4(pos_tex, raw_cell_index);
    //real z_dis = real(0.0);
    real4 dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    /*
    if (collide_with_wall<DER>(mypos.z)) {
        z_dis += wall_interaction(mypos.z);
    }
    */
    //const int nm_index = raw_cell_index-nm_start;

    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = nmc.conn.size();
    //printf("conn:%d\n", (int)conn_sz);
    for (int i = 0; i < conn_sz; i++) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= nm_start || nmcst[op_idx - nm_start] == DER) {
            /*
            if (nmcst[op_idx - nm_start] == DER) {
                // if (no_double_count(raw_cell_index, nmc.conn[i]))
                cell_movement_calc(&dr, mypos, op, c_der_to_der());

            }
            else {
                cell_movement_calc(&dr, mypos, op, c_other_nm());
            }
            */
            /*
            real spf = K_DESMOSOME;
            if (nmcat[raw_cell_index - nm_start].agek > THRESH_SP &&
                nmcat[op_idx - nm_start].agek > THRESH_SP) {
                spf = K_TOTAL;
            }
            */
            cell_movement_calc(&dr, mypos, op, 
                c_al_air_de_to_al_air_de_fix_mu(nmcat[raw_cell_index - nm_start].agek, nmcat[op_idx - nm_start].agek));
        }
        else {
            cell_movement_calc(&dr, mypos, op, c_other_nm_memb());
        }


    }

    //dr.z += z_dis;
    out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, dr));
}


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
            /*
            const double distlj = sqrt(p_cell_dist_sq(c1, c2));
            const double LJ2 = distlj / (c1->radius + c2->radius);
            return -(Kspring / distlj)*(LJ2 - 1.0);
            */
            //const real distlj_inv = rsqrt(p_dist_sq(c1, c2));
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
#define FIX_TH (32)
#define THREAD_DIV_FIX (32)
__global__ void FIX_interaction(const cudaTextureObject_t pos_tex, const CellAttr* nmcat, const CELL_STATE* nmcst, const NonMembConn* all_nm_conn, const CellIndex* FIX_filtered, size_t nm_start, size_t sz, CellPos*out) {
    /*
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= sz)return;
    const int raw_cell_index = FIX_filtered[index];
    const CellPos mypos = tex1Dfetch_real4(pos_tex, raw_cell_index);
    real4 dr = { real(0.0), real(0.0), real(0.0), real(0.0) };

    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = nmc.conn.size();
    const CellAttr myattr = nmcat[raw_cell_index - nm_start];
    for (int i = 0; i < conn_sz; i++) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= nm_start) {
            const CELL_STATE st = nmcst[op_idx - nm_start];

            switch (st) {
            case FIX:case MUSUME:
                if (myattr.pair != op_idx) {
                    cell_movement_calc(&dr, mypos, op, c_fix_mu_to_fix_mu());
                }
                break;
            case ALIVE:case AIR:case DEAD:
                cell_movement_calc(&dr, mypos, op,
                    c_al_air_de_to_al_air_de_fix_mu(myattr.agek, nmcat[op_idx - nm_start].agek));
                break;
            default:
                cell_movement_calc(&dr, mypos, op, c_other_nm());
                break;
            }
        }
        else {
            
                        if (myattr.dermis == op_idx) {
                            cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_fix_to_memb());
                        }
                        else {
                            cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_other_nm_memb());
                        }
                        
        }


    }

    //dr.z += z_dis;
    out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, dr));
    */

    const int index = threadIdx.x / THREAD_DIV_FIX + blockIdx.x*blockDim.x / THREAD_DIV_FIX;
    const int th_id = threadIdx.x%THREAD_DIV_FIX;
    const int b_id = threadIdx.x / THREAD_DIV_FIX;
    __shared__ real4 mypos_sh[FIX_TH / THREAD_DIV_FIX];
    __shared__ int rci_sh[FIX_TH / THREAD_DIV_FIX];
    __shared__ size_t connsz_sh[FIX_TH / THREAD_DIV_FIX];
    __shared__ real4 out_dr[FIX_TH];
    if (index >= sz)return;
    if (th_id == 0) {
        memset(out_dr, 0x00, sizeof(out_dr));
        rci_sh[b_id] = FIX_filtered[index];
        connsz_sh[b_id] = all_nm_conn[rci_sh[b_id]].conn.size();
        mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, rci_sh[b_id]);
    }
    __syncthreads();
    const int raw_cell_index = rci_sh[b_id];
    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = connsz_sh[b_id];
    if (conn_sz <= th_id)return;


    const CellPos mypos = mypos_sh[b_id];
    //real z_dis = real(0.0);
    real4& dr = out_dr[threadIdx.x]; dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    /*
    if (collide_with_wall<DER>(mypos.z)) {
    z_dis += wall_interaction(mypos.z);
    }
    */
    //const int nm_index = raw_cell_index-nm_start;
    const CellAttr& myattr = nmcat[raw_cell_index - nm_start];




    for (int i = th_id; i < conn_sz; i += THREAD_DIV_FIX) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= nm_start) {
            const CELL_STATE st = nmcst[op_idx - nm_start];

            switch (st) {
            case FIX:case MUSUME:
                if (myattr.pair != op_idx) {
                    cell_movement_calc(&dr, mypos, op, c_fix_mu_to_fix_mu());
                }
                break;
            case ALIVE:case AIR:case DEAD:
                
                cell_movement_calc(&dr, mypos, op,
                    c_al_air_de_to_al_air_de_fix_mu(myattr.agek, nmcat[op_idx - nm_start].agek));
                
                break;
            default:
                
                cell_movement_calc(&dr, mypos, op, c_other_nm());
                
                break;
            }
        }
        else {
            
            if (myattr.dermis == op_idx) {
                cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_fix_to_memb());
            }
            else {
                cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_other_nm_memb());
                //cell_movement_calc(&dr, mypos, op, c_other_nm_memb());
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
    //dr.z += z_dis;
    if (th_id == 0) {
        out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, out_dr[0 + b_id*THREAD_DIV_FIX]));
    }
}
#define MUSUME_TH (64)
#define THREAD_DIV_MU (32)
__global__ void MUSUME_interaction(const cudaTextureObject_t pos_tex, const CellAttr* nmcat, const CELL_STATE* nmcst, const NonMembConn* all_nm_conn, const CellIndex* MUSUME_filtered, size_t nm_start, size_t sz, CellPos*out) {
    const int index = threadIdx.x/ THREAD_DIV_MU + blockIdx.x*blockDim.x/ THREAD_DIV_MU;
    const int th_id = threadIdx.x%THREAD_DIV_MU;
    const int b_id = threadIdx.x / THREAD_DIV_MU;
    __shared__ real4 mypos_sh[MUSUME_TH / THREAD_DIV_MU];
    __shared__ int rci_sh[MUSUME_TH / THREAD_DIV_MU];
    __shared__ size_t connsz_sh[MUSUME_TH / THREAD_DIV_MU];
    __shared__ real4 out_dr[MUSUME_TH];
    if (index >= sz)return;
    if (th_id == 0) {
        memset(out_dr, 0x00, sizeof(out_dr));
        rci_sh[b_id] = MUSUME_filtered[index];
        connsz_sh[b_id] = all_nm_conn[rci_sh[b_id]].conn.size();
        mypos_sh[b_id] = tex1Dfetch_real4(pos_tex, rci_sh[b_id]);
        //printf("idx:%d max:%d\n", index, (int)sz);
    }
    __syncthreads();
    const int raw_cell_index = rci_sh[b_id];
    const NonMembConn& nmc = all_nm_conn[raw_cell_index];
    const size_t conn_sz = connsz_sh[b_id];
    if (conn_sz <= th_id)return;
    const CellPos mypos = mypos_sh[b_id];
    //real z_dis = real(0.0);
    real4& dr = out_dr[threadIdx.x];dr= { real(0.0), real(0.0), real(0.0), real(0.0) };
    /*
    if (collide_with_wall<DER>(mypos.z)) {
    z_dis += wall_interaction(mypos.z);
    }
    */
    //const int nm_index = raw_cell_index-nm_start;
    const CellAttr& myattr = nmcat[raw_cell_index - nm_start];
    const bool paired_with_fix =
        myattr.pair >= 0 ?
        nmcst[myattr.pair - nm_start] == FIX ? true : false
        : false;
    const real spr = paired_with_fix ? Kspring :
        myattr.rest_div_times > 0 ? Kspring_d : 0;
        

    

    for (int i = th_id; i < conn_sz; i+= THREAD_DIV_MU) {

        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        const int op_idx = nmc.conn[i];
        if (op_idx >= nm_start) {
            const CELL_STATE st = nmcst[op_idx - nm_start];

            switch (st) {
            case FIX:case MUSUME:
                if (myattr.pair != op_idx) {
                    cell_movement_calc(&dr, mypos, op, c_fix_mu_to_fix_mu());
                }
                break;
            case ALIVE:case AIR:case DEAD:
                cell_movement_calc(&dr, mypos, op,
                    c_al_air_de_to_al_air_de_fix_mu(myattr.agek, nmcat[op_idx - nm_start].agek));
                break;
            default:
                cell_movement_calc(&dr, mypos, op, c_other_nm());
                break;
            }
        }
        else {
            
            if (myattr.dermis == op_idx) {
                cell_movement_calc_eachAdd(&dr,&out[op_idx], mypos, op, c_mu_to_memb(spr));
            }
            else {
                cell_movement_calc_eachAdd(&dr, &out[op_idx], mypos, op, c_other_nm_memb());
            }

        }


    }
    __syncthreads();
    //sum reduction
    for (int r = (THREAD_DIV_MU>>1); r >= 1; r >>= 1) {
        __syncthreads();
        if (th_id < r) {
            vadda(&out_dr[th_id+b_id*THREAD_DIV_MU], out_dr[th_id + r+ b_id*THREAD_DIV_MU]);
        }
    }
    //dr.z += z_dis;
    if (th_id == 0) {
        out[raw_cell_index] = vadd(mypos, cvmul(DT_Cell, out_dr[0 + b_id*THREAD_DIV_MU])); 
    }
}
struct c_pair {
    const real spr_nat_len;
    __device__ c_pair(real _slen):spr_nat_len(_slen){}
    __device__ real operator()(const real4 c1, const real4 c2)const {
        return -Kspring_division*(real(1.0) - spr_nat_len*rsqrt(p_dist_sq(c1, c2)));
    }
};

__global__ void pair_interaction(const cudaTextureObject_t pos_tex, const CellAttr* nmcat, const CELL_STATE* nmcst, const NonMembConn* all_nm_conn, const CellIndex* pair_filtered, size_t nm_start, size_t sz, CellPos*out) {
    const int index = threadIdx.x  + blockIdx.x*blockDim.x ;
    if (index >= sz)return;
    const int raw_cell_index = pair_filtered[index];
    real4 dr= { real(0.0), real(0.0), real(0.0), real(0.0) };
    const CellAttr& cat = nmcat[raw_cell_index-nm_start];
    const CellPos mypos= tex1Dfetch_real4(pos_tex, raw_cell_index);
    //printf("pairs:%d %d\n", raw_cell_index, (int)(cat.pair));
    const CellPos pairpos= tex1Dfetch_real4(pos_tex, cat.pair);
    cell_movement_calc(&dr, mypos, pairpos, c_pair(cat.spr_nat_len));
    vadda(&out[raw_cell_index], cvmul(DT_Cell, dr));
}
void calc_cell_movement(CellManager&cman) {
    size_t msz = cman.memb_size();
    size_t asz = cman.all_size();
    int flt_num;
    const CellIndex* flt = cman.nm_filter.filter_by_state<DER>(&flt_num);
        calc_memb_bend << <msz / 32 + 1, 32 >> > (cman.get_pos_tex(), cman.get_device_mconn(), msz, cman.get_device_pos_all_out());
        cudaDeviceSynchronize();
       
        MEMB_interaction << <msz / 64 + 1, 64 >> >(cman.get_pos_tex(), cman.get_device_non_memb_state(), cman.get_device_mconn(), cman.get_device_all_nm_conn(), msz, cman.get_device_pos_all_out());
        DER_interaction << <THREAD_DIV_DER*flt_num / DER_TH + 1, DER_TH >> >(cman.get_pos_tex(),cman.get_device_non_memb_state(),cman.get_device_all_nm_conn(),flt, msz,flt_num, cman.get_device_pos_all_out());
       
        cudaDeviceSynchronize();
        
        flt = cman.nm_filter.filter_by_state<ALIVE,AIR,DEAD>(&flt_num);

        AL_AIR_DE_interaction << <flt_num / 64 + 1, 64 >> >(cman.get_pos_tex(), 
            cman.get_device_nmattr(),
            cman.get_device_non_memb_state(), cman.get_device_all_nm_conn(), flt, msz, flt_num, cman.get_device_pos_all_out());
        
        cudaDeviceSynchronize();
        
        flt = cman.nm_filter.filter_by_state<FIX>(&flt_num);
        FIX_interaction << <THREAD_DIV_FIX*flt_num / FIX_TH + 1, FIX_TH >> >(cman.get_pos_tex(),
            cman.get_device_nmattr(),
            cman.get_device_non_memb_state(), cman.get_device_all_nm_conn(), flt, msz, flt_num, cman.get_device_pos_all_out());
        cudaDeviceSynchronize();
        
        flt = cman.nm_filter.filter_by_state<MUSUME>(&flt_num);

        MUSUME_interaction << <THREAD_DIV_MU* flt_num / MUSUME_TH + 1, MUSUME_TH >> >(cman.get_pos_tex(),
            cman.get_device_nmattr(),
            cman.get_device_non_memb_state(), cman.get_device_all_nm_conn(), flt, msz, flt_num, cman.get_device_pos_all_out());
        cudaDeviceSynchronize();
        
        //printf("possw\n");
        flt = cman.nm_filter.filter_by_pair(&flt_num);
       // printf("filtered pair:%d\n", flt_num);
        pair_interaction << <flt_num / 128 + 1, 128 >> >(cman.get_pos_tex(),
            cman.get_device_nmattr(),
            cman.get_device_non_memb_state(), cman.get_device_all_nm_conn(), flt, msz, flt_num, cman.get_device_pos_all_out());
        cudaDeviceSynchronize();
        
        cman.pos_swap_device();
        //printf("posswe\n");
        cman.refresh_pos_tex();
        
}