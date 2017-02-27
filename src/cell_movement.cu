#include "define.h"
#include "cell_movement.h"
#include "math_helper.h"
#include "CellManager.h"

#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"

template<class Fn>
__device__ void cell_movement_calc(real4* accum_out, const real4& c1, const real4& c2, Fn calc_fn) {
    vadda(accum_out, cvmul(calc_fn(c1, c2), vpsub(c1, c2)));
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


__global__ void MEMB_interaction(cudaTextureObject_t pos_tex,const MembConn* mconn,size_t sz, CellPos* out) {
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
    dr.z += z_dis;

    vadda(&out[index] , cvmul(DT_Cell, dr));

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

    const real dist_sq = sqrt(p_dist_sq(d1, d2));
   // const double rad_sum = c1->radius + c2->radius;
    const real LJ6 = POW6(NON_MEMB_RAD_SUM) / POW3(dist_sq);
    //LJ6 = LJ6*LJ6*LJ6;
    return real(4.0)*eps_m*LJ6*(LJ6 - real(1.0)) / dist_sq;

}

__device__ inline real ljmain_nm_memb(const real4 d1, const real4 d2) {

    const real dist_sq = sqrt(p_dist_sq(d1, d2));
    // const double rad_sum = c1->radius + c2->radius;
    const real LJ6 = POW6(NON_MEMB_RAD+MEMB_RAD) / POW3(dist_sq);
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
__global__ void DER_interaction(const cudaTextureObject_t pos_tex,const CELL_STATE* nmcst, const NonMembConn* nmconn,const CellIndex* DER_filtered, size_t nm_start, size_t sz, CellPos*out) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= sz)return;
    const int raw_cell_index = DER_filtered[index];
    const CellPos mypos = tex1Dfetch_real4(pos_tex, raw_cell_index);
    real z_dis =real(0.0);
    real4 dr = { real(0.0), real(0.0), real(0.0), real(0.0) };
    if (collide_with_wall<DER>(mypos.z)) {
        z_dis += wall_interaction(mypos.z);
    }
    const int nm_index = raw_cell_index-nm_start;
    const NonMembConn& nmc = nmconn[nm_index];
    const size_t conn_sz = nmc.conn.size();
    for (int i = 0; i < conn_sz; i++) {
        const real4 op = tex1Dfetch_real4(pos_tex, nmc.conn[i]);
        if (nmc.conn[i] >= nm_start) {
            if (nmcst[nmc.conn[i] - nm_start] == DER) {
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

    dr.z += z_dis;
    out[raw_cell_index]=vadd(mypos, cvmul(DT_Cell, dr));
}

void calc_cell_movement(CellManager&cman) {
    size_t msz = cman.memb_size();
    size_t asz = cman.all_size();
    int flt_num;
    const CellIndex* flt = cman.nm_filter.filter_by_state<DER>(&flt_num);
        calc_memb_bend << <msz / 32 + 1, 32 >> > (cman.get_pos_tex(), cman.get_device_mconn(), msz, cman.get_device_pos_all_out());
        cudaDeviceSynchronize();
        MEMB_interaction << <msz / 64 + 1, 64 >> >(cman.get_pos_tex(), cman.get_device_mconn(), msz, cman.get_device_pos_all_out());
        DER_interaction << <flt_num / 32 + 1, 32 >> >(cman.get_pos_tex(),cman.get_device_non_memb_state(),cman.get_device_nmconn(),flt, msz,flt_num, cman.get_device_pos_all_out());
        cman.pos_swap_device();
        cman.refresh_pos_tex();
}