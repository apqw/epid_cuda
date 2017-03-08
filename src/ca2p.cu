#include "CellManager.h"
#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"
__device__ inline real ca2p_by_ext_stim(real _ext_stim) {
    

    return kbc * _ext_stim*_ext_stim * Cout / (Hb + _ext_stim*_ext_stim);
}

__device__ inline real ca2p_into_storage(real ca2p_in_cell) {
    
    return v_gamma * ca2p_in_cell / (kg + ca2p_in_cell);
}

__device__ inline real ER_domain1_active(real ip3) {

    return (mu0 + mu1 * ip3 / (ip3 + kmu));
}

__device__ inline real ER_domain2_active(real ca2p) {

    return (para_b + para_bb * ca2p / (para_k1 + ca2p));
}

__device__ inline real ca2p_reaction_factor(real u, real ER_domain3_deactive, real p, real B)
{
    

    const real ca2p_from_storage = k_flux * ER_domain1_active(p) * ER_domain2_active(u) * ER_domain3_deactive;

    return
        ca2p_from_storage
        - ca2p_into_storage(u)
        + leak_from_storage
        + ca2p_by_ext_stim(B);

}

__device__ inline real calc_th(bool is_alive, real agek) {

    
    //using namespace cont;


    return is_alive ?
        thgra + ((thpri - thgra)*real(0.5)) * (real(1.0) + tanh((THRESH_SP - agek) / delta_th)) :
        thpri;
}

__device__ inline real calc_Kpa(bool is_alive, real agek) {

    
    return is_alive ?
        Kgra + ((Kpri - Kgra)*real(0.5)) * (real(1.0) + tanh((THRESH_SP - agek) / delta_K)) :
        Kpri;
}

__device__ inline real calc_IAG(bool is_alive, real agek) {
    
    return is_alive ?
        real(0.5)*(real(1.0) + tanh((agek - THRESH_SP) / delta_I)) :
        iage_kitei;
}
__device__ inline real ex_inert_diff(real ca2p, real current_ex_inert, real _th) {
    
    return ((para_k2*para_k2 / (para_k2*para_k2 + ca2p*ca2p) - current_ex_inert) / _th);
}

__device__ inline real IP3_default_diff(real _Kpa, real a_avg, real current_IP3) {
    

    return (_Kpa*a_avg / (H0 + a_avg) - Kpp*current_IP3);
}

__device__ inline real fw(real diff, real w)
{

    return 0.5 - w + 0.5*tanh((wd - diff) / epsw0); //epsw0 == 0.1 //done
                                                            //-1. <-????
}

__global__ void dead_IP3_calc(const real* IP3_in,NonMembConn*nmconn,CellIterateRange_device cir,real* IP3_out) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_DEAD>()) return;
    const int raw_idx = cir.idx<CI_DEAD>(index);
    const int al_head = cir.nums[CS_alive_hd];
    const int al_end = cir.nums[CS_musume_hd];
    NonMembConn& nmc = nmconn[raw_idx];
    const size_t sz = nmc.conn.size();
    real tmp = 0.0;
    const real myip3 = IP3_in[raw_idx];
    for (int i = 0; i < sz; i++) {
        const int opidx = nmc.conn[i];
        if (al_head <= opidx&&opidx < al_end) {
            tmp += IP3_in[opidx] - myip3;
        }
    }
    IP3_out[raw_idx]=myip3+ DT_Ca*(dp*tmp - Kpp*myip3);
}

struct gj_data {
    real data[CELL_CONN_NUM];
    __device__ real& operator[](int idx) {
        return data[idx];
    }
    __device__ const real& operator[](int idx)const {
        return data[idx];
    }
};

__global__ void supra_calc(const CellPos*cpos, 
    const real* ca2p_in,real* ca2p_out,
    real* ca2p_avg_out,
    real*diff_u_out, gj_data* gj, 
    real* exinert,
    const real* IP3_in,real* IP3_out,
    NonMembConn*nmconn, const real* agekset,const cudaSurfaceObject_t ATP_first, const cudaSurfaceObject_t ext_stim_first, CellIterateRange_device cir) {
    const int index = threadIdx.x + blockIdx.x*blockDim.x;
    if (index >= cir.size<CI_ALIVE,CI_MUSUME,CI_FIX>()) return;
    const int raw_idx = cir.idx<CI_ALIVE, CI_MUSUME, CI_FIX>(index);
    const real4 mypos = cpos[raw_idx];
    const int ix = int(mypos.x*inv_dx);
    const int iy = int(mypos.y*inv_dy);
    const int iz = int(mypos.z*inv_dz);
    const real myagek = agekset[raw_idx];
    const bool is_alive = cir.nums[CS_alive_hd] <= raw_idx&&raw_idx < cir.nums[CS_musume_hd];
    const real _th = calc_th(is_alive,myagek );
    const real _Kpa = calc_Kpa(is_alive, myagek);
    const real IAGv = calc_IAG(is_alive, myagek);
    NonMembConn& nmc = nmconn[raw_idx];
    const size_t sz = nmc.conn.size();
    real tmp_diffu = 0.0;
    real tmp_IP3 = 0.0;
    //real tmp_gj = 0.0;
    const real myca2p = ca2p_in[raw_idx];
    gj_data& mygj = gj[raw_idx];

    for (int i = 0; i < sz; i++) {
        const int opidx = nmc.conn[i];
        //st==ALIVE||st==DEAD||st==FIX||st==MUSUME
        const bool criteria_all = cir.nums[CS_fix_hd] <= opidx && !(cir.nums[CS_der_hd] <= opidx&&opidx < cir.nums[CS_dead_hd]);
        const bool criteria_alive = cir.nums[CS_alive_hd] <= opidx && opidx<cir.nums[CS_musume_hd];
        if (criteria_all) {
            tmp_diffu += mygj[opidx] * (ca2p_in[opidx] - myca2p);
            tmp_IP3 += mygj[opidx] * (ca2p_in[opidx] - myca2p);
            if (criteria_alive) {
                mygj[opidx] +=DT_Ca* fw(fabs(ca2p_in[opidx] - myca2p), mygj[opidx]);
            }
        }
    }

   const real duo= diff_u_out[raw_idx]= ca2p_reaction_factor(myca2p, exinert[raw_idx], IP3_in[raw_idx],
       grid_avg8_sobj(ext_stim_first, ix, iy, iz)) + ca2p_du*IAGv*tmp_diffu;
   IP3_out[raw_idx] =
       IP3_in[raw_idx] + DT_Ca*(IP3_default_diff(_Kpa, grid_avg8_sobj(ATP_first, ix, iy, iz), IP3_in[raw_idx]) + dp*IAGv*tmp_IP3);
  
   const real cs = myca2p + DT_Ca*duo;
   ca2p_out[raw_idx] = cs;
   ca2p_avg_out[raw_idx] += cs;
   exinert[raw_idx]+=DT_Ca*ex_inert_diff(myca2p, exinert[raw_idx], _th);

   // c->IP3.set_next(c->IP3() + DT_Ca*(IP3_default_diff(_Kpa, grid_avg8(ATP_first(), ix, iy, iz), c->IP3()) + dp*IAGv*tmp_IP3));

}
__device__ inline real fa(real diffu, real A) {
    
    //using namespace cont;
    return STIM11*min0(diffu) - A*Kaa;
}
#define CDIM (4)
__global__ void ATP_refresh(const cudaTextureObject_t cmap1,
    const cudaTextureObject_t cmap2,
    const real* diffu_map,
    const cudaSurfaceObject_t ATP_in, cudaSurfaceObject_t ATP_out,
    const real*zzmax) {
    const int iz_bound = (int)((*zzmax + FAC_MAP*NON_MEMB_RAD) *inv_dz);

    const int z = threadIdx.z + blockIdx.z*CDIM;
    if (z >= iz_bound)return;
    const int x = threadIdx.x + blockIdx.x*CDIM;
    if (x >= NX)return;
    const int y = threadIdx.y + blockIdx.y*CDIM;
    if (y >= NY)return;

    const int prev_x = x == 0 ? NX - 1 : x - 1; const int next_x = x == NX - 1 ? 0 : x + 1;
    const int prev_y = y == 0 ? NY - 1 : y - 1; const int next_y = y == NY - 1 ? 0 : y + 1;
    const int prev_z = z == 0 ? 1 : z - 1; const int next_z = z == NZ - 1 ? NZ - 2 : z + 1;
#define midx(xx,yy,zz) (xx+NX*(yy+NY*zz))
    const int cidx = tex1Dfetch<int>(cmap1, midx(x, y, z));

    const real diffu = cidx < 0 ? 0.0 : diffu_map[cidx];
    const real asf = cidx < 0 ? 0.0 : AIR_STIM;
    const real myv = surf3Dread_real(ATP_in, x, y, z);
    const real outv = myv +
        DT_Ca*(Da*(
            tex1Dfetch<int>(cmap2, midx(prev_x, y, z))*(surf3Dread_real(ATP_in, prev_x, y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(next_x, y, z))*(surf3Dread_real(ATP_in, next_x, y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, prev_y, z))*(surf3Dread_real(ATP_in, x, prev_y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, next_y, z))*(surf3Dread_real(ATP_in, x, next_y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, y, prev_z))*(surf3Dread_real(ATP_in, x, y, prev_z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, y, next_z))*(surf3Dread_real(ATP_in, x, y, next_z) - myv)
            )*inv_dx*inv_dx
            + fa(diffu, myv)+asf);
    surf3Dwrite_real(outv, ATP_out, x, y, z);
}
__global__ void test22() {
    printf("LUL\n");
}
__global__ void initialize_ca2p_calc(CellIterateRange_device cir,CellAttr*cat,
		gj_data*gj,real* ca2p1,real* ca2p2,real* ca2p_avg,real* diffu,real*exinert,real*IP3_1,real*IP3_2,real*agek) {
const int index=threadIdx.x+blockIdx.x*blockDim.x;
if(index>=cir.nums[CS_asz])return;
ca2p1[index]=ca2p_init;
ca2p2[index]=ca2p_init;
ca2p_avg[index]=real(0.0);
diffu[index]=real(0.0);
exinert[index]=cat[index].ex_inert;
IP3_1[index]=cat[index].IP3;
IP3_2[index]=cat[index].IP3;
agek[index]=cat[index].agek;
for(int i=0;i<CELL_CONN_NUM;i++){
	gj[index][i]=gj_init;
}
}
__global__ void initialize_ATP(cudaSurfaceObject_t ATP_1,cudaSurfaceObject_t ATP_2) {
surf3Dwrite_real(real(0.0),ATP_1,threadIdx.x,blockIdx.x,blockIdx.y);
surf3Dwrite_real(real(0.0),ATP_2,threadIdx.x,blockIdx.x,blockIdx.y);
}
template<typename T>
__device__ void swap_POD(T& s1,T& s2){
	T tmp=s1;
	s1=s2;
	s2=tmp;
}
__global__ void finalize_ca2p(CellIterateRange_device cir,CellAttr*cat,real* ca2p_avg,real*exinert,real*IP3){
	const int index=threadIdx.x+blockIdx.x*blockDim.x;
	if(index>=cir.nums[CS_asz])return;
	cat[index].ca2p_avg=ca2p_avg[index]/Ca_ITR;
	cat[index].ex_inert=exinert[index];
	cat[index].IP3=IP3[index];
}
__global__ void ca2p_proc_parent(
		CellAttr*cat,NonMembConn*nmc,CellPos*cpos,cudaSurfaceObject_t extstim,cudaTextureObject_t cmap1,cudaTextureObject_t cmap2,
		const real*zzmax,
		cudaSurfaceObject_t ATP_1,cudaSurfaceObject_t ATP_2,
		CellIterateRange_device cir,gj_data*gj,
		real* ca2p1,real* ca2p2,real* ca2p_avg,real* diffu,real*exinert,real*IP3_1,real*IP3_2,real*agek){
	if(!(threadIdx.x==0&&blockIdx.x==0))return;
	if(cir.nums[CS_count_sw]>=SW_THRESH){
		printf("test sw\ntest sw\ntest sw\ntest sw\ntest sw\n");
		initialize_ca2p_calc<<<cir.nums[CS_asz]/256+1,256>>>
				(cir,cat,gj,ca2p1,ca2p2,ca2p_avg,diffu,exinert,IP3_1,IP3_2,agek);
		initialize_ATP<<<dim3(NY,NZ),NX>>>(ATP_1,ATP_2);

		for(unsigned int i=0;i<Ca_ITR;i++){
		dead_IP3_calc<<<cir.size<CI_DEAD>()/64+1,64>>>(IP3_1,nmc,cir,IP3_2);

		supra_calc<<<cir.size<CI_ALIVE,CI_MUSUME,CI_FIX>()/64+1,64>>>(cpos,ca2p1,ca2p2,ca2p_avg,diffu,gj,exinert,IP3_1,IP3_2,nmc,agek,ATP_1,extstim,cir);
		ATP_refresh<<<dim3(NX/CDIM+1,NY/CDIM+1,NZ/CDIM+1),dim3(CDIM,CDIM,CDIM)>>>(cmap1,cmap2,diffu,ATP_1,ATP_2,zzmax);

		_wrap_bound<<<dim3(NY, NZ), NX >>>(ATP_2);
		swap_POD(ATP_1,ATP_2);
		swap_POD(ca2p1,ca2p2);
		swap_POD(IP3_1,IP3_2);
		}
		finalize_ca2p<<<cir.nums[CS_asz]/256+1,256>>>
						(cir,cat,ca2p_avg,exinert,IP3_1);
		cir.nums[CS_count_sw]=0;
	}
}

void calc_ca2p(CellManager&cm,const cudaSurfaceObject_t extstim,cudaTextureObject_t cmap1,cudaTextureObject_t cmap2) {
	const size_t ubs=2*cm.all_size();
    static thrust::device_vector<real> ca2p1(ubs), ca2p2(ubs),
ca2p_avg(ubs), diffu(ubs), exinert(ubs), IP3_1(ubs), IP3_2(ubs),agek(ubs);
    static thrust::device_vector<gj_data>gj(ubs);
    static cuda3DSurface<real> ATP_1(NX, NY, NZ), ATP_2(NX, NY, NZ);
#define _dptr(v) thrust::raw_pointer_cast(v.data())

    ca2p_proc_parent<<<1,32>>>(cm.get_device_attr(),cm.get_device_all_nm_conn(),cm.get_device_pos_all(),extstim,cmap1,cmap2,cm.zzmax_ptr(),
    		ATP_1.st,ATP_2.st,cm.get_cell_iterate_range_d(),
    		_dptr(gj),
    		_dptr(ca2p1),_dptr(ca2p2),_dptr(ca2p_avg),_dptr(diffu),_dptr(exinert),_dptr(IP3_1),_dptr(IP3_2),_dptr(agek));
}
