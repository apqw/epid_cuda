#include "define.h"
#include "CubicDynArr.h"
#include "device_launch_parameters.h"
#include "CellManager.h"
#include "cuda_helper_misc.h"
#include "calc_ext_stim.h"
__device__ static inline real fB(real age, real B, bool cornif) {
    

    //using namespace cont;
    return (cornif&&age > THRESH_DEAD - DUR_ALIVE&&age <= THRESH_DEAD + DUR_DEAD ? real(1.0) : real(0.0)) - kb*B;
}
/*
__global__ void _calc_ext_stim(const cudaSurfaceObject_t ext_stim, 
    const cudaTextureObject_t cmap1,
    const cudaTextureObject_t cmap2,
    const CellAttr* cat,
    const CellIterateRange_device cir,
    const real* zzmax, 
    cudaSurfaceObject_t out_stim)
{
    const int iz_bound = (int)((*zzmax + FAC_MAP*NON_MEMB_RAD) *inv_dz);
    const int z = blockIdx.y;
    if (z >= iz_bound)return;
    const int x = threadIdx.x;
    if (x >= NX)return;
    const int y = blockIdx.x;
    const int prev_x = (x - 1 + NX) % NX; const int next_x = (x + 1) % NX;
    const int prev_y = (y - 1 + NY) % NY; const int next_y = (y + 1) % NY;
    const int prev_z = z == 0 ? 1 : z - 1; const int next_z = z == NZ-1 ? NZ-2 : z + 1;
#define midx(xx,yy,zz) (xx+NX*(yy+NY*zz))
    const int cidx = tex1Dfetch<int>(cmap1,midx(x,y,z));
    const bool flg_cornified = cir.nums[CS_dead_hd] <= cidx&&cidx < cir.nums[CS_musume_hd];
    const real dum_age = flg_cornified ? cat[cidx].agek:0.0;

    const real myv = surf3Dread_real(ext_stim, x, y, z);
    const real outv = myv +
        DT_Ca*(DB*(
            tex1Dfetch<int>(cmap2, midx(prev_x, y, z))*(surf3Dread_real(ext_stim, prev_x, y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(next_x, y, z))*(surf3Dread_real(ext_stim, next_x, y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, prev_y, z))*(surf3Dread_real(ext_stim, x, prev_y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, next_y, z))*(surf3Dread_real(ext_stim, x, next_y, z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, y, prev_z))*(surf3Dread_real(ext_stim, x, y, prev_z) - myv)
            + tex1Dfetch<int>(cmap2, midx(x, y, next_z))*(surf3Dread_real(ext_stim, x, y, next_z) - myv)
            )*inv_dz*inv_dz
            + fB(dum_age, myv, flg_cornified));
   surf3Dwrite_real(outv, out_stim, x, y, z);
}
*/
#define CDIM (4)
__global__ void _calc_ext_stim_3d(const cudaSurfaceObject_t ext_stim,
    const cudaTextureObject_t cmap1,
    const cudaTextureObject_t cmap2,
    const CellAttr* cat,
    const CellIterateRange_device cir,
    const real* zzmax,
    cudaSurfaceObject_t out_stim)
{
    const int iz_bound = (int)((*zzmax + FAC_MAP*NON_MEMB_RAD) *inv_dz);

    const int z = threadIdx.z+blockIdx.z*CDIM;
    if (z >= iz_bound)return;
    const int x = threadIdx.x+blockIdx.x*CDIM;
    if (x >= NX)return;
    const int y = threadIdx.y+blockIdx.y*CDIM;
    if (y >= NY)return;
    const int prev_x = x==0?NX-1:x-1; const int next_x = x==NX-1?0:x+1;
    const int prev_y = y==0?NY-1:y-1; const int next_y = y==NY-1?0:y+1;
    const int prev_z = z == 0 ? 1 : z - 1; const int next_z = z == NZ - 1 ? NZ - 2 : z + 1;
#define midx(xx,yy,zz) (xx+NX*(yy+NY*zz))
    const int cidx = tex1Dfetch<int>(cmap1, midx(x, y, z));
    //if(cidx!=-1&&z==NZ/4)printf("cidx ext:%d\n",cidx);
    const bool flg_cornified = cir.nums[CS_dead_hd] <= cidx&&cidx < cir.nums[CS_musume_hd];
    const real dum_age = flg_cornified ? cat[cidx].agek : 0.0;

    const real myv = surf3Dread_real(ext_stim, x, y, z);
   // if(x==NX/2&&y==NY/2&&z==NZ/4)printf("myv ext:%f\n",myv);
    const real outv = myv +
        DT_Ca*(DB*(
            intmask_real(tex1Dfetch<int>(cmap2, midx(prev_x, y, z)),(surf3Dread_real(ext_stim, prev_x, y, z) - myv))
            + intmask_real(tex1Dfetch<int>(cmap2, midx(next_x, y, z)),(surf3Dread_real(ext_stim, next_x, y, z) - myv))
            + intmask_real(tex1Dfetch<int>(cmap2, midx(x, prev_y, z)),(surf3Dread_real(ext_stim, x, prev_y, z) - myv))
            + intmask_real(tex1Dfetch<int>(cmap2, midx(x, next_y, z)),(surf3Dread_real(ext_stim, x, next_y, z) - myv))
            + intmask_real(tex1Dfetch<int>(cmap2, midx(x, y, prev_z)),(surf3Dread_real(ext_stim, x, y, prev_z) - myv))
            + intmask_real(tex1Dfetch<int>(cmap2, midx(x, y, next_z)),(surf3Dread_real(ext_stim, x, y, next_z) - myv))
            )*inv_dz*inv_dz
            + fB(dum_age, myv, flg_cornified));
    surf3Dwrite_real(outv, out_stim, x, y, z);
    //if(x==NX/2&&y==NY/2&&z==NZ/4)printf("outv ext:%f %d %d %dtest:%f test2:%f\n",outv,prev_x,y,z,fB(dum_age, myv, flg_cornified),surf3Dread_real(ext_stim, prev_x, y, z));
}

void calc_ext_stim(CellManager&cm, cudaSurfaceObject_t* ext_stim,
    const CubicDynArrTexReader<int> cmap1,
    const CubicDynArrTexReader<CMask_t> cmap2, const real* zzmax, cudaSurfaceObject_t* out_stim) {
    _calc_ext_stim_3d<<<dim3(NX/ CDIM +1,NY/ CDIM +1,NZ/ CDIM +1),dim3(CDIM, CDIM, CDIM)>>>(*ext_stim, cmap1.ct, cmap2.ct, cm.get_device_attr(), cm.get_cell_iterate_range_d(), zzmax, *out_stim);
    wrap_bound(*out_stim);
    std::swap(*ext_stim, *out_stim);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}
