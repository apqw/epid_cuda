#include "define.h"
#include "CubicDynArr.h"
#include "LFStack.h"
#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"
#include "math_helper.h"
#include "cell_connection.h"
#include "CellManager.h"

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
static constexpr int	N2 = 400; //max conn num //ok

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

__global__ void connect_proc(cudaTextureObject_t pos_tex,NonMembConn* nmconn, CubicDynArrAccessor<LFStack<int, N3>> darr, int nmemb_start,size_t sz) {
    //need memset
    const int index = nmemb_start+blockIdx.x * blockDim.x + threadIdx.x;
    static constexpr int srange = 2;
    if (index < sz) {
        const CellPos cp = tex1Dfetch_real4(pos_tex, index);
        const int anx = (int)(cp.x * AREA_GRID_INV);
        const int any = (int)(cp.y* AREA_GRID_INV);
        const int anz = (int)(cp.z* AREA_GRID_INV);
        
        assert(!(anx >= (int)ANX || any >= (int)ANY || anz >= (int)ANZ || anx < 0 || any < 0 || anz < 0));
        
        const int xstart = anx - srange;
        const int xend = anx + srange;
        const int ystart = any - srange;
        const int yend = any + srange;
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
                        const CellPos ocp = tex1Dfetch_real4(pos_tex, oi);
                        const real rad_sum = oi >= nmemb_start ? NON_MEMB_RAD + NON_MEMB_RAD : MEMB_RAD + NON_MEMB_RAD;
                        if (p_dist_sq(cp, ocp) <= LJ_THRESH*LJ_THRESH*rad_sum*rad_sum) {
                            nmconn[index-nmemb_start].conn.push_back_d(oi);
                           // printf("eeey %d <-> %d\n", index, oi);
                        }
                    }
                }
            }
        }
    }
}

void connect_cell(CellManager & cman)
{
    static CubicDynArrGenerator<LFStack<int, N3>> area(ANX, ANY, ANZ);
    area.memset_zero();

    size_t sz = cman.all_size();

    grid_init<<<sz/64+1,64>>>(cman.get_pos_tex(), area.acc, sz);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
    connect_proc << <sz / 128 + 1, 128 >> >(cman.get_pos_tex(),cman.get_device_nmconn(), area.acc,cman.memb_size(), sz);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //CUDA_SAFE_CALL(cudaGetLastError());
}