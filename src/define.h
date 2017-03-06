#pragma once

#include <vector_types.h>
#include <cfloat>
using CELL_STATE_internal = unsigned short;

#define USE_DOUBLE_AS_REAL
static constexpr double M_PId = double(3.141592653589793238463);
static constexpr float M_PIf = float(3.14159265358979f);
#ifdef USE_DOUBLE_AS_REAL
typedef double4 real4;
typedef double3 real3;
typedef double2 real2;
typedef double1 real1;
typedef double real;
#define R4FMT "%lf"
#define REAL_MAX (DBL_MAX)
#define sqrtr(v) sqrt((double)(v))
#define M_PI M_PId
#else
typedef float real;
typedef float4 real4;
typedef float3 real3;
typedef float2 real2;
typedef float1 real1;
#define R4FMT "%f"
#define REAL_MAX (FLT_MAX)
#define sqrtr(v) sqrtf(v)
#define M_PI M_PIf
#endif
using CellPos = real4;
enum CELL_STATE :CELL_STATE_internal {
    ALIVE = 0,
    DEAD = 1,
    DISA = 2,
    UNUSED = 3, //
    FIX = 4,
    BLANK = 5,
    DER = 6,
    MUSUME = 7,
    AIR = 8,
    MEMB = 9,
    DUMMY_FIX = 10
};

using CellIndex = int;
using CMask_t = int;
#define MEMB_NUM_X (100)
#define MEMB_NUM_Y (114)
#define CELL_CONN_NUM (200)
static constexpr real  LX = real(50.0);
static constexpr real  LY = real(50.0);
static constexpr real  LZ = real(100.0);
static constexpr unsigned int NX = 100;
static constexpr unsigned int NY = 100;
static constexpr unsigned int NZ = 200;
static constexpr real  dx = LX/NX; static constexpr real  inv_dx = NX/LX;
static constexpr real  dy = LY / NY; static constexpr real  inv_dy = NY / LY;
static constexpr real  dz = LZ / NZ; static constexpr real  inv_dz = NZ / LZ;
static constexpr real FAC_MAP = 2.0;
#define LJ_THRESH real(1.2)
#define MEMB_RAD real(1.0)
#define NON_MEMB_RAD real(1.4)
static constexpr real MEMB_RAD_SUM = (MEMB_RAD + MEMB_RAD);
static constexpr real MEMB_RAD_SUM_SQ = (MEMB_RAD_SUM*MEMB_RAD_SUM);
static constexpr real NON_MEMB_RAD_SUM = (NON_MEMB_RAD + NON_MEMB_RAD);
static constexpr real delta_R = real(0.4)*NON_MEMB_RAD;
static constexpr real para_ljp2 = 0.005;
static constexpr real K_TOTAL = 3.0;
static constexpr real K_DESMOSOME_RATIO = 0.01;
static constexpr real K_DESMOSOME = K_TOTAL*K_DESMOSOME_RATIO;
static constexpr real  THRESH_SP = 3.0;
static constexpr real Kspring = 25.0;
static constexpr real Kspring_d = 5.0;
static constexpr real Kspring_division = 5.0;

static constexpr real accel_div = 1.0;
static constexpr real eps_kb = 0.12;
static constexpr real alpha_b = 5.0;
static constexpr real S2 = 0.1;
static constexpr real ca2p_init = 0.122;
static constexpr real S0 = 0.1*0.2;
static constexpr real eps_ks = 0.10*0.5;
static constexpr real accel_diff = 1.0;
static constexpr real alpha_k = 2.0;
static constexpr real eps_kk = 0.10*0.5;
static constexpr real S1 = 0.1;
static constexpr real lipid_rel = 0.05*4.0;

static constexpr real ubar = 0.45;


static constexpr real delta_lipid = 0.1;
static constexpr real delta_lipid_inv = 1.0/delta_lipid;
static constexpr real delta_sig_r1 = 0.1;
static constexpr real delta_sig_r1_inv = 1.0/delta_sig_r1;
static constexpr real lipid = 0.6;

static constexpr bool STOCHASTIC = true;
static constexpr real stoch_div_time_ratio = 0.25;
static constexpr real delta_L = 0.01*NON_MEMB_RAD;
static constexpr real THRESH_DEAD = 22.0;
static constexpr real ADHE_CONST = 31.3;
static constexpr int DISA_conn_num_thresh = 11; //Nc

                                                /** MUSUME用分裂開始年齢のしきい値*/
static constexpr real agki_max = 6.0;
static constexpr real eps_L = 0.14;//ok
static constexpr real unpair_dist_coef = 0.9;

/** FIX用分裂開始年齢の倍率 */
static constexpr real fac = 1;

/** FIX用分裂開始年齢のしきい値 */
static constexpr real agki_max_fix = fac*agki_max;

static constexpr real kb = 0.025;
static constexpr real DUR_ALIVE = 0.5;
static constexpr real DUR_DEAD = 2.0;
static constexpr real kbc = 0.4*1.2;
static constexpr real Hb = 0.01;
static constexpr real Cout = 1.0;
static constexpr real kg = 0.1;
static constexpr real gamma = 2.0;
static constexpr real mu0 = 0.567;
static constexpr real mu1 = 0.1;
static constexpr real kmu = 0.05;
static constexpr real para_b = 0.11;
static constexpr real para_bb = 0.89;
static constexpr real para_k1 = 0.7;
static constexpr real k_flux = 8.1;

static constexpr real beta_zero = 0.02;
static constexpr real CA_OUT = 1.0;
static constexpr real leak_from_storage = CA_OUT*beta_zero;
static constexpr real thgra = 0.2;
static constexpr real delta_th = 1.0;
static constexpr real thpri = 1.0;
static constexpr real kpa = 4.0;
static constexpr real Kgra = kpa;
static constexpr real delta_K = 1.0;
static constexpr real Kpri = 6.0;
static constexpr real delta_I = 1.5;
static constexpr real iage_kitei = 0;//ok
static constexpr real para_k2 = 0.7;
static constexpr real H0 = 0.5;
static constexpr real wd = 0.1;//ok
static constexpr real epsw0 = 0.1;
static constexpr real Ca_avg_time = 10.0;
static constexpr real DT_Ca = 0.01;
static constexpr unsigned Ca_ITR = (int)(Ca_avg_time / DT_Ca);
static constexpr real ca2p_du = 0.01;
static constexpr real Kpp = 0.3;
static constexpr real dp = 0.1;
#define DT_Cell real(0.01)

static constexpr real DB = 0.0009;
static constexpr real Da = 1.0;
static constexpr real AIR_STIM = 0.1;
static constexpr real STIM11 = 0.002;//ok
static constexpr real Kaa = 0.5;//ok
#define KBEND real(0.5*20.0)
#define eps_m real(0.01)
static constexpr real COMPRESS_FACTOR = 4.0;
static constexpr real P_MEMB = 1.0/COMPRESS_FACTOR;
static constexpr real DER_DER_CONST = 0.08;//harden original 0.2
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)

#define WARP_SIZE (32)
#ifdef NDEBUG
#define DBG_ONLY(s) do{}while(0)
#define dbgprintf(x,...) do{}while(0)
#else
#define DBG_ONLY(s) do{s;}while(0)
#define dbgprintf(x,...) printf((x),__VA_ARGS__)
#endif