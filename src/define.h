#pragma once

#include <vector_types.h>
#include <cfloat>
using CELL_STATE_internal = unsigned short;

#define USE_DOUBLE_AS_REAL

#ifdef USE_DOUBLE_AS_REAL
typedef double4 real4;
typedef double3 real3;
typedef double real;
#define R4FMT "%lf"
#define REAL_MAX (DBL_MAX)
#define sqrtr(v) sqrt((double)(v))
#else
typedef float real;
typedef float4 real4;
typedef float3 real3;

#define R4FMT "%f"
#define REAL_MAX (FLT_MAX)
#define sqrtr(v) sqrtf(v)
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
static constexpr double Kspring_division = 5.0;
#define DT_Cell real(0.01)
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
