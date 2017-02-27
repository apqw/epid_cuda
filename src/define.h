#pragma once

#include <vector_types.h>
using CELL_STATE_internal = unsigned short;

#define USE_DOUBLE_AS_REAL

#ifdef USE_DOUBLE_AS_REAL
typedef double4 real4;
typedef double3 real3;
typedef double real;
#define R4FMT "%lf"
#else
typedef float real;
typedef float4 real4;
typedef float3 real3;
#define R4FMT "%f"
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
#define MEMB_NUM_X (100)
#define MEMB_NUM_Y (114)
#define LX real(50.0)
#define LY real(50.0)
#define LZ real(100.0)
#define LJ_THRESH real(1.2)
#define MEMB_RAD real(1.0)
#define NON_MEMB_RAD real(1.4)
#define CUDA_SAFE_CALL(func) \
do { \
     cudaError_t err = (func); \
     if (err != cudaSuccess) { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
     } \
} while(0)