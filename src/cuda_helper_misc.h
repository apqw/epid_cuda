#pragma once

#include "define.h"
#include "math_helper.h"
#include <driver_functions.h>
//#include <helper_math.h>
#include <channel_descriptor.h>
#include <surface_indirect_functions.h>
cudaResourceDesc make_real4_resource_desc(CellPos*r4ptr, size_t len);
void wrap_bound(cudaTextureObject_t data);
__global__ void _wrap_bound(cudaTextureObject_t data);
__device__ inline float intmask_real(int mask, float v) {
    return __int_as_float(__float_as_int(v)&mask);
}
__device__ inline double intmask_real(int mask, double v) {
    return __hiloint2double(__double2hiint(v)&mask, __double2loint(v)&mask);
}
__device__ inline double __atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ inline float __atomicAdd(float* address, float val)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(val + __int_as_float(assumed)));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ inline double __atomicSub(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val - __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ inline void atomicvadda(double4* dest, const double4 v) {
    __atomicAdd(&dest->x, v.x);
    __atomicAdd(&dest->y, v.y);
    __atomicAdd(&dest->z, v.z);
    //atomicSub(&dest.w, v.w);
}
__device__ inline void atomicvsuba(double4* dest, const double4 v) {
    __atomicAdd(&dest->x, -v.x);
    __atomicAdd(&dest->y, -v.y);
    __atomicAdd(&dest->z, -v.z);
    //atomicSub(&dest.w, v.w);
}

template<typename T>
struct cuda3DSurface {
    cudaSurfaceObject_t st;
    cudaArray_t arr;

    template<typename U>
    void init(size_t w, size_t h, size_t d) {
        cudaChannelFormatDesc cdc = cudaCreateChannelDesc<U>();

        cudaMalloc3DArray(&arr, &cdc, make_cudaExtent(w, h, d));
        cudaResourceDesc resDesc;
        //memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = arr;


        cudaCreateSurfaceObject(&st, &resDesc);
    }
    cuda3DSurface(size_t w, size_t h, size_t d) {
        init<T>(w, h, d);
    }
    void delete_obj() {
        cudaDestroySurfaceObject(st);
        cudaFreeArray(arr);
    }
};

template<> template<>
void cuda3DSurface<double>::init<double>(size_t w, size_t h, size_t d);

#ifdef USE_DOUBLE_AS_REAL

inline __device__ real4 tex1Dfetch_real4(cudaTextureObject_t tex, int i) {
    int4 d12 = tex1Dfetch<int4>(tex, i*2);
    int4 d34 = tex1Dfetch<int4>(tex, i*2+1);
    //printf("!!%d %d %d %d\n", d12.x, d12.y, d12.z, d12.w);
    real d1 = __hiloint2double(d12.y, d12.x);
    real d2 = __hiloint2double(d12.w, d12.z);

    real d3 = __hiloint2double(d34.y, d34.x);
    real d4 = __hiloint2double(d34.w, d34.z);
    return{ d1,d2,d3,d4 };
}
//#include <cuda_runtime.h>
inline __device__ real surf3Dread_real(const cudaSurfaceObject_t sur,const int x,const int y,const int z) {
    const int2 d1 = surf3Dread<int2>(sur,x*sizeof(int2),y,z);
    /*
    int4 d12 = tex1Dfetch<int4>(tex, i * 2);
    int4 d34 = tex1Dfetch<int4>(tex, i * 2 + 1);
    //printf("!!%d %d %d %d\n", d12.x, d12.y, d12.z, d12.w);
    real d1 = __hiloint2double(d12.y, d12.x);
    real d2 = __hiloint2double(d12.w, d12.z);

    real d3 = __hiloint2double(d34.y, d34.x);
    real d4 = __hiloint2double(d34.w, d34.z);
    */
    return __hiloint2double(d1.y, d1.x);
}

inline __device__ void surf3Dwrite_real(const double v,cudaSurfaceObject_t sur, const  int x, const  int y, const  int z) {

    surf3Dwrite(make_int2(__double2loint(v), __double2hiint(v)), sur, x * sizeof(int2), y, z);
}

#else
inline __device__ real4 tex1Dfetch_real4(cudaTextureObject_t tex, int i) {
    return tex1Dfetch<real4>(tex, i);
}

inline __device__ real surf3Dread_real(cudaSurfaceObject_t sur, int x, int y, int z) {
    /*
    int4 d12 = tex1Dfetch<int4>(tex, i * 2);
    int4 d34 = tex1Dfetch<int4>(tex, i * 2 + 1);
    //printf("!!%d %d %d %d\n", d12.x, d12.y, d12.z, d12.w);
    real d1 = __hiloint2double(d12.y, d12.x);
    real d2 = __hiloint2double(d12.w, d12.z);

    real d3 = __hiloint2double(d34.y, d34.x);
    real d4 = __hiloint2double(d34.w, d34.z);
    */
    return surf3Dread<float>(sur, x, y, z);
}

#endif

__device__ inline real grid_avg8_sobj(cudaSurfaceObject_t sur, int ix, int iy, int iz) {
	const int next_ix=ix==NX-1?0:ix+1;
	const int next_iy=iy==NY-1?0:iy+1;
	const int next_iz=iz==NZ-1?NZ-1:iz+1;
    return real(0.125)*(surf3Dread_real(sur,ix,iy,iz) + surf3Dread_real(sur,next_ix,iy,iz) + surf3Dread_real(sur,ix,next_iy,iz)
        + surf3Dread_real(sur,ix,iy,next_iz) + surf3Dread_real(sur,next_ix,next_iy,iz) + surf3Dread_real(sur,next_ix,iy,next_iz)
        + surf3Dread_real(sur,ix,next_iy,next_iz) + surf3Dread_real(sur,next_ix,next_iy,next_iz));
}
