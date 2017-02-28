#pragma once
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T>
class CubicDynArrAccessor {
    int X, Y, Z;
    T* data;
public:
    CubicDynArrAccessor(){}
    CubicDynArrAccessor(int x, int y, int z, T*ptr) :X(x),Y(y),Z(z),data(ptr){
    }

    __host__ void init(int x, int y, int z, T*ptr) {
        X = x;
        Y = y;
        Z = z;
        data = ptr;
    }
    __host__ __device__ T& at(int x, int y, int z) {
        return data[x + X*(y + Y*z)];
    }

    __host__ __device__ const T& at(int x, int y, int z)const {
        return data[x + X*(y + Y*z)];
    }

};

template<typename T>
class CubicDynArrGenerator {
    int X, Y, Z;
    thrust::device_vector<T> data;
    
public:
    CubicDynArrAccessor<T> acc;
    CubicDynArrGenerator(int x, int y, int z) :X(x), Y(y), Z(z), data(x*y*z) {
        acc.init(x, y, z, thrust::raw_pointer_cast(&data[0]));
    }

    void memset_zero() {
        cudaMemset(thrust::raw_pointer_cast(&data[0]), 0x00, X*Y*Z);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }


};