#pragma once
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<typename T>
class CubicDynArrAccessor {
    //should change type from int to size_t?
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
    __host__ __device__ T& raw_at(int idx) {
        return data[idx];
    }
    __host__ __device__ int size()const {
        return X*Y*Z;
    }

};

template<typename T>
class CubicDynArrTexReader {
    int X, Y, Z;
    
    T* data;
public:
    cudaTextureObject_t ct;

    CubicDynArrTexReader(int x, int y, int z, T*ptr) :X(x), Y(y), Z(z), data(ptr),ct(0) {
    }
    CubicDynArrTexReader():CubicDynArrTexReader(0,0,0,nullptr) {}
    template<typename U>
    __host__ cudaResourceDesc make_rd(T*ptr, size_t bsize) {
        throw std::logic_error("DO NOT USE TexReader with incompatible types");
    }
    
    __host__ void refresh() {
    	if(ct!=0)cudaDestroyTextureObject(ct);
        cudaTextureObject_t tmp_ct;
        cudaResourceDesc resDesc = make_rd<T>(data, X*Y*Z * sizeof(T));
        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;
        cudaCreateTextureObject(&tmp_ct, &resDesc, &texDesc, NULL);
        ct = tmp_ct;
    }
    __host__ void init(int x, int y, int z, T*ptr) {
        X = x;
        Y = y;
        Z = z;
        data = ptr;

        refresh();
        
        
    }
    /*
    __device__ T& at(int x, int y, int z) = delete;
    */
    __device__ const T& at(int x, int y, int z)const {
        return tex1Dfetch<T>(ct, x + X*(y + Y*z));
        //return data[];
    }

};
template<>template<>
__host__ cudaResourceDesc CubicDynArrTexReader<int>::make_rd<int>(int* ptr, size_t bsize);
template<>template<>
__host__ cudaResourceDesc CubicDynArrTexReader<unsigned int>::make_rd<unsigned int>(unsigned int* ptr, size_t bsize);
template<typename T>
class CubicDynArrGenerator {
    int X, Y, Z;
    thrust::device_vector<T> data;
    
public:
    CubicDynArrAccessor<T> acc;
    CubicDynArrTexReader<T> make_texr() {
        CubicDynArrTexReader<T> texr;
        texr.init(X, Y, Z, thrust::raw_pointer_cast(data.data()));
        return texr;
    }
    CubicDynArrGenerator(int x, int y, int z) :X(x), Y(y), Z(z), data(x*y*z) {
        acc.init(x, y, z, thrust::raw_pointer_cast(data.data()));
        
    }



    void memset_zero() {
        cudaMemset(thrust::raw_pointer_cast(data.data()), 0x00, sizeof(T)*X*Y*Z);
       // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }


};
