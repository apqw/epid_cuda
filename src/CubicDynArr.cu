#include "CubicDynArr.h"

template<>template<>
__host__ cudaResourceDesc CubicDynArrTexReader<int>::make_rd<int>(int* ptr, size_t bsize) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = ptr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.desc.y = 0;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;
    resDesc.res.linear.sizeInBytes = bsize;
    return resDesc;
}

template<>template<>
__host__ cudaResourceDesc CubicDynArrTexReader<unsigned int>::make_rd<unsigned int>(unsigned int* ptr, size_t bsize) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = ptr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.desc.y = 0;
    resDesc.res.linear.desc.z = 0;
    resDesc.res.linear.desc.w = 0;
    resDesc.res.linear.sizeInBytes = bsize;
    return resDesc;
}