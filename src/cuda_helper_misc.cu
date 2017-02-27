#include "cuda_helper_misc.h"
#ifdef USE_DOUBLE_AS_REAL
cudaResourceDesc make_real4_resource_desc(CellPos*r4ptr, size_t len) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = r4ptr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.desc.z = 32;
    resDesc.res.linear.desc.w = 32;
    resDesc.res.linear.sizeInBytes = len * sizeof(real4);
    return resDesc;
}
#else
cudaResourceDesc make_real4_resource_desc(CellPos*r4ptr, size_t len) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = r4ptr;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // bits per channel
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.desc.z = 32;
    resDesc.res.linear.desc.w = 32;
    resDesc.res.linear.sizeInBytes = len * sizeof(real4);
    return resDesc;
}
#endif