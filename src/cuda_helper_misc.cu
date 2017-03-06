#include "cuda_helper_misc.h"

template<> template<>
void cuda3DSurface<double>::init<double>(size_t w, size_t h, size_t d) {
    cudaChannelFormatDesc cdc = cudaCreateChannelDesc<int2>();

    cudaMalloc3DArray(&arr, &cdc, make_cudaExtent(w, h, d));
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;


    cudaCreateSurfaceObject(&st, &resDesc);
}

__global__ void _wrap_bound(cudaTextureObject_t data) {
    if (threadIdx.x >= NX)return;
    const real r1 = surf3Dread_real(data, threadIdx.x, 0, blockIdx.y);
    surf3Dwrite_real(r1, data, threadIdx.x, NY - 1, blockIdx.y);

    const real r2 = surf3Dread_real(data, NX - 1, blockIdx.x, blockIdx.y);
    surf3Dwrite_real(r2, data, 0, blockIdx.x, blockIdx.y);


}

void wrap_bound(cudaTextureObject_t data) {
    _wrap_bound << <dim3(NY, NZ), NX >> > (data);
}

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