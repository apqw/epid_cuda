#pragma once

#include "define.h"
cudaResourceDesc make_real4_resource_desc(CellPos*r4ptr, size_t len);
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

#else
inline __device__ real4 tex1Dfetch_real4(cudaTextureObject_t tex, int i) {
    return tex1Dfetch<real4>(tex, i);
}
#endif