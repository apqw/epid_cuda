#include "cuda_test.h"
#include "device_launch_parameters.h"
#include "cuda_helper_misc.h"
__global__ void cm_disp_ker(CellPos* cpptr,int size, cudaTextureObject_t ct) {
    int index= 0+blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        CellPos* cp = cpptr + index;
        real4 r4 = tex1Dfetch_real4(ct, index);
        printf("orig %d/%d:%f %f %f :%f %f %f\n", index,size,cp->x, cp->y, cp->z, r4.x, r4.y, r4.z);
        
        //printf("cmpr %d/%d:%f %f %f\n", index, size, r4.x, r4.y, r4.z);
    }
}
void cm_disp_test(CellManager& cm) {
    cudaTextureObject_t ct;
    cudaResourceDesc resDesc=make_real4_resource_desc(cm.get_device_pos_all(), cm.all_size());
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&ct, &resDesc, &texDesc, NULL);
    cm_disp_ker << <128, 128 >> > (cm.get_device_pos_all(), int(cm.all_size()),ct);
}