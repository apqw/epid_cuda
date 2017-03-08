

#include "pbuf/cell.pb.h"
#include "pbuf/multidim.pb.h"
#include "CellManager.h"
#include "cuda_test.h"
#include "cell_connection.h"
#include "cell_movement.h"
#include "CubicDynArr.h"
#include "map.h"
//#include "filter.h"
#include "cell_state_renew.h"
#include "cuda_helper_misc.h"
#include "calc_ext_stim.h"
#include "ca2p.h"
#include <cmdline/cmdline.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
void print_device_infos(){
    int n;    //デバイス数
    CUDA_SAFE_CALL(cudaGetDeviceCount(&n));

    for(int i = 0; i < n; ++i){
        cudaDeviceProp dev;

        // デバイスプロパティ取得
        CUDA_SAFE_CALL(cudaGetDeviceProperties(&dev, i));

        printf("device %d\n", i);
        printf(" device name : %s\n", dev.name);
        printf(" total global memory : %d (MB)\n", dev.totalGlobalMem/1024/1024);
        printf(" shared memory / block : %d (KB)\n", dev.sharedMemPerBlock/1024);
        printf(" register / block : %d\n", dev.regsPerBlock);
        printf(" warp size : %d\n", dev.warpSize);
        printf(" max pitch : %d (B)\n", dev.memPitch);
        printf(" max threads / block : %d\n", dev.maxThreadsPerBlock);
        printf(" max size of each dim. of block : (%d, %d, %d)\n", dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
        printf(" max size of each dim. of grid  : (%d, %d, %d)\n", dev.maxGridSize[0], dev.maxGridSize[1], dev.maxGridSize[2]);
        printf(" clock rate : %d (MHz)\n", dev.clockRate/1000);
        printf(" total constant memory : %d (KB)\n", dev.totalConstMem/1024);
        printf(" compute capability : %d.%d\n", dev.major, dev.minor);
        printf(" alignment requirement for texture : %d\n", dev.textureAlignment);
        printf(" device overlap : %s\n", (dev.deviceOverlap ? "ok" : "not"));
        printf(" num. of multiprocessors : %d\n", dev.multiProcessorCount);
        printf(" kernel execution timeout : %s\n", (dev.kernelExecTimeoutEnabled ? "on" : "off"));
        printf(" integrated : %s\n", (dev.integrated ? "on" : "off"));
        printf(" host memory mapping : %s\n", (dev.canMapHostMemory ? "on" : "off"));

        printf(" compute mode : ");
        if(dev.computeMode == cudaComputeModeDefault) printf("default mode (multiple threads can use) \n");
        else if(dev.computeMode == cudaComputeModeExclusive) printf("exclusive mode (only one thread will be able to use)\n");
        else if(dev.computeMode == cudaComputeModeProhibited) printf("prohibited mode (no threads can use)\n");

    }
}
__global__ void real_3d_init(cudaSurfaceObject_t cso){
	surf3Dwrite_real(real(0.0),cso,threadIdx.x,blockIdx.x,blockIdx.y);
}
int main(int argc,char**argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    print_device_infos();
    cudaSetDevice(0);

    printf("test13333\n");
    cmdline::parser psr;
    psr.add<std::string>("input", 'i', "input cell data",true);
    psr.add("old", '\0', "use old format");
    psr.parse_check(argc, argv);

    CellManager cm;
    bool use_old_format = psr.exist("old");
    std::string path = psr.get<std::string>("input");
    if (psr.exist("old")) {
        cm.load_old(path);
    }
    else {
        cm.load(path);
    }

    cm.set_up_after_load();
    //cuda3DSurface<int> cmap1(NX, NY, NZ);
    //cuda3DSurface<CMask_t> cmap2(NX, NY, NZ);
    CubicDynArrGenerator<int> cmap1(NX, NY, NZ);
    CubicDynArrGenerator<CMask_t> cmap2(NX, NY, NZ);
    cuda3DSurface<real> ext_stim(NX, NY, NZ);
    cuda3DSurface<real> ext_stim_out(NX, NY, NZ);
    real_3d_init<<<dim3(NY,NZ),NX>>>(ext_stim.st);
    real_3d_init<<<dim3(NY,NZ),NX>>>(ext_stim_out.st);
        /*
        CubicDynArrGenerator<real> ext_stim(NX, NY, NZ);
    CubicDynArrGenerator<real> ext_stim_out(NX, NY, NZ);
    */
    //cm_disp_test(cm);
    //cudaDeviceSynchronize();
    std::cout << "Continue." << std::endl;
    connect_cell(cm);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
    
    CUDA_SAFE_CALL(cudaGetLastError());
    cm.output_old("precalc");
    
    CubicDynArrTexReader<int> cmap1_texr = cmap1.make_texr();
    CubicDynArrTexReader<CMask_t> cmap2_texr = cmap2.make_texr();
   // cmap1.texr.refresh(); cmap2.texr.refresh();
    DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
    printf("test1\n");
   // CUDA_SAFE_CALL(cudaDeviceSynchronize());
   
   // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int i = 0; i < 10000; i++) {
        
        const size_t msz = cm.memb_size();
        calc_cell_movement(cm);
        DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
        cm.refresh_pos_tex();
        exec_renew(cm);
        DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
        cm.refresh_pos_tex();
        //cm.asz_fetch();
        connect_cell(cm);
        DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
        map_gen(cm, cmap1.acc, cmap2.acc);
        cm.refresh_zzmax();
        cmap1_texr.refresh(); cmap2_texr.refresh();
        calc_ext_stim(cm, &ext_stim.st, cmap1_texr, cmap2_texr, cm.zzmax_ptr(), &ext_stim_out.st);
        calc_ca2p(cm,ext_stim.st,cmap1_texr.ct,cmap2_texr.ct);
        if ((i % 1000 == 0&&i!=0)) {
            printf("fetching\n");
            cm.fetch();
            printf("done\n");
            cm.output_old(std::to_string(i));
            printf("out %d", i);
        }
        DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
        //DBG_ONLY(printf("count:%d\n", i));
    }
    

    

    std::cout << "End." << std::endl;
std::this_thread::sleep_for(std::chrono::seconds(3));
    google::protobuf::ShutdownProtobufLibrary();
}
