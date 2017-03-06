

#include "pbuf/cell.pb.h"
#include "pbuf/multidim.pb.h"
#include "CellManager.h"
#include "cuda_test.h"
#include "cell_connection.h"
#include "cell_movement.h"
#include "CubicDynArr.h"
#include "map.h"
#include "filter.h"
#include "cell_state_renew.h"
#include "cuda_helper_misc.h"
#include "calc_ext_stim.h"
#include <cmdline/cmdline.h>
#include <iostream>
#include <fstream>

int main(int argc,char**argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
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
        if (i % 1000 == 0&&i!=0) {
            printf("fetching\n");
            cm.fetch();
            printf("done\n");
            cm.output_old(std::to_string(i));
            printf("out %d", i);
        }
        DBG_ONLY(CUDA_SAFE_CALL(cudaDeviceSynchronize()));
        DBG_ONLY(printf("count:%d\n", i));
    }
    

    

    std::cout << "End." << std::endl;

    google::protobuf::ShutdownProtobufLibrary();
}
