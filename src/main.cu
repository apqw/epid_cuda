

#include "pbuf/cell.pb.h"
#include "pbuf/multidim.pb.h"
#include "CellManager.h"
#include "cuda_test.h"
#include "cell_connection.h"
#include "cell_movement.h"
#include "CubicDynArr.h"
#include "map.h"
#include "filter.h"
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
    cm.refresh_memb_conn_host();
    std::cout << "Eee" << std::endl;
    //for(int i=0;i<1000;i++)
    cm.push_to_device();
    CubicDynArrGenerator<int> cmap1(NX, NY, NZ);
    CubicDynArrGenerator<CMask_t> cmap2(NX, NY, NZ);
    //cm_disp_test(cm);
    //cudaDeviceSynchronize();
    std::cout << "Continue." << std::endl;
    connect_cell(cm);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
    for (int i = 0; i < 10000; i++) {
        calc_cell_movement(cm);
        //CUDA_SAFE_CALL(cudaGetLastError());
        if (i % 1000 == 0) {
            printf("fetching\n");
            cm.fetch();
            printf("done\n");
            cm.output(std::to_string(i));
            printf("out %d", i);
        }
        //cudaDeviceSynchronize();
    }
    

    map_gen(cm, cmap1.acc, cmap2.acc);

    std::cout << "End." << std::endl;

    google::protobuf::ShutdownProtobufLibrary();
}
