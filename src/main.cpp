

#include "pbuf/cell.pb.h"
#include "CellManager.h"
#include "cuda_test.h"
#include "cell_connection.h"
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

    //cm_disp_test(cm);
    //cudaDeviceSynchronize();
    std::cout << "Continue." << std::endl;
    connect_cell(cm);
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
    std::cout << "End." << std::endl;

    google::protobuf::ShutdownProtobufLibrary();
}