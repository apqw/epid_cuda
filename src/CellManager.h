#pragma once
#pragma warning(disable: 4819)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "define.h"
#include "LFStack.h"
#include <lz4/lz4.h>


//#define CM_EQUALITY_DBG 1
template<typename T>
struct dhv_pair {
    thrust::device_vector<T> dv;
    thrust::host_vector<T> hv;
    void push_to_device(){
        dv = hv;
    }
    T* get_raw_device_ptr() {
        return thrust::raw_pointer_cast(&dv[0]);
    }

    const T* get_raw_device_ptr()const {
        return thrust::raw_pointer_cast(&dv[0]);
    }
};

#define MEMB_CONN_NUM (6)

struct MembConn {
    CellIndex conn[MEMB_CONN_NUM];
    friend bool operator== (const MembConn &c1, const MembConn &c2);
};
bool operator== (const MembConn &c1, const MembConn &c2);
#define CELL_CONN_NUM (200)
struct NonMembConn {
    LFStack<CellIndex, CELL_CONN_NUM> conn;
    friend bool operator== (const NonMembConn &c1, const NonMembConn &c2);
};
bool operator== (const NonMembConn &c1, const NonMembConn &c2);
class CellAttr {
public:
    CellIndex fix_origin;

    CellIndex pair;
    //real ca2p;
    real ca2p_avg;
    real IP3;
    real ex_inert;
    real agek;
    real ageb;
    real ex_fat;
    real in_fat;
    real spr_nat_len;
    real radius;
    real div_age_thresh;
    int rest_div_times;
    bool is_malignant;
    bool is_touch;
    bool nullified;

    friend bool operator== (const CellAttr &c1, const CellAttr &c2);
    void print()const;
    CellAttr();
};
bool operator== (const CellAttr &c1, const CellAttr &c2);
struct CellDataSet {
    dhv_pair<CellPos> cpos;
    dhv_pair<CELL_STATE> cstate;
    dhv_pair<CellAttr> cattr;
    void verify_host_state()const;
    size_t size_on_host()const;
    void push_to_device();

    friend bool operator== (const CellDataSet &c1, const CellDataSet &c2);
};
bool operator== (const CellDataSet &c1, const CellDataSet &c2);

class CellManager
{
    CellDataSet memb_data; CellDataSet non_memb_data;

    dhv_pair<MembConn> mconn; dhv_pair<NonMembConn> nmconn;

    thrust::device_vector<CellPos> cpos_all;
    thrust::device_vector<CELL_STATE> cstate_all;
    thrust::device_vector<CellAttr> cattr_all;
    cudaTextureObject_t pos_tex;
public:
    struct CellAccessor {
        CELL_STATE* state;
        CellPos* pos;
        MembConn* mconn; NonMembConn* nmconn;
        CellAttr* attr;
    };
    void load(std::string pb_path);
    void load_old(std::string old_data_path);
    void output(std::string out);
    void add_cell_host(const CellAccessor*);
    void refresh_memb_conn_host();
    //void refresh_non_memb_conn_host();
    void verify_host_internal_state()const;
    CellAccessor get_memb_host(int idx);
    CellAccessor get_non_memb_host(int idx);
    size_t memb_size()const;
    size_t non_memb_size()const;
    size_t all_size()const;
    CellPos* get_device_pos_all();
    CELL_STATE* get_device_state_all();
    CellAttr* get_device_attr_all();
    NonMembConn* get_device_nmconn();
    cudaTextureObject_t get_pos_tex();
    void push_to_device();
    friend bool operator== (const CellManager &c1, const CellManager &c2);
    CellManager();
    ~CellManager();
};
bool operator== (const CellManager &c1, const CellManager &c2);
