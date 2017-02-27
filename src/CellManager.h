#pragma once
#pragma warning(disable: 4819)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include "define.h"
#include "LFStack.h"
#include <lz4/lz4.h>
#include <thrust/distance.h>
#include <thrust/count.h>

//#define CM_EQUALITY_DBG 1
template<typename T>
struct dhv_pair {
    thrust::device_vector<T> dv;
    thrust::host_vector<T> hv;
    void push_to_device(){
        dv = hv;
    }
    void fetch() {
        hv = dv;
    }
    T* get_raw_device_ptr() {
        return thrust::raw_pointer_cast(&dv[0]);
    }

    const T* get_raw_device_ptr()const {
        return thrust::raw_pointer_cast(&dv[0]);
    }

    void memset_zero_both() {
        std::memset(&hv[0], 0x00, sizeof(T)*hv.size());
        CUDA_SAFE_CALL(cudaMemset(get_raw_device_ptr(), 0x00, sizeof(T)*dv.size()));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
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
    CellIndex dermis;
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
    //void fetch();

    friend bool operator== (const CellDataSet &c1, const CellDataSet &c2);
};
bool operator== (const CellDataSet &c1, const CellDataSet &c2);





class CellManager
{
    CellDataSet memb_data; CellDataSet non_memb_data;

    dhv_pair<MembConn> mconn; dhv_pair<NonMembConn> nmconn;

    //maybe read_only
    thrust::device_vector<CellPos> cpos_all;
    //thrust::device_vector<CELL_STATE> cstate_all;
    //thrust::device_vector<CellAttr> cattr_all;
    cudaTextureObject_t pos_tex;
    thrust::device_vector<CellPos> cpos_all_out;
    
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
    void output_old(std::string out);
    void add_cell_host(const CellAccessor*);
    void refresh_memb_conn_host();
    //void refresh_non_memb_conn_host();
    void verify_host_internal_state()const;
    void clear_all_non_memb_conn_both();
    CellAccessor get_memb_host(int idx);
    CellAccessor get_non_memb_host(int idx);
    size_t memb_size()const;
    size_t non_memb_size()const;
    size_t all_size()const;
    CellPos* get_device_pos_all();
    CellPos* get_device_pos_all_out();
    void pos_swap_device();
   
    //const CELL_STATE* get_device_state_all();
    //const CellAttr* get_device_attr_all();
    CELL_STATE* get_device_non_memb_state();
    NonMembConn* get_device_nmconn();
    MembConn* get_device_mconn();
    CellAttr* get_device_nmattr();
    cudaTextureObject_t get_pos_tex();
    void refresh_pos_tex();
    void push_to_device();
    void fetch();
    
    struct NonMembIndexFilter {
    private:
    thrust::device_vector<int> filtered_result;



 
    CellManager* parent;
    public:
        struct teststs {
            __host__ __device__ bool operator()(const int i)const {
                return true;
            }
        };
        template<CELL_STATE...state_Or>
        struct CstPred {
            const CELL_STATE*cst;
            const int offset;
            CstPred(const CELL_STATE* _ptr,int _offset=0) :cst(_ptr),offset(_offset) {}
            __host__ __device__ bool operator()(const int i)const {
                using expand_type = int[];
                bool sor = false;
                (void)expand_type {
                    0, (sor = sor || (cst[i+offset] == state_Or), void(), 0)...
                };
                return sor;
            }
        };
        NonMembIndexFilter(CellManager* _cm):parent(_cm){}
        void resize(size_t s) {
            filtered_result.resize(s);
        }
        
        template<CELL_STATE...state_Or>
        const CellIndex* filter_by_state(int*num) {
            size_t msz = parent->memb_size();
            
            thrust::device_vector<CellIndex>::iterator out_end = 
                thrust::copy_if(thrust::make_counting_iterator<int>(msz)
                , thrust::make_counting_iterator<int>(parent->all_size())
                , filtered_result.begin()
                , CstPred<state_Or...>(parent->get_device_non_memb_state(),-msz));
                
            *num = thrust::distance(filtered_result.begin(), out_end);
            return thrust::raw_pointer_cast(&filtered_result[0]);
        }
        
    };
    NonMembIndexFilter nm_filter;
    friend bool operator== (const CellManager &c1, const CellManager &c2);
    CellManager();
    ~CellManager();
};
bool operator== (const CellManager &c1, const CellManager &c2);
