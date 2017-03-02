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
//#include "filter.h"

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


struct CellIterateRange {
    int head;
    int size;
    int mod;
    int base;
    CellIterateRange(size_t _h, size_t _s) :head((int)_h), size((int)_s),mod(0),base(0) {}
    CellIterateRange(size_t _h, size_t _s, size_t _m) :head((int)_h), size((int)_s), mod((int)_m),base(0) {}
    CellIterateRange(size_t _h, size_t _s, size_t _m, size_t _b) :head((int)_h), size((int)_s), mod((int)_m),base((int)_b) {}
    __device__ int idx_full(int index)const {
        return (index + head) % mod + base;
    }


    __device__ int idx_simple(int index)const {
        return index + head;
    }
};
enum CellIterateType {
    CI_ALL,
    CI_MEMB,
    CI_NON_MEMB,
    CI_FIX,
    CI_DER,
    CI_AIR,
    CI_DEAD,
    CI_ALIVE,
    CI_MUSUME,
    CI_PAIR
};


class CellManager
{
    //CellDataSet memb_data; CellDataSet non_memb_data;

    dhv_pair<MembConn> mconn; 
    //dhv_pair<NonMembConn> nmconn;
    
    //maybe read_only
    //thrust::device_vector<CellPos> cpos_all;
    //thrust::device_vector<CELL_STATE> cstate_all;
    //thrust::device_vector<CellAttr> cattr_all;
    cudaTextureObject_t pos_tex;
    thrust::device_vector<CellPos> cpos_all_out;
    
    dhv_pair<CellPos> cpos_all;
    dhv_pair<CellAttr> cattr_all;
    dhv_pair<CELL_STATE> cstate_all;
    dhv_pair<NonMembConn> all_nm_conn;
    size_t _msz;
    size_t _asz;
    size_t _fix_hd;
    size_t _der_hd;
    size_t _air_hd;
    size_t _dead_hd;
    size_t _alive_hd;
    size_t _musume_hd;
    size_t _pair_hd;
    size_t _pair_end;
public:

    //Cell order
    //MEMB-FIX-DER-AIR-DEAD-ALIVE-MUSUME
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
    void verify_host_internal_state()const;
    void clear_all_non_memb_conn_both();
    CellAccessor get_cell_acc(int idx);
    size_t memb_size()const;
    size_t all_size()const;
    size_t non_memb_size()const;
    template<CellIterateType...CIR>
    CellIterateRange get_cell_iterate_range() {
        throw std::logic_error("Undefined cell iterate type.");
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_ALL>() {
        return CellIterateRange{ 0,_asz };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_NON_MEMB>() {
        return{ _msz,_asz-_msz };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_FIX>() {
        return{ _fix_hd,_der_hd-_fix_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_DER>() {
        return{ _der_hd,_air_hd - _der_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_AIR>() {
        return{ _air_hd,_dead_hd - _air_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_DEAD>() {
        return{ _dead_hd,_alive_hd - _dead_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_ALIVE>() {
        return{ _alive_hd,_musume_hd - _alive_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_MUSUME>() {
        return{ _musume_hd,_asz - _musume_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_PAIR>() {
        return{ _pair_hd,_pair_end-_pair_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_AIR, CI_DEAD, CI_ALIVE>() {
        return{ _air_hd,_musume_hd - _air_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_MUSUME,CI_FIX>() {
        return{ _musume_hd-_fix_hd,(_asz - _musume_hd)+(_der_hd-_fix_hd),(_asz - _fix_hd),_fix_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_ALIVE,CI_MUSUME, CI_FIX>() {
        return{ _alive_hd-_fix_hd,(_asz - _alive_hd) + (_der_hd - _fix_hd),_asz-_fix_hd,_fix_hd };
    }
    template<>
    CellIterateRange get_cell_iterate_range<CI_DER,CI_AIR,CI_DEAD,CI_MEMB>() {
        return{ _der_hd,_alive_hd-_der_hd+_fix_hd,_alive_hd,0 };
    }

    template<CellIterateType...CIR>
    std::pair<int, int> get_cell_state_range() {
        throw std::logic_error("Undefined cell range type.");
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_ALL>() {
        return std::make_pair(0, (int)_asz);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_NON_MEMB>() {
        return std::make_pair(_fix_hd, (int)_asz);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_FIX>() {
        return std::make_pair(_fix_hd, _der_hd);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_DER>() {
        return std::make_pair(_der_hd, _air_hd);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_AIR>() {
        return std::make_pair(_air_hd, _dead_hd);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_DEAD>() {
        return std::make_pair(_dead_hd, _alive_hd);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_ALIVE>() {
        return std::make_pair(_alive_hd, _musume_hd);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_MUSUME>() {
        return std::make_pair(_musume_hd, (int)_asz);
    }
    template<>
    std::pair<int, int> get_cell_state_range<CI_PAIR>() {
        return std::make_pair(_pair_hd,_pair_end);
    }

    void _refresh_cell_pair_count();
    void _refresh_cell_count();
    void _setup_order_and_count_host();

    void set_up_after_load();

    CellPos* get_device_pos_all();
    CellPos* get_device_pos_all_out();
    void pos_swap_device();

    void correct_internal_host_state();
   
    NonMembConn* get_device_all_nm_conn();
    MembConn* get_device_mconn();
    CellAttr* get_device_attr();
    CELL_STATE* get_device_cstate();
    cudaTextureObject_t get_pos_tex();
    void refresh_pos_tex();
    void push_to_device();
    void fetch();
    /*
    struct NonMembIndexFilter {
    private:
    



        thrust::device_vector<int> filtered_result;
    CellManager* parent;
    public:
        int* get_flt_head() {
            return thrust::raw_pointer_cast(&filtered_result[0]);
        }
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

        struct PairPred {
            const CellAttr* nmcat;
            const int offset;
            PairPred(const CellAttr* _ptr, int _offset = 0) :nmcat(_ptr), offset(_offset) {}
            __host__ __device__ bool operator()(const int i)const {
                return nmcat[i + offset].pair >= 0;
            }
        };
        NonMembIndexFilter(CellManager* _cm):parent(_cm){}
        void resize(size_t s) {
            filtered_result.resize(s);
        }
        
        template<CELL_STATE...state_Or>
        const CellIndex* filter_by_state(int*num) {
            const int msz = int( parent->memb_size() );
            
            thrust::device_vector<CellIndex>::iterator out_end = 
                thrust::copy_if(thrust::make_counting_iterator<int>(msz)
                    , thrust::make_counting_iterator<int>(int( parent->all_size() ))
                , filtered_result.begin()
                , CstPred<state_Or...>(parent->get_device_non_memb_state(),-msz));
                
            *num = int( thrust::distance(filtered_result.begin(), out_end) );
            return thrust::raw_pointer_cast(&filtered_result[0]);
        }

        const CellIndex* filter_by_pair(int*num) {
            const int msz = int(parent->memb_size());
            thrust::device_vector<CellIndex>::iterator out_end =
                thrust::copy_if(thrust::make_counting_iterator<int>(msz)
                    , thrust::make_counting_iterator<int>(int( parent->all_size() ))
                    , filtered_result.begin()
                    , PairPred(parent->get_device_nmattr(), -msz));
            *num = int( thrust::distance(filtered_result.begin(), out_end) );
            return thrust::raw_pointer_cast(&filtered_result[0]);
        }

        
    };
    NonMembIndexFilter nm_filter;
    */
    friend bool operator== (const CellManager &c1, const CellManager &c2);
    CellManager();
    ~CellManager();
};
bool operator== (const CellManager &c1, const CellManager &c2);
