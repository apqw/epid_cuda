#pragma once
#pragma warning(disable: 4819)
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include "define.h"
#include "LFStack.h"
#include "cuda_helper_misc.h"
#include <lz4/lz4.h>
#include <thrust/distance.h>
#include <thrust/count.h>

//#define CM_EQUALITY_DBG 1

#define MEMB_CONN_NUM (6)

struct MembConn {
    CellIndex conn[MEMB_CONN_NUM];
    friend bool operator== (const MembConn &c1, const MembConn &c2);
};
bool operator== (const MembConn &c1, const MembConn &c2);

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
    __host__ __device__ CellAttr();
};
bool operator== (const CellAttr &c1, const CellAttr &c2);

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
    CI_PAIR,
    CI_MUSUME_NOPAIR
};

enum CSIZE_STR :int {
    CS_msz = 0,
    CS_asz = 1,
    CS_fix_hd = 2,
    CS_der_hd = 3,
    CS_air_hd = 4,
    CS_dead_hd = 5,
    CS_alive_hd = 6,
    CS_musume_hd = 7,
    CS_pair_hd = 8,
    //CS_pair_end=9,
    CS_count_alive = 10,
    CS_count_musume = 11,
    CS_count_dead = 12,
    CS_count_alive_act = 13,
    CS_count_store = 14,
    CS_count_available = 15,
    CS_count_air = 16,
    //CS_count_dead = 17,
    CS_count_removed = 18,
    CS_count_removed_air = 19,
    CS_count_removed_dead = 20,
    CS_count_musume_divided = 21,
    CS_count_pair = 22,
    CS_count_musume_nopair = 23,
    CS_count_sw=24,
    CS_count_loop_ca2p=25,
    CS_count_num_sc=26,
    CS_count_actual_removed_air=27,
    CS_count_actual_removed_dead=28
};
struct CellIterateRange_device {
    int* nums;
    CellIterateRange_device(int* _str) :nums(_str) {}
    __device__ void use_cache(int* cache_ptr) {
        nums = cache_ptr;
    }
    __device__ inline int _idx_simple(int index, int head)const {
        return index + head;
    }

    __device__ inline int _idx_full(int index, int head, int mod, int base)const {
        return (index + head) % mod + base;
    }
    template<CellIterateType...CIR>
    __device__ int idx(int index)const {
        //static_assert(false, "Undefined CIR.");
        assert(false);
        return -8181;
    }

    template<CellIterateType...CIR>
    __device__ int size()const {
        assert(false);
        return -8181;
    }

};
template<>
__device__ inline int CellIterateRange_device::idx<CI_ALL>(int index)const {
    return _idx_simple(index, 0);
}

template<>
__device__ inline int CellIterateRange_device::size<CI_ALL>()const {
    return nums[CS_asz];
}

template<>
__device__ inline int CellIterateRange_device::idx<CI_NON_MEMB>(int index)const {
    return _idx_simple(index, nums[CS_msz]);
}

template<>
__device__ inline int CellIterateRange_device::size<CI_NON_MEMB>()const {
    return nums[CS_asz] - nums[CS_msz];
}
template<>
__device__  inline int CellIterateRange_device::idx<CI_FIX>(int index)const {
    return _idx_simple(index, nums[CS_fix_hd]);
}

template<>
__device__  inline int CellIterateRange_device::size<CI_FIX>()const {
    return nums[CS_der_hd] - nums[CS_fix_hd];
}
template<>
__device__ inline  int CellIterateRange_device::idx<CI_DER>(int index)const {
    return _idx_simple(index, nums[CS_der_hd]);
}

template<>
__device__ inline  int CellIterateRange_device::size<CI_DER>()const {
    return nums[CS_air_hd] - nums[CS_der_hd];
}
template<>
__device__ inline  int CellIterateRange_device::idx<CI_AIR>(int index)const {
    return _idx_simple(index, nums[CS_air_hd]);
}

template<>
__device__  inline int CellIterateRange_device::size<CI_AIR>()const {
    return nums[CS_dead_hd] - nums[CS_air_hd];
}
template<>
__device__ inline  int CellIterateRange_device::idx<CI_DEAD>(int index)const {
    return _idx_simple(index, nums[CS_dead_hd]);
}
template<>
__device__  inline int CellIterateRange_device::size<CI_DEAD>()const {
    return nums[CS_alive_hd] - nums[CS_dead_hd];
}
template<>
__device__ inline  int CellIterateRange_device::idx<CI_ALIVE>(int index)const {
    return _idx_simple(index, nums[CS_alive_hd]);
}
template<>
__device__  inline int CellIterateRange_device::size<CI_ALIVE>()const {
    return nums[CS_musume_hd] - nums[CS_alive_hd];
}
template<>
__device__  inline int CellIterateRange_device::idx<CI_DEAD, CI_ALIVE>(int index)const {
    return _idx_simple(index, nums[CS_dead_hd]);
}
template<>
__device__  inline int CellIterateRange_device::size<CI_DEAD, CI_ALIVE>()const {
    return nums[CS_musume_hd] - nums[CS_dead_hd];
}
template<>
__device__  inline int CellIterateRange_device::idx<CI_MUSUME>(int index)const {
    return _idx_simple(index, nums[CS_musume_hd]);
}
template<>
__device__ inline  int CellIterateRange_device::size<CI_MUSUME>()const {
    return nums[CS_asz] - nums[CS_musume_hd];
}
template<>
__device__ inline  int CellIterateRange_device::idx<CI_PAIR>(int index)const {
    return _idx_simple(index, nums[CS_pair_hd]);
}
template<>
__device__ inline  int CellIterateRange_device::size<CI_PAIR>()const {
    return nums[CS_asz] - nums[CS_pair_hd];
}

template<>
__device__  inline int CellIterateRange_device::idx<CI_MUSUME_NOPAIR>(int index)const {
    return _idx_simple(index, nums[CS_musume_hd]);
}
template<>
__device__  inline int CellIterateRange_device::size<CI_MUSUME_NOPAIR>()const {
    return nums[CS_pair_hd] - nums[CS_musume_hd];
}
template<>
__device__ inline  int CellIterateRange_device::idx<CI_AIR, CI_DEAD, CI_ALIVE>(int index)const {
    return _idx_simple(index, nums[CS_air_hd]);
}

template<>
__device__  inline int CellIterateRange_device::size<CI_AIR, CI_DEAD, CI_ALIVE>()const {
    return nums[CS_musume_hd] - nums[CS_air_hd];
}

template<>
__device__ inline  int CellIterateRange_device::idx<CI_ALIVE, CI_MUSUME>(int index)const {
    return _idx_simple(index, nums[CS_alive_hd]);
}

template<>
__device__  inline int CellIterateRange_device::size<CI_ALIVE, CI_MUSUME>()const {
    return nums[CS_asz] - nums[CS_alive_hd];
}
template<>
__device__ inline  int CellIterateRange_device::idx<CI_MUSUME, CI_FIX>(int index)const {
    return _idx_full(index, nums[CS_musume_hd] - nums[CS_fix_hd], nums[CS_asz] - nums[CS_fix_hd], nums[CS_fix_hd]);
}
template<>
__device__  inline int CellIterateRange_device::size<CI_MUSUME, CI_FIX>()const {
    return (nums[CS_asz] - nums[CS_musume_hd]) + (nums[CS_der_hd] - nums[CS_fix_hd]);
}

template<>
__device__  inline int CellIterateRange_device::idx<CI_ALIVE, CI_MUSUME, CI_FIX>(int index)const {
    return _idx_full(index, nums[CS_alive_hd] - nums[CS_fix_hd], nums[CS_asz] - nums[CS_fix_hd], nums[CS_fix_hd]);
}
template<>
__device__  inline int CellIterateRange_device::size<CI_ALIVE, CI_MUSUME, CI_FIX>()const {
    return (nums[CS_asz] - nums[CS_alive_hd]) + (nums[CS_der_hd] - nums[CS_fix_hd]);
}
template<>
__device__  inline int CellIterateRange_device::idx<CI_DER, CI_AIR, CI_DEAD, CI_MEMB>(int index)const {
    return _idx_full(index, nums[CS_der_hd], nums[CS_alive_hd], 0);
}
template<>
__device__  inline int CellIterateRange_device::size<CI_DER, CI_AIR, CI_DEAD, CI_MEMB>()const {
    return (nums[CS_alive_hd] - nums[CS_der_hd]) + (nums[CS_fix_hd]);
}


class CellManager
{
	static constexpr real margin_ratio=1.5;
	static constexpr real resize_rest_ratio=0.2;
    dhv_pair<MembConn> mconn; 
    cudaTextureObject_t pos_tex;
    thrust::device_vector<CellPos> cpos_all_out;
    
    dhv_pair<CellPos> cpos_all;
    dhv_pair<CellAttr> cattr_all;
    dhv_pair<CELL_STATE> cstate_all;
    dhv_pair<NonMembConn> all_nm_conn;

    thrust::device_vector<CellPos> cpos_all_tmp_d;
    thrust::device_vector<CellAttr> cattr_all_tmp_d;
    thrust::device_vector<CELL_STATE> cstate_all_tmp_d;
    thrust::device_vector<NonMembConn> all_nm_conn_tmp_d;
    thrust::device_vector<int> tmp_pending_flg;
    thrust::device_vector<int> swap_available;
    thrust::device_vector<int> dev_csize_storage;
    thrust::device_vector<real> v_zzmax;
    //this order
    int _msz;
    int _asz;
    int _fix_hd;
    int _der_hd;
    int _air_hd;
    int _dead_hd;
    int _alive_hd;
    int _musume_hd;
    int _pair_hd;
    int _pair_end;
    
public:
    int* swap_data_ptr();
    int* swap_idx_store_ptr();
    //Cell order
    //MEMB-FIX-DER-AIR-DEAD-ALIVE-MUSUME
    struct CellAccessor {
        CELL_STATE* state;
        CellPos* pos;
        MembConn* mconn; NonMembConn* nmconn;
        CellAttr* attr;
    };
    void asz_fetch();
    void load(std::string pb_path);
    void load_old(std::string old_data_path);
    void output(std::string out);
    void output_old(std::string out);
    void add_cell_host(const CellAccessor*);
    void refresh_memb_conn_host();
    void verify_host_internal_state()const;
    void clear_all_non_memb_conn_both();
    const real* zzmax_ptr()const;
    real* zzmax_ptr();
    CellAccessor get_cell_acc(int idx);
    size_t memb_size()const;
    size_t all_size()const;
    size_t non_memb_size()const;
    void _push_cell_heads();
    int* get_dev_csize_ptr();
    CellIterateRange_device get_cell_iterate_range_d();
    void buffer_size_check();
    template<CellIterateType...CIR>
    std::pair<int, int> get_cell_state_range() {
        throw std::logic_error("Undefined cell range type.");
    }

    
    void _refresh_cell_pair_count();
    void _refresh_cell_count();
    void _setup_order_and_count_host();

    void set_up_after_load();

    CellPos* get_device_pos_all();
    CellPos* get_device_pos_all_out();
    void pos_swap_device();

    void correct_internal_host_state();
   
    void refresh_zzmax();
    NonMembConn* get_device_all_nm_conn();
    MembConn* get_device_mconn();
    CellAttr* get_device_attr();
    CELL_STATE* get_device_cstate();
    cudaTextureObject_t get_pos_tex();
    void refresh_pos_tex();
    void push_to_device();
    void fetch();
    friend bool operator== (const CellManager &c1, const CellManager &c2);
    CellManager();
    ~CellManager();
};

template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_ALL>() {
    return std::make_pair(0, (int)_asz);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_NON_MEMB>() {
    return std::make_pair(_fix_hd, (int)_asz);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_FIX>() {
    return std::make_pair(_fix_hd, _der_hd);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_DER>() {
    return std::make_pair(_der_hd, _air_hd);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_AIR>() {
    return std::make_pair(_air_hd, _dead_hd);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_DEAD>() {
    return std::make_pair(_dead_hd, _alive_hd);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_ALIVE>() {
    return std::make_pair(_alive_hd, _musume_hd);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_MUSUME>() {
    return std::make_pair(_musume_hd, (int)_asz);
}
template<>
inline std::pair<int, int> CellManager::get_cell_state_range<CI_PAIR>() {
    return std::make_pair(_pair_hd, _pair_end);
}
bool operator== (const CellManager &c1, const CellManager &c2);
void reduce_zmax(const CellPos*cp,real*optr, CellIterateRange_device cir);
