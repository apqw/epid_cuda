

#include "CellManager.h"
#include "pbuf/cell.pb.h"
#include "helper.h"
#include "cuda_helper_misc.h"
#include <fstream>
#include <exception>
#include <string>
#include <algorithm>
#include <iomanip>
#include <numeric>
__host__ __device__ CellAttr::CellAttr()
    :fix_origin(-1),
    pair(-1),
    ca2p_avg(0.0),
    IP3(0.0),
    ex_inert(0.0),
    agek(0.0),
    ageb(0.0),
    ex_fat(0.0),
    in_fat(0.0),
    spr_nat_len(0.0),
    radius(0.0),
    div_age_thresh(0.0),
    rest_div_times(0),
    is_malignant(false),
    is_touch(false),
    dermis(-1),
    nullified(false) {}
/*
bool operator== (const CellDataSet &c1, const CellDataSet &cds) {
    struct r4cmp {
        bool operator()(real4 v1, real4 v2) {
            return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
        }
    };
    bool eq_pos = std::equal(c1.cpos.hv.begin(), c1.cpos.hv.end(), cds.cpos.hv.begin(), r4cmp());

    bool eq_state = std::equal(c1.cstate.hv.begin(), c1.cstate.hv.end(), cds.cstate.hv.begin());

    bool eq_attr = std::equal(c1.cattr.hv.begin(), c1.cattr.hv.end(), cds.cattr.hv.begin());

#ifdef CM_EQUALITY_DBG 
    std::cout << "CellManager Equality Dbg:CellDataSet" << std::endl;
    std::cout << "eq_pos:" << eq_pos << std::endl;
    std::cout << "eq_state:" << eq_state << std::endl;
    std::cout << "eq_attr:" << eq_attr << std::endl;
    if (!eq_attr) {
        auto mism = std::mismatch(c1.cattr.hv.begin(), c1.cattr.hv.end(), cds.cattr.hv.begin(), cds.cattr.hv.end());
        (*mism.first).print();
        (*mism.second).print();
    }
#endif
    return eq_pos&&eq_state&&eq_attr;
}
*/
bool operator== (const MembConn &c1, const MembConn &c2) {
    return std::equal(&c1.conn[0], &c1.conn[0] + MEMB_CONN_NUM
        , &c2.conn[0]);

}

bool operator== (const NonMembConn &c1, const NonMembConn &c2) {
    return c1.conn == c2.conn;
}
bool operator== (const CellAttr &c1, const CellAttr &c2)
{
    return c1.fix_origin == c2.fix_origin&&
        c1.pair == c2.pair&&
        c1.ca2p_avg == c2.ca2p_avg&&
        c1.IP3 == c2.IP3&&
        c1.ex_inert == c2.ex_inert&&
        c1.agek == c2.agek&&
        c1.ageb == c2.ageb&&
        c1.ex_fat == c2.ex_fat&&
        c1.in_fat == c2.in_fat&&
        c1.spr_nat_len == c2.spr_nat_len&&
        c1.radius == c2.radius&&
        c1.div_age_thresh == c2.div_age_thresh&&
        c1.rest_div_times == c2.rest_div_times&&
        c1.is_malignant == c2.is_malignant&&
        c1.is_touch == c2.is_touch&&
        c1.nullified == c2.nullified;
}
/*
void CellDataSet::verify_host_state() const
{
    
    bool size_match =
        (cpos.hv.size() == cstate.hv.size())
        && (cstate.hv.size() == cattr.hv.size())
        && (cattr.hv.size() == cpos.hv.size());
    if (!size_match) {
        throw std::runtime_error("Vector sizes are mismatching. cpos:"_s
            + std::to_string(cpos.hv.size()) + " cstate:"
            + std::to_string(cstate.hv.size()) + " cattr:"
            + std::to_string(cattr.hv.size()));
    }
    

}
size_t CellDataSet::size_on_host() const {
    return _asz;

}

void CellDataSet::push_to_device()
{
    cpos.push_to_device();
    cstate.push_to_device();
    cattr.push_to_device();
}
*/


static bool verify_pb_data(const CellDataPB::CellSet* cs) {
    CELL_STATE last_state = MEMB;
    for (int i = 0; i < cs->cell_size(); i++) {
        const CellDataPB::Cell & cell = cs->cell(i);
        CELL_STATE current_state = CELL_STATE(cell.state());
        if (i == 0 && current_state != MEMB) {
            throw std::runtime_error("Memb data not found.");
            //return false;
        }
        if (last_state != MEMB && current_state == MEMB) {
            throw std::runtime_error("Invalid data order. Memb data is found after non-memb data.");
            //return false;
        }
        last_state = current_state;
    }
    return true;
}

static void convCPBToNative(const CellDataPB::Cell & cell, CellManager::CellAccessor * cacc) {
    cacc->pos->x = cell.x(); cacc->pos->y = cell.y(); cacc->pos->z = cell.z();
    *cacc->state = CELL_STATE(cell.state());
    cacc->attr->ageb = cell.ageb();
    cacc->attr->agek = cell.agek();
    cacc->attr->ca2p_avg = cell.ca2p_avg();
    cacc->attr->rest_div_times = cell.rest_div_times();
    cacc->attr->ex_fat = cell.ex_fat();
    cacc->attr->ex_inert = real(0.0);
    cacc->attr->fix_origin = cell.fix_origin();
    cacc->attr->in_fat = cell.in_fat();
    cacc->attr->IP3 = real(0.0);
    cacc->attr->is_malignant = cell.malignant();
    cacc->attr->is_touch = cell.is_touch();
    cacc->attr->pair = cell.pair_index();
    cacc->attr->radius = cell.radius();
    cacc->attr->spr_nat_len = cell.spr_nat_len();
    cacc->attr->nullified = cell.nullified();
    cacc->attr->div_age_thresh = real(0.0);
}

static void convNativeToCPB(const CellManager::CellAccessor& _cacc, CellDataPB::Cell * cell) {
    const CellManager::CellAccessor* cacc = &_cacc;
    cell->set_x(cacc->pos->x); cell->set_y(cacc->pos->y); cell->set_z(cacc->pos->z);
    cell->set_state(CellDataPB::Cell_CellState(*cacc->state));
    cell->set_ageb(cacc->attr->ageb);
    cell->set_agek(cacc->attr->agek);
    cell->set_ca2p_avg(cacc->attr->ca2p_avg);
    cell->set_rest_div_times(cacc->attr->rest_div_times);
    cell->set_ex_fat(cacc->attr->ex_fat);
    //cacc->attr->ex_inert = real(0.0);
    cell->set_fix_origin(cacc->attr->fix_origin);
    cell->set_in_fat(cacc->attr->in_fat);
    //cacc->attr->IP3 = real(0.0);
    cell->set_malignant(cacc->attr->is_malignant);
    cell->set_is_touch(cacc->attr->is_touch);
    cell->set_pair_index(cacc->attr->pair);
    cell->set_radius(cacc->attr->radius);
    cell->set_spr_nat_len(cacc->attr->spr_nat_len);
    cell->set_nullified(cacc->attr->nullified);
    //div_age_thresh?
}
int * CellManager::swap_data_ptr()
{
    return thrust::raw_pointer_cast(tmp_pending_flg.data());
}

int * CellManager::swap_idx_store_ptr()
{
    return thrust::raw_pointer_cast(swap_available.data());
}
void CellManager::asz_fetch()
{
    _asz = dev_csize_storage[CS_asz];
}
void CellManager::load(std::string pb_path)
{
    std::ifstream ifs(pb_path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open the following Protocol Buffers file:" + pb_path);
    }

    ifs.seekg(0, ifs.end);
    size_t cinputsize = ifs.tellg();
    ifs.clear();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> ibuf(cinputsize);
    ifs.read(&ibuf[0], cinputsize);
    ifs.close();
    size_t dinputsize = cinputsize * 2 + 8;
    std::vector<char> decbuf(dinputsize);
    int dsz = 0;
    while ((dsz = LZ4_decompress_safe(&ibuf[0], &decbuf[0], int(cinputsize), int(dinputsize))) < 0) {
        dinputsize *= 2;
        decbuf.resize(dinputsize);
    }

    CellDataPB::CellSet cs;
    if (!cs.ParseFromArray(&decbuf[0], dsz)) {
        throw std::runtime_error("Failed to parse the following Protocol Buffers file:" + pb_path);
    }
    try {
        verify_pb_data(&cs);
    }
    catch (std::exception& e) {
        throw std::runtime_error("Failed to verify the cell data set with the following reason(s):"_s + e.what());
    }
    CellPos cp;
    CELL_STATE cst;
    CellAttr cat;
    CellAccessor cacc;
    cacc.pos = &cp;
    cacc.state = &cst;
    cacc.attr = &cat;
    for (int i = 0; i < cs.cell_size(); i++) {
        const CellDataPB::Cell & cell = cs.cell(i);
        convCPBToNative(cell, &cacc);

        add_cell_host(&cacc);
    }
}

void CellManager::load_old(std::string old_data_path)
{
    std::ifstream ifs(old_data_path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open the following file:" + old_data_path);
    }
    std::string line;
    //unsigned int phase = 0;
    //int nmemb = 0;
    //int nder = 0;
    CellPos cp;
    CELL_STATE cst;
    CellAttr cat;
    CellAccessor cacc;
    cacc.pos = &cp;
    cacc.state = &cst;
    cacc.attr = &cat;
    CELL_STATE last_state = MEMB;
    int count = 0;
    while (std::getline(ifs, line)) {
        int stateint; real ca2p; int touch;
        int nullified; int is_malignant;
        sscanf(line.c_str(), "%*d %d " R4FMT " " R4FMT " " R4FMT " " R4FMT " " R4FMT " " R4FMT " " R4FMT " " R4FMT " %d " R4FMT " " R4FMT " %d " R4FMT " %d %d %d %d",
            &stateint, &cacc.attr->radius, &cacc.attr->ageb, &cacc.attr->agek
            , &ca2p, &cacc.pos->x, &cacc.pos->y, &cacc.pos->z, &cacc.attr->ca2p_avg, &cacc.attr->rest_div_times
            , &cacc.attr->ex_fat, &cacc.attr->in_fat, &touch, &cacc.attr->spr_nat_len, &cacc.attr->pair
            , &cacc.attr->fix_origin, &nullified, &is_malignant);
        *cacc.state = CELL_STATE(stateint);
        if (count == 0 && *cacc.state != MEMB) {
            throw std::runtime_error("Memb data not found.");
        }
        if (last_state != MEMB&&*cacc.state == MEMB) {
            throw std::runtime_error("Invalid data order. Memb data is found after non-memb data.");
        }
        //ca2p unused
        cacc.attr->is_touch = touch == 1;
        cacc.attr->nullified = nullified == 1;
        cacc.attr->is_malignant = is_malignant == 1;

        add_cell_host(&cacc);
        count++;
    }
}
size_t CellManager::memb_size()const {
    verify_host_internal_state();
    return _msz;
}

size_t CellManager::non_memb_size()const {
    verify_host_internal_state();
    return _asz - _msz;
}

size_t CellManager::all_size()const {
    return _asz;
}
void CellManager::_push_cell_heads() {
    dev_csize_storage.resize(64);
    thrust::device_vector<int> tmpv(64,0);
    tmpv[1]=81818;
    thrust::host_vector<int> tmph(64,0);
    //thrust::fill(dev_csize_storage.begin(), dev_csize_storage.end(), 0);
printf("ueeuue\n");
tmph[CS_msz] = _msz;
    printf("ueeuue__\n");
    tmph[CS_asz] = _asz;
    tmph[CS_fix_hd] = _fix_hd;
    tmph[CS_der_hd] = _der_hd;
    tmph[CS_air_hd] = _air_hd;
    tmph[CS_dead_hd] = _dead_hd;
    tmph[CS_alive_hd] = _alive_hd;
    tmph[CS_musume_hd] = _musume_hd;
    tmph[CS_pair_hd] = _pair_hd;

    thrust::copy(tmph.begin(),tmph.end(),dev_csize_storage.begin());
    printf("ueeuue2\n");
    //dev_csize_storage[CS_pair_end] = _pair_end;

}
int * CellManager::get_dev_csize_ptr()
{
    return thrust::raw_pointer_cast(dev_csize_storage.data());
}
CellIterateRange_device CellManager::get_cell_iterate_range_d()
{
    return CellIterateRange_device(get_dev_csize_ptr());
}
void CellManager::_refresh_cell_pair_count()
{
    auto mur = get_cell_state_range<CI_MUSUME>();
    int count = 0;
    for (int i = mur.first; i < mur.second; i++) {
        if (cattr_all.hv[i].pair < 0)count++;
    }
    _pair_hd = _musume_hd+count;
    _pair_end = _asz;
}

void CellManager::_refresh_cell_count()
{
    assert(
        cpos_all.hv.size() == cattr_all.hv.size()
        && cattr_all.hv.size() == cstate_all.hv.size()
        && cstate_all.hv.size() == all_nm_conn.hv.size()
        && all_nm_conn.hv.size() == cpos_all.hv.size()
    );

    int ccount[16] = { 0 };
    for (int i = 0; i < _asz; i++) {
        ccount[(int)(cstate_all.hv[i])]++;
    }
    _fix_hd = _msz = ccount[(int)MEMB];
    _der_hd = ccount[(int)FIX] + _fix_hd;
    _air_hd = ccount[(int)DER] + _der_hd;
    _dead_hd = ccount[(int)AIR] + _air_hd;
    _alive_hd = ccount[(int)DEAD] + _dead_hd;
    _musume_hd = ccount[(int)ALIVE] + _alive_hd;
    _asz = ccount[(int)MUSUME] + _musume_hd;
    cpos_all.make_margin(_asz);
    cattr_all.make_margin(_asz);
    cstate_all.make_margin(_asz);
    all_nm_conn.make_margin(_asz);
    size_t actual_size = cpos_all.actual_vector_size();
    cpos_all_tmp_d.resize(actual_size);
    cattr_all_tmp_d.resize(actual_size);
    cstate_all_tmp_d.resize(actual_size);
    all_nm_conn_tmp_d.resize(actual_size);
    tmp_pending_flg.resize(actual_size);
    swap_available.resize(actual_size);
    //cell_count[0] = _asz;
    //cell_current_limit[0] = actual_size;
    

    
}

void CellManager::_setup_order_and_count_host()
{
    struct st_comp {
        const CELL_STATE* sptr;
        st_comp(const CELL_STATE* _p):sptr(_p) {}
        int nummap(CELL_STATE st)const {
            switch (st) {
            case MEMB:
                return 0;
            case FIX:case DUMMY_FIX:
                return 1;
            case DER:
                return 2;
            case AIR:
                return 3;
            case DEAD:
                return 4;
            case ALIVE:
                return 5;
            case MUSUME:
                return 6;
            case UNUSED:
            case DISA:
            case BLANK:
                return 1023;
            default:
                throw std::runtime_error("Undefined cell state found.");
            }
        }
        bool operator()(int v1, int v2)const {
            return nummap(sptr[v1]) < nummap(sptr[v2]);
        }
    };
    assert(
        cpos_all.hv.size() == cattr_all.hv.size()
        && cattr_all.hv.size() == cstate_all.hv.size()
        && cstate_all.hv.size() == all_nm_conn.hv.size()
        && all_nm_conn.hv.size() == cpos_all.hv.size()
    );
    _asz = cpos_all.hv.size();
    std::vector<int> indices(_asz);
    std::vector<int> subindices(_asz);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), st_comp(&cstate_all.hv[0]));
    thrust::host_vector<CellPos> tmp_cpos(_asz);
    thrust::host_vector<CellAttr> tmp_cattr(_asz);
    thrust::host_vector<CELL_STATE> tmp_cst(_asz);
    thrust::host_vector<NonMembConn> tmp_nmconn(_asz);

    int qq = 0;
    for (int i = 0; i < _asz; i++) {
        if (cattr_all.hv[i].pair >= 0)printf("%d:%d  %d %d\n", i, cattr_all.hv[i].pair, cattr_all.hv[cattr_all.hv[i].pair].pair == i ? 1 : 0,qq++);
    }

    printf("\n\n");
    for (int i = 0; i < _asz; i++) {
        subindices[indices[i]] = i;
    }
    for (int i = 0; i < _asz; i++) {
        tmp_cpos[i] = cpos_all.hv[indices[i]];
        tmp_cattr[i] = cattr_all.hv[indices[i]];
        if (tmp_cattr[i].pair >= 0) tmp_cattr[i].pair = subindices[tmp_cattr[i].pair];
        tmp_cst[i] = cstate_all.hv[indices[i]];
        tmp_nmconn[i] = all_nm_conn.hv[indices[i]];
    }
    cpos_all.hv.swap(tmp_cpos);
    cattr_all.hv.swap(tmp_cattr);
    cstate_all.hv.swap(tmp_cst);
    all_nm_conn.hv.swap(tmp_nmconn);
    _refresh_cell_count();
    auto mur = get_cell_state_range<CI_MUSUME>();
    std::iota(indices.begin()+mur.first, indices.begin() + mur.second, mur.first);

    qq = 0;
    for (int i = 0; i < _asz; i++) {
        if (cattr_all.hv[i].pair >= 0)printf("%d:%d:%d  %d %d\n", mur.first, i, cattr_all.hv[i].pair, cattr_all.hv[cattr_all.hv[i].pair].pair== i?1:0,qq++);
    }
    printf("\n\n");

    std::stable_partition(indices.begin() + mur.first, indices.begin() + mur.second,
        [&](int i)->bool {
        return cattr_all.hv[i].pair < 0;
    });

    for (int i = mur.first; i < mur.second; i++) {
        subindices[indices[i]] = i;
    }
    for (int i = mur.first; i < mur.second; i++) {
        tmp_cpos[i] = cpos_all.hv[indices[i]];
        tmp_cattr[i] = cattr_all.hv[indices[i]];
        if (tmp_cattr[i].pair >= 0) tmp_cattr[i].pair = subindices[tmp_cattr[i].pair];
        tmp_cst[i] = cstate_all.hv[indices[i]];
        tmp_nmconn[i] = all_nm_conn.hv[indices[i]];
    }

    for (int i = mur.first; i < mur.second; i++) {
        cpos_all.hv[i]= tmp_cpos[i];
        cattr_all.hv[i]= tmp_cattr[i];
        cstate_all.hv[i]= tmp_cst[i];
        all_nm_conn.hv[i]=tmp_nmconn[i];
    }
    for (int i = _fix_hd; i < _der_hd; i++) {
        if (cattr_all.hv[i].pair >= 0) {
            cattr_all.hv[i].pair = subindices[cattr_all.hv[i].pair];
            cattr_all.hv[cattr_all.hv[i].pair].pair = i;
        }
    }

    qq = 0;
    for (int i = 0; i < _asz; i++) {
        if(cattr_all.hv[i].pair>=0)printf("%d:%d:%d %d %d\n", mur.first, i, cattr_all.hv[i].pair, cattr_all.hv[cattr_all.hv[i].pair].pair == i ? 1 : 0,qq++);
    }
    

    _refresh_cell_pair_count();
printf("eusyo\n");

    _push_cell_heads();
    printf("eusy2\n");
}

void CellManager::set_up_after_load()
{
    _setup_order_and_count_host();
    refresh_memb_conn_host();
    push_to_device();
}

CellPos * CellManager::get_device_pos_all()
{
    return thrust::raw_pointer_cast(cpos_all.dv.data());
}
CellPos * CellManager::get_device_pos_all_out()
{
    return thrust::raw_pointer_cast(cpos_all_out.data());
}
void CellManager::pos_swap_device()
{
    cpos_all.dv.swap(cpos_all_out);
}
void CellManager::correct_internal_host_state()
{

}

/*
CELL_STATE * CellManager::get_device_non_memb_state()
{
    return cstate_all.get_raw_device_ptr() + _msz;
}
*/
/*
CELL_STATE * CellManager::get_device_state_all()
{
    return thrust::raw_pointer_cast(&cstate_all[0]);
}

CellAttr * CellManager::get_device_attr_all()
{
    return thrust::raw_pointer_cast(&cattr_all[0]);
}
*/
NonMembConn * CellManager::get_device_all_nm_conn()
{
    return all_nm_conn.get_raw_device_ptr();
}

MembConn * CellManager::get_device_mconn()
{
    return mconn.get_raw_device_ptr();
}

CellAttr * CellManager::get_device_attr()
{
    return cattr_all.get_raw_device_ptr();
}

CELL_STATE * CellManager::get_device_cstate()
{
    return cstate_all.get_raw_device_ptr();
}


cudaTextureObject_t CellManager::get_pos_tex()
{
    return pos_tex;
}

void CellManager::refresh_pos_tex()
{
    size_t asz = cpos_all.actual_vector_size();

    cudaTextureObject_t ct;
    cudaResourceDesc resDesc = make_real4_resource_desc(get_device_pos_all(), asz);
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;

    cudaCreateTextureObject(&ct, &resDesc, &texDesc, NULL);
    pos_tex = ct;
}

template<typename T>
static void concat_dev_vec(const T& base_dvec,const T& add_vec, T& out_dvec) {
    out_dvec = base_dvec;
    out_dvec.insert(out_dvec.end(), add_vec.begin(), add_vec.end());
}
/*

template<typename T>
static void concat_host_vec_into_dev_vec(const thrust::host_vector<T>& base_hvec, const thrust::host_vector<T>& add_hvec, T& out_dvec) {
    thrust::host_vector<T> tmp = base_hvec;
    tmp.insert(tmp.end(),add_hvec)
}
*/
void CellManager::push_to_device()
{
    //memb_data.push_to_device();
    //non_memb_data.push_to_device();
    mconn.push_to_device(); 
    all_nm_conn.push_to_device();
    cpos_all.push_to_device();
    cattr_all.push_to_device();
    cstate_all.push_to_device();
    //concat_dev_vec(memb_data.cpos.dv, non_memb_data.cpos.dv, cpos_all);
    
    //concat_dev_vec(memb_data.cstate.dv, non_memb_data.cstate.dv, cstate_all);
    //concat_dev_vec(memb_data.cattr.dv, non_memb_data.cattr.dv, cattr_all);
    CUDA_SAFE_CALL(cudaGetLastError());

    size_t asz = all_size();
//    nm_filter.resize(asz);

    cpos_all_out.resize(cpos_all.actual_vector_size());
    thrust::copy(cpos_all.dv.begin(), cpos_all.dv.end(), cpos_all_out.begin());
    refresh_pos_tex();

    
}
void CellManager::fetch()
{
    verify_host_internal_state();
    //memb_data.cattr.fetch();
    //non_memb_data.cattr.fetch();
    cpos_all.fetch();
    cattr_all.fetch();
    cstate_all.fetch();
    _asz = dev_csize_storage[CS_asz];
    //thrust::copy(cpos_all.begin(), cpos_all.begin() + memb_size(), memb_data.cpos.hv.begin());
   // thrust::copy(cpos_all.begin() + memb_size(), cpos_all.end(), non_memb_data.cpos.hv.begin());
    
}
bool operator== (const CellManager &c1, const CellManager &oc)
{
    bool eq_md = true;// c1.memb_data == oc.memb_data;
    bool eq_nmd = true;// c1.non_memb_data == oc.non_memb_data;
    bool eq_mconn = std::equal(c1.mconn.hv.begin(), c1.mconn.hv.end(), oc.mconn.hv.begin());
    bool eq_nmconn = std::equal(c1.all_nm_conn.hv.begin(), c1.all_nm_conn.hv.end(), oc.all_nm_conn.hv.begin());
#ifdef CM_EQUALITY_DBG
    std::cout << "CellManager Equality Dbg:" << std::endl;
    std::cout << "eq_md:" << eq_md << std::endl;
    std::cout << "eq_nmd:" << eq_nmd << std::endl;
    std::cout << "eq_mconn:" << eq_mconn << std::endl;
    std::cout << "eq_nmconn:" << eq_nmconn << std::endl;
#endif

    return eq_md&&eq_nmd&&eq_mconn&&eq_nmconn;
}

void CellManager::verify_host_internal_state()const
{
    
    /*
    bool memb_size_match =
        (memb_data.size_on_host() == mconn.hv.size());

    if (!memb_size_match) {
        throw std::runtime_error("The following 2 vectors' size are mismatching. memb_data:"_s
            + std::to_string(memb_data.size_on_host()) + " mconn:" + std::to_string(mconn.hv.size()));
    }
    */
    /*
    bool non_memb_size_match =
        (non_memb_data.size_on_host() == nmconn.hv.size());

    if (!non_memb_size_match) {
        throw std::runtime_error("The following 2 vectors' size are mismatching. non_memb_data:"_s
            + std::to_string(non_memb_data.size_on_host()) + " nmconn:" + std::to_string(nmconn.hv.size()));
    }
    */
    /*
    bool memb_num_correct = mconn.hv.size() == MEMB_NUM_X*MEMB_NUM_Y;
    if (!memb_num_correct) {
        printf("mnum:%zu\n", mconn.hv.size());
        throw std::runtime_error("Memb num incorrect.");
    }
    */
}

void CellManager::clear_all_non_memb_conn_both()
{
    all_nm_conn.memset_zero_both();
}
const real * CellManager::zzmax_ptr()const
{
    return thrust::raw_pointer_cast(v_zzmax.data());
}

real * CellManager::zzmax_ptr()
{
    return thrust::raw_pointer_cast(v_zzmax.data());
}
/*
CellManager::CellAccessor CellManager::get_memb_host(int idx)
{
    CellAccessor cat;
    cat.state = &memb_data.cstate.hv[idx];
    cat.pos = &memb_data.cpos.hv[idx];
    cat.mconn = &mconn.hv[idx];
    cat.nmconn = &all_nm_conn.hv[idx];
    cat.attr = &memb_data.cattr.hv[idx];
    return cat;
}
*/

CellManager::CellAccessor CellManager::get_cell_acc(int idx)
{
    CellAccessor cat;
    cat.state = &cstate_all.hv[idx];
    cat.pos = &cpos_all.hv[idx];
    cat.mconn = &mconn.hv[idx];
    cat.nmconn = &all_nm_conn.hv[idx];
    cat.attr = &cattr_all.hv[idx];
    return cat;
}
/*
CellManager::CellAccessor CellManager::get_non_memb_host(int idx)
{
    CellAccessor cat;
    cat.state = &non_memb_data.cstate.hv[idx];
    cat.pos = &non_memb_data.cpos.hv[idx];
    cat.nmconn = &all_nm_conn.hv[idx+memb_size()];
    cat.attr = &non_memb_data.cattr.hv[idx];
    return cat;
}
*/

void CellManager::output(std::string out)
{
    std::ofstream ofs(out, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open the following file:" + out);
    }

    CellDataPB::CellSet cs;

    //const size_t ms = memb_size();
    const size_t asz = memb_size();

    for (int i = 0; i < asz; i++) {
        convNativeToCPB(get_cell_acc(i), cs.add_cell());
    }
    /*
    const size_t nms = non_memb_size();
    for (int i = 0; i < nms; i++) {
        convNativeToCPB(get_non_memb_host(i), cs.add_cell());
    }
    */
    std::string obuf;
    if (!cs.SerializeToString(&obuf)) {
        throw std::runtime_error("Failed to serialize cell data into the following file:" + out);
    }
    std::vector<char> cbuf(obuf.size());
    int osize = 0;
    if ((osize = LZ4_compress_default(obuf.c_str(), &cbuf[0], int(obuf.size()), int(obuf.size()))) <= 0) {
        throw std::runtime_error("Failed to compress cell data.");
    }
    ofs.write(&cbuf[0], osize);


}



void CellManager::output_old(std::string out)
{
    //for dbg
    struct __output_old_fn {
        static void out(const CellManager::CellAccessor& cacc, std::ofstream& wfile,int idx) {
            using namespace std;
            wfile << idx << " "
                << *cacc.state << " "
                << fixed << setprecision(15) << cacc.attr->radius << " "
                << fixed << setprecision(15) << cacc.attr->ageb << " "
                << fixed << setprecision(15) << cacc.attr->agek << " "
                << fixed << setprecision(15) << cacc.attr->ca2p_avg << " " //ca2p
                << fixed << setprecision(15) << cacc.pos->x << " "
                << fixed << setprecision(15) << cacc.pos->y << " "
                << fixed << setprecision(15) << cacc.pos->z << " "
                << fixed << setprecision(15) << cacc.attr->ca2p_avg << " "
                << cacc.attr->rest_div_times << " "
                << fixed << setprecision(15) << cacc.attr->ex_fat << " "
                << fixed << setprecision(15) << cacc.attr->in_fat << " "
                << (cacc.attr->is_touch ? 1 : 0) << " "
                << fixed << setprecision(15) << cacc.attr->spr_nat_len << " "
                << (int)(cacc.attr->pair <0 ? (int)-1 : (int)(cacc.attr->pair)) << " "
                << cacc.attr->fix_origin << " " << (int)(cacc.attr->nullified ? (int)1 : (int)0) << " "
                << (int)(cacc.attr->is_malignant ? 1 : 0)

                << std::endl;
        }
    };
    std::ofstream wfile(out);
    if (!wfile) {
        std::cout << "Output file creation error. Filename:" << out << std::endl;
        exit(1);
    }
    const size_t ms = memb_size();
    const size_t asz = all_size();
    for (int i = 0; i < asz; i++) {
        __output_old_fn::out(get_cell_acc(i), wfile, i);
    }

    /*
    const size_t nms = non_memb_size();
    for (size_t i = 0; i < nms; i++) {
        __output_old_fn::out(get_non_memb_host(int(i)), wfile, int(i+ms));
    }
    */
}

void CellManager::add_cell_host(const CellAccessor * cacc)
{
    /*
    if (*cacc->state == MEMB) {
        //assert(cacc->mconn != nullptr);
        memb_data.cpos.hv.push_back(*cacc->pos);
        memb_data.cstate.hv.push_back(*cacc->state);
        memb_data.cattr.hv.push_back(*cacc->attr);
        mconn.hv.push_back(MembConn());
    }
    else {
        non_memb_data.cpos.hv.push_back(*cacc->pos);
        non_memb_data.cstate.hv.push_back(*cacc->state);
        non_memb_data.cattr.hv.push_back(*cacc->attr);
        
    }
    */
    cpos_all.hv.push_back(*cacc->pos);
    cstate_all.hv.push_back(*cacc->state);
    cattr_all.hv.push_back(*cacc->attr);
    all_nm_conn.hv.push_back(NonMembConn());

    if (*cacc->state == MEMB) {
        mconn.hv.push_back(MembConn()); 
        //_msz++;
    }
    else {
        
    }
    _asz++;
}

void CellManager::refresh_memb_conn_host()
{
    verify_host_internal_state();
    const size_t mbsz = memb_size();

    for (int j = 0; j < mbsz; j++) {

        size_t jj = j%MEMB_NUM_X;
        //size_t kk = j / MEMB_NUM_X;
        MembConn& mb = mconn.hv[j];
        if (jj == 0) {
            mb.conn[3] = j + MEMB_NUM_X - 1;

        }
        else {
            mb.conn[3] = j - 1;
        }

        if (jj == MEMB_NUM_X - 1) {
            mb.conn[0] = j - (MEMB_NUM_X - 1);

        }
        else {
            mb.conn[0] = j + 1;
        }
    }

    for (int j = 0; j < mbsz; j++) {

        //size_t jj = j%MEMB_NUM_X;
        size_t kk = j / MEMB_NUM_X;
        MembConn& mb = mconn.hv[j];

        int top = int( (j - MEMB_NUM_X + mbsz) % mbsz );
        int bot = int( (j + MEMB_NUM_X) % mbsz );

        if (kk % 2 == 0) {
            mb.conn[1] = top;
            mb.conn[2] = mconn.hv[top].conn[3];
            mb.conn[4] = mconn.hv[bot].conn[3];
            mb.conn[5] = bot;
        }
        else {
            mb.conn[1] = mconn.hv[top].conn[0];
            mb.conn[2] = top;
            mb.conn[4] = bot;
            mb.conn[5] = mconn.hv[bot].conn[0];
        }
    }
}




CellManager::CellManager()
{
    _msz = 0;
    _asz = 0;
    _fix_hd = 0;
    _der_hd = 0;
    _air_hd = 0;
    _dead_hd = 0;
    _alive_hd = 0;
    _musume_hd = 0;
    _pair_end=0;
    _pair_hd=0;
    pos_tex=0;
    v_zzmax.resize(1);
}


CellManager::~CellManager()
{
}

void CellAttr::print()const
{
#define _pca(nm) std::cout <<#nm<<":" << nm << std::endl

    _pca(fix_origin);
    _pca(pair);
    _pca(ca2p_avg);
    _pca(IP3);
    _pca(ex_inert);
    _pca(agek);
    _pca(ageb);
    _pca(ex_fat);
    _pca(in_fat);
    _pca(spr_nat_len);
    _pca(radius);
    _pca(div_age_thresh);
    _pca(rest_div_times);
    _pca(is_malignant);
    _pca(is_touch);
    _pca(nullified);
#undef _pca
}

__inline__ __device__
real warpReduceMax(real val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = max(val, __shfl_down(val, offset));
    return val;
}

__inline__ __device__
real blockReduceMax(real val) {

    static __shared__ real shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);     // Each warp performs partial reduction

    if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

                                  //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -REAL_MAX;

    if (wid == 0) val = warpReduceMax(val); //Final reduce within first warp

    return val;
}
__global__ void deviceReduceMaxPos(const CellPos *in, real* out, CellIterateRange_device cir) {
    real val = -REAL_MAX;
    const int N = cir.nums[CS_asz];
    //reduce multiple elements per thread
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N;
        i += blockDim.x * gridDim.x) {
        val=max(val, in[i].z);
    }
    val = blockReduceMax(val);
    if (threadIdx.x == 0)
        out[blockIdx.x] = val;
}

__global__ void deviceReduceMaxPos_finalize(const real *in, real* out_zzmax) {
    real val = in[threadIdx.x];
    val = blockReduceMax(val);
    if (threadIdx.x == 0)
        *out_zzmax = val;
}
#define REDU_ZMAX_BLC (1024)
#define REDU_ZMAX_TH (512)
void CellManager::refresh_zzmax()
{
    static thrust::device_vector<real> tmp_max_out(REDU_ZMAX_BLC);
    
    deviceReduceMaxPos << <REDU_ZMAX_BLC, REDU_ZMAX_TH >> >(get_device_pos_all(), thrust::raw_pointer_cast(tmp_max_out.data()), get_cell_iterate_range_d());
    deviceReduceMaxPos_finalize << <1, REDU_ZMAX_BLC >> > (thrust::raw_pointer_cast(tmp_max_out.data()), zzmax_ptr());
}
