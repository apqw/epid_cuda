#pragma once
#include "CellManager.h"
#include "CubicDynArr.h"
void calc_ext_stim(CellManager&cm, cudaSurfaceObject_t* ext_stim,
    const CubicDynArrTexReader<int> cmap1,
    const CubicDynArrTexReader<CMask_t> cmap2, const real* zzmax, cudaSurfaceObject_t* out_stim);