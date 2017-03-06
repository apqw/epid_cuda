
#include "define.h"
#include "CubicDynArr.h"
#include "cuda_helper_misc.h"
class CellManager;
void map_gen(CellManager&cm, CubicDynArrAccessor<int> cmap1, CubicDynArrAccessor<CMask_t> cmap2);