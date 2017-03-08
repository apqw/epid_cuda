/*
 * ca2p.h
 *
 *  Created on: 2017/03/06
 *      Author: y1
 */

#ifndef CA2P_H_
#define CA2P_H_
#include "CubicDynArr.h"
class CellManager;
void calc_ca2p(CellManager&cm,const cudaSurfaceObject_t extstim,cudaTextureObject_t cmap1,cudaTextureObject_t cmap2);


#endif /* CA2P_H_ */
