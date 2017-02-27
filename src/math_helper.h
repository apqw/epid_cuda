/*
 * dh_helper.h
 *
 *  Created on: 2017/02/13
 *      Author: yasu7890v
 */

#ifndef DH_HELPER_H_
#define DH_HELPER_H_
#include "define.h"
/*
__host__ __device__ inline real rmod(real v1,const real r1){
	while(v1>r1){
		v1-=r1;
	}
	while(v1<=-r1){
		v1+=r1;
	}
	return v1;
}
*/

__host__ __device__ inline real p_diff_x(const real x1, const real x2){
	const real diff = x1 - x2;
	if (diff > real(0.5)*LX)return diff - LX;
	if (diff <= real(-0.5)*LX)return diff + LX;
	return diff;
}

__host__ __device__ inline real p_diff_y(const real x1, const real x2){
	const real diff = x1 - x2;
	if (diff > real(0.5)*LY)return diff - LY;
	if (diff <= real(-0.5)*LY)return diff + LY;
	return diff;
}

__host__ __device__ inline real p_dist_sq(const real x1, const real y1, const real z1, const real x2, const  real y2, const  real z2)
{
	const real diffx = p_diff_x(x1, x2);
	const real diffy = p_diff_y(y1, y2);
	const real diffz = z1 - z2;
	return diffx*diffx + diffy*diffy + diffz*diffz;
}
__host__ __device__ inline real p_dist_sq(const  real4 v1, const  real4 v2){
	return p_dist_sq(v1.x,v1.y,v1.z,v2.x,v2.y,v2.z);
}

__host__ __device__ inline real4 vadd(const real4 v1, const real4 v2){
	return {v1.x+v2.x,v1.y+v2.y,v1.z+v2.z,v1.w+v2.w};
}

__host__ __device__ inline void vadda(real4* v1, const real4 v2){
	v1->x+=v2.x;
	v1->y+=v2.y;
	v1->z+=v2.z;
	v1->w+=v2.w;
}

__host__ __device__ inline real4 vsub(const real4 v1, const real4 v2){
	return {v1.x-v2.x,v1.y-v2.y,v1.z-v2.z,v1.w-v2.w};
}

__host__ __device__ inline real4 vpsub(const real4 v1, const real4 v2){
	return {p_diff_x(v1.x,v2.x),p_diff_y(v1.y,v2.y),v1.z-v2.z,v1.w-v2.w};
}

__host__ __device__ inline real4 cvmul(const real c1, const real4 v2){
	return {c1*v2.x,c1*v2.y,c1*v2.z,c1*v2.w};
}

__host__ __device__ inline real4 vdiv(const real4 v1, const real c2){
	return {v1.x/c2,v1.y/c2,v1.z/c2,v1.w/c2};
}

__host__ __device__ inline real dot3(const real4 v1, const real4 v2){
	return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

__host__ __device__ inline real4 cross3(const real4 v1, const real4 v2){
	return {v1.y*v2.z-v1.z*v2.y,
            v1.z*v2.x-v1.x*v2.z,
            v1.x*v2.y-v1.y*v2.x,real(0.0)};
}

__host__ __device__ inline real get_norm3(const real4 v1){
	return sqrtr(v1.x*v1.x+v1.y*v1.y+v1.z*v1.z);
}

__device__ inline real get_norm_inv3(const real4 v1) {
    return rsqrt(v1.x*v1.x + v1.y*v1.y + v1.z*v1.z);
}

__host__ __device__ inline real get_norm3sq(const real4 v1) {
    return v1.x*v1.x + v1.y*v1.y + v1.z*v1.z;
}
__host__ __device__ inline real4 normalize3(const real4 v1){
	return vdiv(v1,get_norm3(v1));
}

__device__ inline real4 normalize3_r_dev_w_inv(const real4 v1) {
    return cvmul(get_norm_inv3(v1),v1);
}
__host__ __device__ inline real4 normalize3(const real4 v1,real*norm){
	return vdiv(v1,*norm=get_norm3(v1));
}
__device__ inline real4 normalize3_r_dev_w_inv(const real4 v1, real*norm_inv) {
    return cvmul(*norm_inv=get_norm_inv3(v1), v1);
}
__host__ __device__ inline real min0(const real v) {
    return v > real(0.0) ? v : real(0.0);
}
#define POW3(v) ((v)*(v)*(v))
#define POW6(v) (POW3(v)*POW3(v))

#endif /* DH_HELPER_H_ */
