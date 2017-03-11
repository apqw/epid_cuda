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
#define FAST_PDIFF
#ifdef FAST_PDIFF
__device__ inline double p_diff_x(const double x1, const double x2){
	constexpr double thr=0.5*LX;
	const double diff = x1 - x2;
	long long int sgn2 = __double_as_longlong(thr-__longlong_as_double(__double_as_longlong(diff)&0x7FFFFFFFFFFFFFFFLL))&0x8000000000000000LL;
	sgn2>>=32;
	sgn2>>=16;
	sgn2>>=8;
	sgn2>>=4;
	sgn2>>=2;
	sgn2>>=1;
	return diff-__longlong_as_double((__double_as_longlong(LX)^(__double_as_longlong(diff)&0x8000000000000000LL))&sgn2);
}

__device__ inline float p_diff_x(const float x1, const float x2){
	constexpr float thr=0.5f*LX;
	const float diff = x1 - x2;
	int sgn2 = __float_as_int(thr-__int_as_float(__float_as_int(diff)&0x7FFFFFFF))&0x80000000;
	sgn2>>=16;
	sgn2>>=8;
	sgn2>>=4;
	sgn2>>=2;
	sgn2>>=1;
	return diff-__int_as_float((__float_as_int(LX)^(__float_as_int(diff)&0x80000000))&sgn2);

}

__device__ inline double p_diff_y(const double x1, const double x2){
	constexpr double thr=0.5*LY;
	const double diff = x1 - x2;
	long long int sgn2 = __double_as_longlong(thr-__longlong_as_double(__double_as_longlong(diff)&0x7FFFFFFFFFFFFFFFLL))&0x8000000000000000LL;
	sgn2>>=32;
	sgn2>>=16;
	sgn2>>=8;
	sgn2>>=4;
	sgn2>>=2;
	sgn2>>=1;
	return diff-__longlong_as_double((__double_as_longlong(LY)^(__double_as_longlong(diff)&0x8000000000000000LL))&sgn2);

}

__device__ inline float p_diff_y(const float x1, const float x2){
	constexpr float thr=0.5f*LY;
	const float diff = x1 - x2;
	int sgn2 = __float_as_int(thr-__int_as_float(__float_as_int(diff)&0x7FFFFFFF))&0x80000000;
	sgn2>>=16;
	sgn2>>=8;
	sgn2>>=4;
	sgn2>>=2;
	sgn2>>=1;
	return diff-__int_as_float((__float_as_int(LY)^(__float_as_int(diff)&0x80000000))&sgn2);

}
#else
__host__ __device__ inline real p_diff_x(const real x1, const real x2){
	constexpr real thr=real(0.25)*LX*LX;
	const real diff = x1 - x2;
	const real l_or_zero=real(0.5)*(LX+copysign(LX,diff*diff-thr));
	return diff-copysign(l_or_zero,diff);
	/*
	if (diff > real(0.5)*LX)return diff - LX;
	if (diff <= real(-0.5)*LX)return diff + LX;
	return diff;
	*/
}


__host__ __device__ inline float p_diff_y(const float x1, const float x2){
	constexpr real thr=real(0.25)*LY*LY;
	const real diff = x1 - x2;
	const real l_or_zero=real(0.5)*(LY+copysign(LY,diff*diff-thr));
		return diff-copysign(l_or_zero,diff);
	/*
	if (diff > real(0.5)*LY)return diff - LY;
	if (diff <= real(-0.5)*LY)return diff + LY;
	return diff;
	*/
}
#endif

__device__ inline real p_dist_sq(const real x1, const real y1, const real z1, const real x2, const  real y2, const  real z2)
{
	const real diffx = p_diff_x(x1, x2);
	const real diffy = p_diff_y(y1, y2);
	const real diffz = z1 - z2;
	return diffx*diffx + diffy*diffy + diffz*diffz;
}
__device__ inline real p_dist_sq(const  real4 v1, const  real4 v2){
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

__device__ inline real4 vpsub(const real4 v1, const real4 v2){
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
#define _m_max(a,b) ((a)>(b)?(a):(b))
#define _m_min(a,b) ((a)<(b)?(a):(b))
#define _m_norm_sq(x,y,z) ((x)*(x)+(y)*(y)+(z)*(z))
#endif /* DH_HELPER_H_ */
