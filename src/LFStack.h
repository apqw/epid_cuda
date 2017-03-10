/*
 * LFStack.h
 *
 *  Created on: 2017/02/20
 *      Author: yasu7890v
 */

#ifndef LFSTACK_H_
#define LFSTACK_H_
//if T is zero-fillable,this is also zero-fillable
template<typename T,unsigned N>
class LFStack {

public:
    int head;
    T data[N];
	__host__ __device__ LFStack():head(0){

	};
    __device__ void push_back_d(const T& d) {

        data[atomicAdd(&head, 1)] = d;
    };
	__device__ void push_back_d(T&& d){
		data[atomicAdd(&head,1)]=d;
	};

	__device__ void clear_d(){
		atomicExch(&head,0);
	};
	__host__ __device__ int size()const{
		return head;
	};
	__host__ __device__ T& operator[](int idx){
		return data[idx];
	}
	__host__ __device__ const T& operator[](int idx)const{
		return data[idx];
	}
    __host__ __device__ void copy_fast(const LFStack<T,N>* ptr) {
        head = ptr->head;
        memcpy(&data[0], &ptr->data[0], sizeof(T)*ptr->head);
    }

    __host__ __device__ bool isvalid()const{
    	return head<N;
    }

    template<typename T1, unsigned N1>
    friend bool operator== (const LFStack<T1,N1> &c1, const LFStack<T1, N1> &c2);
};
template<typename T, unsigned N>
bool operator== (const LFStack<T, N> &c1, const LFStack<T, N> &c2) {
    bool eq = true;
    for (int i = 0; i < N; i++) {
        eq = eq && (c1.data[i] == c2.data[i]);
    }
    return eq && (c1.head == c2.head);
}
#endif /* LFSTACK_H_ */
