/*
 * helper.h
 *
 *  Created on: 2017/02/13
 *      Author: yasu7890v
 */

#ifndef HELPER_H_
#define HELPER_H_

#include <string>
#include <memory>
std::string operator ""_s(const char*,size_t);

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}


#endif /* HELPER_H_ */
