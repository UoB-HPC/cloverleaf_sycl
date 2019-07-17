/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */


#ifndef CLOVERLEAF_SYCL_CXX14_COMPAT_HPP
#define CLOVERLEAF_SYCL_CXX14_COMPAT_HPP

#include <memory>


// taken from https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
// one of the possible reference implementations
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
	return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

#endif //CLOVERLEAF_SYCL_CXX14_COMPAT_HPP
