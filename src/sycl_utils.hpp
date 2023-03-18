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

#ifndef CLOVERLEAF_SYCL_SYCL_UTILS_HPP
#define CLOVERLEAF_SYCL_SYCL_UTILS_HPP

#include <CL/sycl.hpp>
#include <iostream>
#include <utility>

#define SYCL_DEBUG // enable for debugging SYCL related things, also syncs kernel calls
#define SYNC_KERNELS // enable for fully synchronous (e.g queue.wait_and_throw()) kernel calls

// this namespace houses all SYCL related abstractions
namespace clover {

template <typename T, int N> struct Buffer {};

template <typename T> struct Buffer<T, 1> {
  size_t size;
  T *data;
  explicit Buffer(size_t size, sycl::queue &q) : size(size), data(sycl::malloc_shared<T>(size, q)) {}
  T &operator[](size_t i) const { return data[i]; }
};

template <typename T> struct Buffer<T, 2> {
  size_t sizeX, sizeY;
  T *data;
  Buffer(size_t sizeX, size_t sizeY, sycl::queue &q)
      : sizeX(sizeX), sizeY(sizeY), data(sycl::malloc_shared<T>(sizeX * sizeY, q)) {}
  T &operator()(size_t i, size_t j) const { return data[j + i * sizeY]; }
};

template <typename T> void free(sycl::queue &q, T &&b) { sycl::free(b.data, q); }

template <typename T, typename... Ts> void free(sycl::queue &q, T &&t, Ts &&...ts) {
  free(q, t);
  free(q, std::forward<Ts>(ts)...);
}

struct Range1d {
  const size_t from, to;
  const size_t size;
  template <typename A, typename B> Range1d(A from, B to) : from(from), to(to), size(to - from) {
    assert(from < to);
    assert(size != 0);
  }
  friend std::ostream &operator<<(std::ostream &os, const Range1d &d) {
    os << "Range1d{"
       << " X[" << d.from << "->" << d.to << " (" << d.size << ")]"
       << "}";
    return os;
  }
};

struct Range2d {
  const size_t fromX, toX;
  const size_t fromY, toY;
  const size_t sizeX, sizeY;
  template <typename A, typename B, typename C, typename D>
  Range2d(A fromX, B fromY, C toX, D toY)
      : fromX(fromX), toX(toX), fromY(fromY), toY(toY), sizeX(toX - fromX), sizeY(toY - fromY) {
    assert(fromX < toX);
    assert(fromY < toY);
    assert(sizeX != 0);
    assert(sizeY != 0);
  }
  friend std::ostream &operator<<(std::ostream &os, const Range2d &d) {
    os << "Range2d{"
       << " X[" << d.fromX << "->" << d.toX << " (" << d.sizeX << ")]"
       << " Y[" << d.fromY << "->" << d.toY << " (" << d.sizeY << ")]"
       << "}";
    return os;
  }
};

//// safely offset an id<2> by j and k
// static inline sycl::id<2> offset(const sycl::id<2> idx, const int j, const int k) {
//   int jj = static_cast<int>(idx[0]) + j;
//   int kk = static_cast<int>(idx[1]) + k;
// #ifdef SYCL_DEBUG
//   // XXX only use on runtime that provides assertions, eg: CPU
//   assert(jj >= 0);
//   assert(kk >= 0);
// #endif
//   return sycl::id<2>(jj, kk);
// }

template <class F> constexpr void par_ranged1(sycl::queue &q, const Range1d &range, F functor) {
  auto event = q.parallel_for(sycl::range<1>(range.size), [=](sycl::id<1> idx) { functor(range.from + idx[0]); });
#ifdef SYNC_KERNELS
  event.wait_and_throw();
#endif
}

// delegates to parallel_for, handles flipping if enabled
template <class functorT>
static inline void par_ranged2(sycl::queue &q, const Range2d &range, functorT functor) {

#define RANGE2D_NORMAL 0x01
#define RANGE2D_LINEAR 0x02
#define RANGE2D_ROUND 0x04

#ifndef RANGE2D_MODE
  #error "RANGE2D_MODE not set"
#endif


#if RANGE2D_MODE == RANGE2D_NORMAL
  auto event = q.parallel_for(sycl::range<2>(range.sizeX, range.sizeY), [=](sycl::id<2> idx) {
    functor(idx[0] + range.fromX, idx[1] + range.fromY);
  });
#elif RANGE2D_MODE == RANGE2D_LINEAR
  auto event = q.parallel_for(sycl::range<1>(range.sizeX * range.sizeY), [=](sycl::id<1> id) {
    auto x = (id[0] % range.sizeX) + range.fromX;
    auto y = (id[0] / range.sizeX) + range.fromY;
    functor(x, y);
  });
#elif RANGE2D_MODE == RANGE2D_ROUND
  const size_t minBlockSize = 32;
  const size_t roundedX = range.sizeX % minBlockSize == 0
                              ? range.sizeX //
                              : ((range.sizeX + minBlockSize - 1) / minBlockSize) * minBlockSize;
  const size_t roundedY = range.sizeY % minBlockSize == 0
                              ? range.sizeY //
                              : ((range.sizeY + minBlockSize - 1) / minBlockSize) * minBlockSize;
  auto event = q.parallel_for(sycl::range<2>(roundedX, roundedY), [=](sycl::id<2> idx) {
    if (idx[0] >= range.sizeX) return;
    if (idx[1] >= range.sizeY) return;
    functor(idx[0] + range.fromX, idx[1] + range.fromY);
  });
#else
  #error "Unsupported RANGE2D_MODE"
#endif
// It's an error to not sync with USM
event.wait_and_throw();
}

} // namespace clover

using clover::Range1d;
using clover::Range2d;

#endif // CLOVERLEAF_SYCL_SYCL_UTILS_HPP