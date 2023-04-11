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

// #define SYCL_DEBUG // enable for debugging SYCL related things, also syncs kernel calls
#define SYNC_KERNELS // enable for fully synchronous (e.g queue.wait_and_throw()) kernel calls

using namespace cl;

// this namespace houses all SYCL related abstractions
namespace clover {

// abstracts away sycl::accessor
template <typename T, int N, sycl::access::mode mode> struct Accessor {
  typedef sycl::accessor<T, N, mode, sycl::access::target::device> Type;
  typedef sycl::accessor<T, N, mode, sycl::access::target::host_buffer> HostType;

  inline static Type from(sycl::buffer<T, N> &b, sycl::handler &cgh) {
    return b.template get_access<mode, sycl::access::target::device>(cgh);
  }

  inline static Type from(sycl::buffer<T, N> &b, sycl::handler &cgh, sycl::range<N> accessRange,
                          sycl::id<N> accessOffset) {
    return b.template get_access<mode, sycl::access::target::device>(cgh, accessRange, accessOffset);
  }

  inline static HostType access_host(sycl::buffer<T, N> &b) { return b.template get_access<mode>(); }
};

// abstracts away sycl::buffer
template <typename T, int N> struct Buffer {

  sycl::buffer<T, N> buffer;

  // delegates to the corresponding buffer constructor
  explicit Buffer(sycl::range<N> range) : buffer(range) {}

  explicit Buffer(T *src, sycl::range<N> range) : buffer(src, range) {}

  // delegates to the corresponding buffer constructor
  template <typename Iterator> explicit Buffer(Iterator begin, Iterator end) : buffer(begin, end) {}

  // delegates to accessor.get_access<mode>(handler)
  template <sycl::access::mode mode> inline typename Accessor<T, N, mode>::Type access(sycl::handler &cgh) {
    return Accessor<T, N, mode>::from(buffer, cgh);
  }

  // delegates to accessor.get_access<mode>(handler)
  template <sycl::access::mode mode>
  inline typename Accessor<T, N, mode>::Type access(sycl::handler &cgh, sycl::range<N> accessRange,
                                                    sycl::id<N> accessOffset) {
    return Accessor<T, N, mode>::from(buffer, cgh, accessRange, accessOffset);
  }

  // delegates to accessor.get_access<mode>()
  // **for host buffers only**
  template <sycl::access::mode mode> inline typename Accessor<T, N, mode>::HostType access() {
    return Accessor<T, N, mode>::access_host(buffer);
  }

  template <sycl::access::mode mode> inline T *access_ptr(size_t count) {
    return buffer.template get_access<mode>(count).get_pointer();
  }
};

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

// safely offset an id<2> by j and k
static inline sycl::id<2> offset(const sycl::id<2> idx, const int j, const int k) {
  int jj = static_cast<int>(idx[0]) + j;
  int kk = static_cast<int>(idx[1]) + k;
#ifdef SYCL_DEBUG
  // XXX only use on runtime that provides assertions, eg: CPU
  assert(jj >= 0);
  assert(kk >= 0);
#endif
  return sycl::id<2>(jj, kk);
}

// delegates to parallel_for, handles flipping if enabled
template <typename nameT, class functorT>
static inline void par_ranged(sycl::handler &cgh, const Range1d &range, functorT functor) {
  cgh.parallel_for<nameT>(sycl::range<1>(range.size), [=](sycl::id<1> idx) {
    idx = sycl::id<1>(idx.get(0) + range.from);
    functor(idx);
  });
}

// delegates to parallel_for, handles flipping if enabled
template <typename nameT, class functorT>
static inline void par_ranged(sycl::handler &cgh, const Range2d &range, functorT functor) {

#define RANGE2D_NORMAL 0x01
#define RANGE2D_LINEAR 0x02
#define RANGE2D_ROUND 0x04

#ifndef RANGE2D_MODE
  #error "RANGE2D_MODE not set"
#endif


#if RANGE2D_MODE == RANGE2D_NORMAL
  cgh.parallel_for<nameT>(sycl::range<2>(range.sizeX, range.sizeY), [=](sycl::id<2> idx) {
    idx = sycl::id<2>(idx[0] + range.fromX, idx[1] + range.fromY);
    functor(idx);
  });
#elif RANGE2D_MODE == RANGE2D_LINEAR
  cgh.parallel_for<nameT>(sycl::range<1>(range.sizeX * range.sizeY), [=](sycl::id<1> id) {
    const auto x = (id[0] / range.sizeY) + range.fromX;
    const auto y = (id[0] % range.sizeY) + range.fromY;
    functor(sycl::id<2>(x, y));
  });
#elif RANGE2D_MODE == RANGE2D_ROUND
  const size_t minBlockSize = 32;
  const size_t roundedX = range.sizeX % minBlockSize == 0
                              ? range.sizeX //
                              : ((range.sizeX + minBlockSize - 1) / minBlockSize) * minBlockSize;
  const size_t roundedY = range.sizeY % minBlockSize == 0
                              ? range.sizeY //
                              : ((range.sizeY + minBlockSize - 1) / minBlockSize) * minBlockSize;
  cgh.parallel_for<nameT>(sycl::range<2>(roundedX, roundedY), [=](sycl::id<2> idx) {
    if (idx[0] >= range.sizeX) return;
    if (idx[1] >= range.sizeY) return;
    idx = sycl::id<2>(idx[0] + range.fromX, idx[1] + range.fromY);
    functor(idx);
  });
#else
  #error "Unsupported RANGE2D_MODE"
#endif
}

// delegates to queue.submit(cgf), handles blocking submission if enable
template <typename T> static void execute(sycl::queue &queue, T cgf) {
  try {
    queue.submit(cgf);
#if defined(SYCL_DEBUG) || defined(SYNC_KERNELS)
    queue.wait_and_throw();
#endif
  } catch (sycl::exception &e) {
    std::cerr << "[SYCL] Exception : `" << e.what() << "`" << std::endl;
    throw e;
  }
}
} // namespace clover

#endif // CLOVERLEAF_SYCL_SYCL_UTILS_HPP
