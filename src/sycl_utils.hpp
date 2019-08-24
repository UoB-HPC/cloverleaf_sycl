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

//#define SYCL_DEBUG
#define SYNC_KERNELS

#define SYCL_FLIP_2D

namespace clover {

	template<typename T,
			int N,
			cl::sycl::access::mode mode>
	struct Accessor {
		typedef cl::sycl::accessor<T, N, mode, cl::sycl::access::target::global_buffer> Type;
		typedef cl::sycl::accessor<T, N, mode, cl::sycl::access::target::host_buffer> HostType;

		inline static Type from(cl::sycl::buffer<T, N> &b, cl::sycl::handler &cgh) {
			return b.template get_access<mode, cl::sycl::access::target::global_buffer>(cgh);
		}

		inline static HostType access_host(cl::sycl::buffer<T, N> &b) {
			return b.template get_access<mode>();
		}

	};

	template<typename T, int N>
	struct Buffer {

		cl::sycl::buffer<T, N> buffer;

		//	Buffer() {};

		static cl::sycl::range<N> show(cl::sycl::range<N> range) {
#ifdef SYCL_DEBUG
			if (N == 1)
				std::cout << "Buffer<" << N << ">(range1d=" << range.get(0) << ")\n";
			else if (N == 2)
				std::cout << "Buffer<" << N << ">(range2d=" << range.get(0) << "," << range.get(1) << ")\n";
#endif
			return range;
		}

		// XXX remove
		explicit Buffer(cl::sycl::buffer<T, N> &buffer) : buffer(buffer) {}

		explicit Buffer(cl::sycl::range<N> range) : buffer(show(range)) {}

		template<typename Iterator>
		explicit Buffer(Iterator begin, Iterator end) : buffer(begin, end) {}


		template<cl::sycl::access::mode mode>
		inline typename Accessor<T, N, mode>::Type
		access(cl::sycl::handler &cgh) {
#ifdef SYCL_DEBUG
			if (N == 1) std::cout << "buffer->access_1d( " << buffer.get_range().get(0) << " )\n";
			else if (N == 2)
				std::cout << "buffer->access_2d( " << buffer.get_range().get(0) << "," << buffer.get_range().get(1)
				          << " )\n";
#endif
			return Accessor<T, N, mode>::from(buffer, cgh);
		}


		template<cl::sycl::access::mode mode>
		inline typename Accessor<T, N, mode>::HostType
		access() { return Accessor<T, N, mode>::access_host(buffer); }

	};

	struct Range1d {
		const size_t from, to;
		const size_t size;
		template<typename A, typename B>
		Range1d(A from, B to) : from(from), to(to), size(to - from) {
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
		template<typename A, typename B, typename C, typename D>
		Range2d(A fromX, B fromY, C toX, D toY) :
				fromX(fromX), toX(toX), fromY(fromY), toY(toY),
				sizeX(toX - fromX), sizeY(toY - fromY) {
#ifdef SYCL_DEBUG
			std::cout << "Mk range 2d:x=(" << fromX << "->" << toX << ")"
			          << ",y= (" << fromY << "->" << toY << ")" << std::endl;
#endif
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


	static inline cl::sycl::id<2> offset(const cl::sycl::id<2> idx, const int j, const int k) {
		int jj = static_cast<int>(idx[0]) + j;
		int kk = static_cast<int>(idx[1]) + k;
#ifdef SYCL_DEBUG
		// XXX only use on runtime that provides assertions, eg: CPU
		assert(jj >= 0);
		assert(kk >= 0);
#endif
		return cl::sycl::id<2>(jj, kk);
	}


	template<typename nameT, typename functorT>
	static inline void par_ranged(cl::sycl::handler &cgh, const Range1d &range, const functorT &functor) {
#ifdef SYCL_DEBUG
			std::cout << "par_ranged 1d:x=" << range.from << "(" << range.size << ")" << std::endl;
#endif

		cgh.parallel_for<nameT>(
				cl::sycl::range<1>(range.size),
				cl::sycl::id<1>(range.from),
				functor);
	}
	template<typename nameT, class functorT>
	static inline void par_ranged(cl::sycl::handler &cgh, const Range2d &range, functorT functor) {

#ifdef SYCL_DEBUG
		std::cout << "par_ranged 2d(x=" << range.fromX << "(" << range.sizeX << ")" << ", " << range.fromY << "("
		          << range.sizeY << "))" << std::endl;
#endif

#ifdef SYCL_FLIP_2D
		cgh.parallel_for<nameT>(
				cl::sycl::range<2>(range.sizeY, range.sizeX),
				cl::sycl::id<2>(range.fromY, range.fromX),
				[=](cl::sycl::id<2> idx) {
					functor(cl::sycl::id<2>(idx[1], idx[0]));
				});
#else
		cgh.parallel_for<nameT>(
				cl::sycl::range<2>(range.sizeX, range.sizeY),
				cl::sycl::id<2>(range.fromX, range.fromY),
				functor);
#endif
	}

	template<typename T>
	static void execute(cl::sycl::queue &queue, T cgf) {

#ifdef SYCL_DEBUG
		std::cout << "Execute" << std::endl;
#endif
		try {
			queue.submit(cgf);
#if defined(SYCL_DEBUG) || defined(SYNC_KERNELS)
			queue.wait_and_throw();
#endif
		} catch (cl::sycl::device_error &e) {
			std::cerr << "[SYCL] Device error: : `" << e.what() << "`" << std::endl;
			throw e;
		} catch (cl::sycl::exception &e) {
			std::cerr << "[SYCL] Exception : `" << e.what() << "`" << std::endl;
			throw e;
		}
	}
}


#endif //CLOVERLEAF_SYCL_SYCL_UTILS_HPP
