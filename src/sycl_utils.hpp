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
		if (N == 1)
			std::cout << "Buffer<" << N << ">(range1d=" << range.get(0) << ")\n";
		else if (N == 2)
			std::cout << "Buffer<" << N << ">(range2d=" << range.get(0) << "," << range.get(1) << ")\n";
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

		if (N == 1) std::cout << "buffer->access_1d( " << buffer.get_range().get(0) << " )\n";
		else if (N == 2)
			std::cout << "buffer->access_2d( " << buffer.get_range().get(0) << "," << buffer.get_range().get(1)
			          << " )\n";
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
};

struct Range2d {
	const size_t fromX, toX;
	const size_t fromY, toY;
	const size_t sizeX, sizeY;
	template<typename A, typename B, typename C, typename D>
	Range2d(A fromX, B fromY, C toX, D toY) :
			fromX(fromX), toX(toX), fromY(fromY), toY(toY),
			sizeX(toX - fromX), sizeY(toY - fromY) {
		if (DEBUG)
			std::cout << "Mk range 2d:x=(" << fromX << "->" << toX << ")"
			          << ",y= (" << fromY << "->" << toY << ")" << std::endl;
		assert(fromX < toX);
		assert(fromY < toY);
		assert(sizeX != 0);
		assert(sizeY != 0);
	}
};

template<typename nameT, typename functorT>
inline void par_ranged(cl::sycl::handler &cgh, const Range1d &range, const functorT &functor) {
	if (DEBUG)
		std::cout << "par_ranged 1d:x=" << range.from << "(" << range.size << ")" << std::endl;
	cgh.parallel_for<nameT>(
			cl::sycl::range<1>(range.size),
			cl::sycl::id<1>(range.from),
			functor);
}
template<typename nameT, typename functorT>
inline void par_ranged(cl::sycl::handler &cgh, const Range2d &range, const functorT &functor) {
	if (DEBUG)
		std::cout << "par_ranged 2d(x=" << range.fromX << "(" << range.sizeX << ")" << ", " << range.fromY << "("
		          << range.sizeY << "))" << std::endl;
	cgh.parallel_for<nameT>(
			cl::sycl::range<2>(range.sizeX, range.sizeY),
			cl::sycl::id<2>(range.fromX, range.fromY),
			functor);
}
template<typename T>
inline void execute(cl::sycl::queue &queue, T cgf) {


	if (DEBUG) std::cout << "Execute" << std::endl;

	try {
		queue.submit(cgf);
		queue.wait_and_throw();
	} catch (cl::sycl::device_error &e) {
		std::cerr << "Execution failed: `" << e.what() << "`" << std::endl;
		throw e;
	}
}



#endif //CLOVERLEAF_SYCL_SYCL_UTILS_HPP
