#include <CL/sycl.hpp>

#include <array>
#include <iostream>

using cl::sycl::accessor;

constexpr cl::sycl::access::mode R = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode W = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode RW = cl::sycl::access::mode::read_write;

template<typename T,
		int N,
		cl::sycl::access::mode mode,
		cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
struct Accessor {
	typedef cl::sycl::accessor<T, N, mode, target> View;

	inline static View from(cl::sycl::buffer<T, N> &b, cl::sycl::handler &cgh) {
		return b.template get_access<mode, target>(cgh);
	}

};


template<typename T, int N>
struct Buffer {

	cl::sycl::buffer<T, N> &buffer;

	explicit Buffer(cl::sycl::buffer<T, N> &buffer) : buffer(buffer) {}

	template<cl::sycl::access::mode mode,
			cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
	inline typename Accessor<T, N, mode, target>::View
	access(cl::sycl::handler &cgh) {
		return Accessor<T, N, mode, target>::from(buffer, cgh);
	}

};


template<typename T>
class SimpleVadd;


template<typename T, size_t N>
struct Data2 {
	cl::sycl::buffer<T, 2> ab;
	cl::sycl::buffer<T, 1> c;
	Data2(const std::array<T, 2 * N> &VA, std::array<T, N> &VC) :
			ab(VA.data(), cl::sycl::range<2>(2, N)),
			c(VC.data(), cl::sycl::range<1>(N)) {}
};

using namespace std::placeholders;

template<typename T, size_t N>
void doIt(
		const cl::sycl::stream &os,
		cl::sycl::id<1> &wiID,
		const typename Accessor<T, 2, R>::View &accessorAB,
		const typename Accessor<T, 1, W>::View &accessorC) {
	auto i = wiID[0];
	auto b = accessorAB[0][i] + accessorAB[1][i];
	os << i << ", " << wiID[1] << "\n";
	accessorC[wiID] = b * 10;
}

template<typename T, size_t N>
void simple_vadd(const std::array<T, 2 * N> &VA,
                 std::array<T, N> &VC) {
	cl::sycl::queue deviceQueue;
	cl::sycl::range<1> numOfItems{N};


//	Data2<T, N> data(VA, VC);



	auto cb = cl::sycl::buffer<T, 1>(VC.data(), cl::sycl::range<1>(N));
	auto abb = cl::sycl::buffer<T, 2>(VA.data(), cl::sycl::range<2>(2, N));

	Buffer<T, 1> c = Buffer<T, 1>(cb);
	Buffer<T, 2> ab = Buffer<T, 2>(abb);


	deviceQueue.submit([&](cl::sycl::handler &cgh) {


		typename Accessor<T, 1, W>::View accessorC = c.template access<W>(cgh);
		typename Accessor<T, 2, R>::View accessorAB = ab.template access<R>(cgh);


		cl::sycl::stream os(1024, 128, cgh);

		auto kern = [=](cl::sycl::id<1> wiID) {




			accessorC[wiID] = -1;
//			doIt<T, N>(os, wiID, accessorAB, accessorC);
			auto i = wiID[0];
			auto b = accessorAB[0][i] + accessorAB[1][i];
			os << i << ", " << wiID[1] << "\n";
////			auto a = accessorAB[0][wiID.];
			accessorC[wiID] = b;
//			accessorC[wiID] = accessorAB[wiID] + accessorAB[wiID] + 1;
		};
		cgh.parallel_for<class SimpleVadd<T>>(cl::sycl::range<1>(N), kern);


//		cgh.parallel_for<class SimpleVadd<T>>(cl::sycl::range<1>(N),
//		                                      std::bind(doIt<T, N>,
//		                                                std::cref(os),
//		                                                _1,
//		                                                std::cref(accessorAB),
//		                                                std::cref(accessorC))
//		);
	});
}

int main() {
	std::array<cl::sycl::cl_int, 8> A = {
			{
					1, 2, 3, 4,
					4, 3, 2, 1
			}
	};
	std::array<cl::sycl::cl_int, 4> Out = {};
	simple_vadd(A, Out);

	for (const auto &x : Out) {
		std::cout << "=" << x << std::endl;
	}

	std::cout << "Done\n";
	return EXIT_SUCCESS;
}
