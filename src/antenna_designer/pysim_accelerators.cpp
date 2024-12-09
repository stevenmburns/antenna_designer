#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex.h>
#include "math.h"


namespace py = pybind11;

std::complex<double> interp[] = {0+0i, 1+1i, 2+2i, 3+3i};

std::complex<double> interp_func(double r) {
  auto scaled_r = r * 4;
  auto index = static_cast<int>(scaled_r);
  auto fraction = scaled_r - index;
  return interp[index]*(1-fraction) + interp[index+1]*fraction;
}

std::complex<double> fast_interp_func(double r) {
  auto scaled_r = r * 4;
  auto index = static_cast<int>(scaled_r);
  return interp[index];
}

std::complex<double> compute_func(double r) {
  std::complex<double> jk = 1i * 2.0 * M_PI;
  if (r == 0.0) {
    return 0.0;
  } else {
    return exp(jk*r)/r;
  }
}

py::array_t<double> dist_outer_product(py::array_t<double> input0,
				       py::array_t<double> input1) {
    auto buf0 = input0.request();
    auto buf1 = input1.request();

    if (buf0.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf1.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf0.shape[1] != buf1.shape[1])
      throw std::runtime_error("Inputs must have same sized second dimension");
    
    size_t rows = buf0.shape[0];
    size_t cols = buf1.shape[0];
    size_t vsize = buf0.shape[1];

    auto result = py::array_t<double>({rows, cols});
    auto result_buf = result.request();

    double *ptr0 = static_cast<double *>(buf0.ptr);
    double *ptr1 = static_cast<double *>(buf1.ptr);
    double *result_ptr = static_cast<double *>(result_buf.ptr);

    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
	auto sumsq = 0.0;
	for (size_t k = 0; k < vsize; k++) {
	  auto diff = ptr0[i*vsize+k] - ptr1[j*vsize+k];
	  sumsq += diff*diff;
	}
        result_ptr[i*cols+j] = sqrt(sumsq);
      }
    }

    return result;
}

PYBIND11_MODULE(pysim_accelerators, m) {
    m.def("dist_outer_product", &dist_outer_product, "Compute point to point euclidean distance");
    m.def("compute_func", py::vectorize(compute_func));
    m.def("interp_func", py::vectorize(interp_func));
    m.def("fast_interp_func", py::vectorize(fast_interp_func));
}
