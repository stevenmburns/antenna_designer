#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex.h>
#include "math.h"

#include <iostream>

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
    return exp(-jk*r)/(4*M_PI*r);
  }
}

py::array_t<std::complex<double> > psi(py::array_t<double> input, py::array_t<double> delta, double wire_radius, double k) {
  auto buf = input.request();
  auto bufd = delta.request();

  if (buf.shape[0] != bufd.shape[0])
    throw std::runtime_error("Number of rows in input must be same as number of elements in delta");

  size_t rows = buf.shape[0];
  size_t cols = buf.shape[1];

  auto result = py::array_t<std::complex<double> >({rows, cols});
  auto result_buf = result.request();
  
  double *ptr = static_cast<double *>(buf.ptr);
  double *ptrd = static_cast<double *>(bufd.ptr);
  std::complex<double> *result_ptr = static_cast<std::complex<double> *>(result_buf.ptr);

  std::complex<double> minus_jk = -1i*k;

  #pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      auto R = ptr[i*cols+j];

      std::complex<double> res;
      if (R == 0.0) {
	res = 1.0/(2.0*M_PI*ptrd[i]) * log(ptrd[i]/wire_radius) + minus_jk/(4.0*M_PI);
      } else {
	res = exp(minus_jk*R)/(4*M_PI*R);
      }

      result_ptr[i*cols+j] = res;
    }
  }


  return result;
}

py::array_t<std::complex<double> > psi_fusion(
    py::array_t<double> input0,
    py::array_t<double> input1,
    py::array_t<double> delta,
    double wire_radius,
    double k
) {
  auto buf0 = input0.request();
  auto buf1 = input1.request();
  auto bufd = delta.request();

    if (buf0.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf1.ndim != 2)
      throw std::runtime_error("Number of dimensions must be two");

    if (buf0.shape[1] != buf1.shape[1])
      throw std::runtime_error("Inputs must have same sized second dimension");

  if (buf0.shape[0] != bufd.shape[0])
    throw std::runtime_error("Number of rows in input must be same as number of elements in delta");

  size_t rows = buf0.shape[0];
  size_t cols = buf1.shape[0];
  size_t vsize = buf0.shape[1];

  auto result = py::array_t<std::complex<double> >({rows, cols});
  auto result_buf = result.request();
  
  double *ptr0 = static_cast<double *>(buf0.ptr);
  double *ptr1 = static_cast<double *>(buf1.ptr);

  double *ptrd = static_cast<double *>(bufd.ptr);
  std::complex<double> *result_ptr = static_cast<std::complex<double> *>(result_buf.ptr);

  std::complex<double> minus_jk = -1i*k;

  #pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      auto sumsq = 0.0;
      for (size_t kk = 0; kk < vsize; kk++) {
	auto diff = ptr0[i*vsize+kk] - ptr1[j*vsize+kk];
	sumsq += diff*diff;
      }

      auto R = sqrt(sumsq);

      std::complex<double> res;
      if (R == 0.0) {
	res = 1.0/(2.0*M_PI*ptrd[i]) * log(ptrd[i]/wire_radius) + minus_jk/(4.0*M_PI);
      } else {
	res = exp(minus_jk*R)/(4*M_PI*R);
      }

      result_ptr[i*cols+j] = res;
    }
  }


  return result;
}

std::complex<double> trapezoid_aux(double theta, double delta, double *n_l_endpoint_ptr, double *n_r_endpoint_ptr, double *m_center_ptr, double wire_radius, double k) {

  std::complex<double> minus_jk = -1i*k;

  double R;
  {
    double sumsq = 0.0;
    for (size_t kk=0; kk<3; ++kk) {
      auto diff = n_l_endpoint_ptr[kk]*(1-theta) + n_r_endpoint_ptr[kk]*theta - m_center_ptr[kk];
      sumsq += diff*diff;
    }
    R = sqrt(sumsq);
  }

  std::complex<double> res;
  if (R < 0.00001) {
    res = 1.0/(2.0*M_PI*delta) * log(delta/wire_radius) + minus_jk/(4.0*M_PI);
  } else {
    res = exp(minus_jk*R)/(4*M_PI*R);
  }

  return res;
}


py::array_t<std::complex<double> > psi_fusion_trapezoid(
    py::array_t<double> input0,
    py::array_t<double> input1,
    double wire_radius,
    double k,
    int ntrap
) {
  auto buf0 = input0.request();
  auto buf1 = input1.request();

  if (buf0.ndim != 2)
    throw std::runtime_error("Number of dimensions must be two");

  if (buf1.ndim != 2)
    throw std::runtime_error("Number of dimensions must be two");

  if (buf0.shape[0] % 2 != 1)
    throw std::runtime_error("Input0 must have odd first dimension size");

  if (buf1.shape[0] % 2 != 1)
    throw std::runtime_error("Input1 must have odd first dimension size");

  if (buf0.shape[1] != buf1.shape[1])
    throw std::runtime_error("Inputs must have same sized second dimension");

  size_t rows = (buf0.shape[0]-1)/2;
  size_t cols = (buf1.shape[0]-1)/2;
  size_t vsize = buf0.shape[1];

  auto result = py::array_t<std::complex<double> >({rows, cols});
  auto result_buf = result.request();
  
  double *ptr0 = static_cast<double *>(buf0.ptr);
  double *ptr1 = static_cast<double *>(buf1.ptr);

  std::complex<double> *result_ptr = static_cast<std::complex<double> *>(result_buf.ptr);

  double one_over_ntrap = 1.0/ntrap;
  double one_over_2_ntrap = 0.5*one_over_ntrap;

  #pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    auto n_l_endpoint_ptr = ptr0 + (2*(i+0))*vsize;
    auto n_r_endpoint_ptr = ptr0 + (2*(i+1))*vsize;

    double delta;
    {
      double sumsq = 0.0;
      for (size_t kk=0; kk<3; ++kk) {
        auto diff = n_r_endpoint_ptr[kk] - n_l_endpoint_ptr[kk];
	sumsq += diff*diff;
      }
      delta = sqrt(sumsq);
    }

    for (size_t j = 0; j < cols; j++) {

      auto m_center_ptr = ptr1 + (2*j+1)*vsize;

      std::complex<double> res = 0.0;

      if (ntrap == 0) {
         res = trapezoid_aux(0.5, delta, n_l_endpoint_ptr, n_r_endpoint_ptr, m_center_ptr, wire_radius, k);
      } else {
	for(size_t kk=0; kk<ntrap+1; kk++) {
	  double theta = static_cast<double>(kk)*one_over_ntrap;
	  double coeff = one_over_2_ntrap;
	  if (kk>0 && kk<ntrap) {
	    coeff = one_over_ntrap;
	  }
	  res += coeff*trapezoid_aux(theta, delta, n_l_endpoint_ptr, n_r_endpoint_ptr, m_center_ptr, wire_radius, k);
	}
      }
      result_ptr[i*cols+j] = res;
    }
  }


  return result;
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
    m.def("psi", &psi, "Compute Psi (Integral) from euclidean distance",
	  py::arg("R"), py::arg("delta"), py::kw_only(), py::arg("wire_radius"), py::arg("k"));
    m.def("psi_fusion", &psi_fusion, "Compute Psi (Integral) from point vectors", py::arg("input0"), py::arg("input1"), py::arg("delta"), py::kw_only(), py::arg("wire_radius"), py::arg("k"));
    m.def("psi_fusion_trapezoid", &psi_fusion_trapezoid, "Compute Psi (Integral) from point vectors using trapezoidal method", py::arg("input0"), py::arg("input1"), py::kw_only(), py::arg("wire_radius"), py::arg("k"), py::arg("ntrap"));
    m.def("compute_func", py::vectorize(compute_func));
    m.def("interp_func", py::vectorize(interp_func));
    m.def("fast_interp_func", py::vectorize(fast_interp_func));
}
