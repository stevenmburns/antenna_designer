#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex.h>
#include "math.h"

#include <iostream>

namespace py = pybind11;



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



std::complex<double> trapezoid_aux(double theta, double delta, double *m_center_ptr, double *n_l_endpoint_ptr, double *n_r_endpoint_ptr, double wire_radius, double k) {

  const std::complex<double> minus_jk(0, -k);

  double R;
  {
    double sumsq = 0.0;
    for (size_t kk=0; kk<3; ++kk) {
      auto diff = n_l_endpoint_ptr[kk]*(1-theta) + n_r_endpoint_ptr[kk]*theta - m_center_ptr[kk];
      sumsq += diff*diff;
    }
    R = sqrt(sumsq);
    std::cout << "R: " << R << std::endl;
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
    auto m_center_ptr = ptr0 + (2*i+1)*vsize;

    for (size_t j = 0; j < cols; j++) {

      auto n_l_endpoint_ptr = ptr1 + (2*(j+0))*vsize;
      auto n_r_endpoint_ptr = ptr1 + (2*(j+1))*vsize;

      double delta;
      {
	double sumsq = 0.0;
	for (size_t kk=0; kk<3; ++kk) {
	  auto diff = n_r_endpoint_ptr[kk] - n_l_endpoint_ptr[kk];
	  sumsq += diff*diff;
	}
	delta = sqrt(sumsq);
      }

      std::complex<double> res = 0.0;

      if (ntrap == 0) {
         res = trapezoid_aux(0.5, delta, m_center_ptr, n_l_endpoint_ptr, n_r_endpoint_ptr, wire_radius, k);
      } else {
	for(size_t kk=0; kk<ntrap+1; kk++) {
	  double theta = static_cast<double>(kk)*one_over_ntrap;
	  double coeff = one_over_2_ntrap;
	  if (kk>0 && kk<ntrap) {
	    coeff = one_over_ntrap;
	  }
	  res += coeff*trapezoid_aux(theta, delta, m_center_ptr, n_l_endpoint_ptr, n_r_endpoint_ptr, wire_radius, k);
	}
      }
      result_ptr[i*cols+j] = res;
    }
  }


  return result;
}


py::array_t<std::complex<double> > new_psi_fusion_trapezoid(
    py::array_t<double> l_endpoints,
    py::array_t<double> r_endpoints,
    double wire_radius,
    double k,
    const int ntrap
) {
  auto bufl = l_endpoints.request();
  auto bufr = r_endpoints.request();

  if (bufl.ndim != 2)
    throw std::runtime_error("l_endpoints must be dimension 2");

  if (bufr.ndim != 2)
    throw std::runtime_error("r_endpoints must be dimension 2");

  if (bufl.shape[0] != bufr.shape[0] || bufl.shape[1] != bufr.shape[1]) 
    throw std::runtime_error("l_endpoints and r_endpoints must have same shape");

  if (bufl.shape[1] != 3) 
    throw std::runtime_error("Second dimension of l_endpoints and r_endpoints must have size 3");

  size_t vsize = bufl.shape[1];
  size_t rows = bufl.shape[0];
  size_t cols = rows;

  double *ptrl = static_cast<double *>(bufl.ptr);
  double *ptrr = static_cast<double *>(bufr.ptr);

  auto result = py::array_t<std::complex<double> >({rows, cols});
  auto result_buf = result.request();

  std::complex<double> *result_ptr = static_cast<std::complex<double> *>(result_buf.ptr);

  //#pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    double c[3];
    auto lptr = ptrl + i*vsize;
    auto rptr = ptrr + i*vsize;

    for (size_t j = 0; j < vsize; j++) {
      c[j] = 0.5 * (lptr[j] + rptr[j]);
    }

    auto m_center_ptr = &c[0];

    for (size_t j = 0; j < cols; j++) {

      auto n_l_endpoint_ptr = ptrl + j*vsize;
      auto n_r_endpoint_ptr = ptrr + j*vsize;

      double delta;
      {
	double sumsq = 0.0;
	for (size_t kk=0; kk<3; ++kk) {
	  auto diff = n_r_endpoint_ptr[kk] - n_l_endpoint_ptr[kk];
	  sumsq += diff*diff;
	}
	delta = sqrt(sumsq);

      }

      std::cout << "i: " << i << " j: " << j << " delta: " << delta << std::endl;

      std::complex<double> res = 0.0;

      if (ntrap == 0) {
         res = trapezoid_aux(0.5, delta, m_center_ptr, n_l_endpoint_ptr, n_r_endpoint_ptr, wire_radius, k);
      } else {
	const double one_over_ntrap = 1.0/ntrap;
	const double one_over_2_ntrap = 0.5*one_over_ntrap;

	for(size_t kk=0; kk<ntrap+1; kk++) {
	  double theta = static_cast<double>(kk)*one_over_ntrap;
	  double coeff = one_over_2_ntrap;
	  if (kk>0 && kk<ntrap) {
	    coeff = one_over_ntrap;
	  }
	  res += coeff*trapezoid_aux(theta, delta, m_center_ptr, n_l_endpoint_ptr, n_r_endpoint_ptr, wire_radius, k);
	}
      }
      result_ptr[i*cols+j] = res;
    }
  }


  return result;
}
PYBIND11_MODULE(pysim_accelerators, m) {
    m.def("dist_outer_product", &dist_outer_product, "Compute point to point euclidean distance");
    m.def("psi_fusion_trapezoid", &psi_fusion_trapezoid, "Compute Psi (Integral) from point vectors using trapezoidal method", py::arg("input0"), py::arg("input1"), py::kw_only(), py::arg("wire_radius"), py::arg("k"), py::arg("ntrap"));
    m.def("new_psi_fusion_trapezoid", &new_psi_fusion_trapezoid, "Compute Psi (Integral) from point vectors using trapezoidal method", py::arg("l_endpoints"), py::arg("r_endpoints"), py::kw_only(), py::arg("wire_radius"), py::arg("k"), py::arg("ntrap"));
}
