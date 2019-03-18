#ifndef __NDARRAY_CONVERTER_H__
#define __NDARRAY_CONVERTER_H__

#include <Python.h>
#include <opencv2/core/core.hpp>

class NDArrayConverter {
 public:
  // must call this first, or the other routines don't work!
  static bool init_numpy();

  static bool toMat(PyObject *o, cv::Mat &m);
  static PyObject *toNDArray(const cv::Mat &mat);
};

//
// Define the type converter
//

#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {

template <>
struct type_caster<cv::Mat> {
 public:
  PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));

  bool load(handle src, bool) { return NDArrayConverter::toMat(src.ptr(), value); }

  static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
    return handle(NDArrayConverter::toNDArray(m));
  }
};

/*
template <typename K>
bool setItem(const PyObject *item, K &k);
*/

template <typename T, int N>
struct type_caster<cv::Vec<T, N>> {
 public:
  // Alias the type
  using Vec = cv::Vec<T, N>;

  /**
   * This macro declares a local variable 'value' of type cv::Vec3b
   */
  PYBIND11_TYPE_CASTER(Vec, _("tuple"));

  // Convert from python to c++
  bool load(handle src, bool) {
    auto tuple = pybind11::reinterpret_borrow<pybind11::tuple>(src);
    for (auto index = 0; index < N; index++) {
      value[index] = tuple[index].cast<T>();
    }
    return true;
  }

  // Convert Vec into a tuple
  static handle cast(const Vec &src, return_value_policy /* policy */, handle /* parent */) {
    pybind11::tuple mytuple(N);
    for (auto index = 0; index < N; index++) {
      mytuple[index] = T(src[index]);
    }
    return mytuple.release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif
