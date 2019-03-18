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

#if 1
template <>
struct type_caster<cv::Vec3b> {
 public:
  /**
   * This macro declares a local variable 'value' of type cv::Vec3b
   */
  PYBIND11_TYPE_CASTER(cv::Vec3b, _("tuple"));

  // Convert from python to c++
  bool load(handle src, bool) {  //

    constexpr auto N = 3;
    for (auto index = 0; index < N; index++) {
      PyObject *item = PyTuple_GetItem(src.ptr(), static_cast<ssize_t>(index));
      if (!item) {
        return false;
      }
      PyObject *tmp = PyNumber_Long(item);
      if (!tmp) {
        return false;
      }
      // Now try to convert into a C++ int
      value[index] = PyLong_AsLong(tmp);
      Py_DECREF(tmp);
      if (PyErr_Occurred()) {
        return false;
      }
    }

    return true;
  }

  // Convert Vec3b into a tuple
  static handle cast(const cv::Vec3b &src, return_value_policy /* policy */, handle /* parent */) {
    constexpr auto N = 3;
    auto object = PyTuple_New(N);
    for (auto index = 0; index < N; index++) {
      auto item = PyLong_FromLong(src[index]);
      PyTuple_SetItem(object, static_cast<ssize_t>(index), item);
    }
    return handle(object);
  }
};
#endif
}  // namespace detail
}  // namespace pybind11

#endif
