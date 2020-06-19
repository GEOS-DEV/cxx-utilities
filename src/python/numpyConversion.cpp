#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSIO
#include <numpy/arrayobject.h>

#include "../SortedArray.hpp"
#include "../MallocBuffer.hpp"
#include "../streamIO.hpp"
#include "numpyConversion.h"

#include <typeindex>

namespace LvArray
{

int getFlags(const std::type_info& tid){
  if ( std::type_index(tid) == std::type_index(typeid(int)) )
    return NPY_INT;
  else if ( std::type_index(tid) == std::type_index(typeid(long)) )
    return NPY_LONG;
  else if ( std::type_index(tid) == std::type_index(typeid(long long)) )
    return NPY_LONGLONG;
  else if ( std::type_index(tid) == std::type_index(typeid(float)) )
    return NPY_FLOAT;
  else if ( std::type_index(tid) == std::type_index(typeid(double)) )
    return NPY_DOUBLE;
  else {
    LVARRAY_ERROR("Unsupported type");
    return -1;
  }
}


template<typename T, typename INDEX_TYPE>
PyObject * sortedArrayToNumpy(SortedArray<T, INDEX_TYPE, MallocBuffer> const & arr){
  npy_intp size = static_cast<npy_intp>(arr.size());
  if(PyArray_API == NULL)
  {
      import_array(); 
  }

  int * const dataPointer = const_cast< int * >( arr.data() );
  PyObject * ret =  PyArray_SimpleNewFromData(
    1, 
    &size, 
    getFlags(typeid(T)), 
    dataPointer );
  return ret;
}


}

#ifdef __cplusplus
extern "C" {
#endif

LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > array;

PyObject * createSortedArray(int start, int stop){
  array.clear();
  for (int i = start; i < stop; ++i)
  {
    array.insert( i );
  }
  return LvArray::sortedArrayToNumpy(array);
}

PyObject * getSortedArray(void){
  return LvArray::sortedArrayToNumpy(array);
}

void printSortedArray()
{
  std::cout << "In C++: " << array << std::endl;
}

#ifdef __cplusplus
}
#endif


