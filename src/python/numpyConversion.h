#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>


#ifdef __cplusplus
extern "C" {
#endif

PyObject * createSortedArray(int start, int stop);

void printSortedArray(void);

PyObject * getSortedArray(void);

#ifdef __cplusplus
}
#endif