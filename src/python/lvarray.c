#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "numpyConversion.h"
 
static PyObject *
create_np_array(PyObject *self, PyObject *args)
{
    int start;
    int stop;
    int * dataPointer;
    int index = 0;
    if (!PyArg_ParseTuple(args, "ii", &start, &stop))
        return NULL;
    if ( stop <= start ){
        Py_RETURN_NONE;
    }
    npy_intp size = stop - start;
    dataPointer = malloc(size * sizeof(int));
    for ( int fill_value = start; fill_value < stop; fill_value++ )
    {
        dataPointer[ index ] = fill_value;
        index += 1;
    }
    return PyArray_NewFromDescr(
        &PyArray_Type,
        PyArray_DescrFromType(NPY_INT),
        1,
        &size,
        NULL,
        dataPointer,
        NPY_ARRAY_OWNDATA,
        NULL
    );
}


static PyObject *
print_array(PyObject *self, PyObject *args)
{
    printSortedArray();
    Py_RETURN_NONE;
}


static PyObject *
set_sorted_array(PyObject *self, PyObject *args)
{
    int start;
    int stop;
    if (!PyArg_ParseTuple(args, "ii", &start, &stop))
        return NULL;
    if ( stop <= start ){
        Py_RETURN_NONE;
    }
    return createSortedArray(start, stop);
}


static PyObject *
get_sorted_array(PyObject *self, PyObject *args)
{
    return getSortedArray();
}


static PyMethodDef TestingMethods[] = {
    {"create_np_array",  create_np_array, METH_VARARGS,
     "Create a numpy array initialized like `range(start, stop)`."},
    {"set_sorted_array",  set_sorted_array, METH_VARARGS,
     "Return the numpy representation of a SortedArray initialized like `range(start, stop)`."},
    {"get_sorted_array",  get_sorted_array, METH_NOARGS,
     "Get the numpy representation of a SortedArray."},
    {"print_array",  print_array, METH_NOARGS,
     "Print the numpy representation of a SortedArray."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static struct PyModuleDef _lvarraymodule = {
    PyModuleDef_HEAD_INIT,
    "lvarray",   /* name of module */
    "Module for testing lvarray conversions", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    TestingMethods
};


PyMODINIT_FUNC
PyInit_lvarray(void)
{
    import_array();
    return PyModule_Create(&_lvarraymodule);
}