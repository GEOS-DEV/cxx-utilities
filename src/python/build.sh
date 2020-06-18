#!/bin/bash

set -e

/usr/tce/packages/clang/clang-10.0.0/bin/clang -pthread -Wno-unused-result -Wsign-compare -g -fwrapv -Wall -fPIC -fPIC -I/usr/tce/packages/python/python-3.7.2/lib/python3.7/site-packages/numpy/core/include -I/usr/workspace/corbett8/cxx-utilities/build-quartz-clang@10.0.0-debug/include -I/usr/workspace/corbett8/RAJA/install-quartz-clang@10.0.0-release/include -I/usr/tce/packages/python/python-3.7.2/include/python3.7m -I/usr/workspace/corbett8/cxx-utilities/src/ -c lvarray.c -o lvarray.o -fopenmp

/usr/tce/packages/clang/clang-10.0.0/bin/clang++ -pthread -Wno-unused-result -Wsign-compare -g -fwrapv -Wall -fPIC -fPIC -I/usr/tce/packages/python/python-3.7.2/lib/python3.7/site-packages/numpy/core/include -I/usr/workspace/corbett8/cxx-utilities/build-quartz-clang@10.0.0-debug/include -I/usr/workspace/corbett8/cxx-utilities/src -I/usr/workspace/corbett8/RAJA/install-quartz-clang@10.0.0-release/include -I/usr/tce/packages/python/python-3.7.2/include/python3.7m -c numpyConversion.cpp -o numpyConversion.o -std=c++14 -fopenmp

/usr/tce/packages/clang/clang-10.0.0/bin/clang++ -pthread -Wno-unused-result -Wsign-compare -g -fwrapv -Wall -fPIC -fPIC -I/usr/workspace/corbett8/cxx-utilities/src -I/usr/workspace/corbett8/cxx-utilities/build-quartz-clang@10.0.0-debug/include -I/usr/workspace/corbett8/RAJA/install-quartz-clang@10.0.0-release/include -c ../stackTrace.cpp -o stackTrace.o -std=c++14 -fopenmp

/usr/tce/packages/clang/clang-10.0.0/bin/clang++ -pthread -shared lvarray.o numpyConversion.o stackTrace.o -L/collab/usr/gapps/python/build/spack-toss3.3/opt/spack/linux-rhel7-x86_64/gcc-4.9.3/python-3.7.2-asydydmavj2puklmx5t6cu3ruzmg2b3a/lib -lpython3.7m -o lvarray.cpython-37m-x86_64-linux-gnu.so
