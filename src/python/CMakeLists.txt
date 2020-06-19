set(lvarrayPythonInterface_headers
    numpyConversion.h
   )

set(lvarrayPythonInterface_sources
    numpyConversion.cpp
    lvarrayPythonInterface.c
    )

blt_add_library(NAME lvarrayPythonInterface
                SOURCES ${lvarrayPythonInterface_sources}
                HEADERS ${lvarrayPythonInterface_headers}
                DEPENDS_ON lvarray Python3::Python Python3::NumPy
                SHARED TRUE
                CLEAR_PREFIX TRUE
                )

target_include_directories(lvarray PUBLIC ${CMAKE_BINARY_DIR}/include)
