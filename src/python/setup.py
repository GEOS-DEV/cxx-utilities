from distutils.core import setup, Extension
import numpy as np
import os

module1 = Extension('lvarray',
                    sources=['lvarray.c'],
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-std=c99"]
                    )

setup(name = 'lvarray_ext',
       version = '1.0',
       description = 'This is a demo package',
       ext_modules= [module1]
)
