"""
This is the setup script.
This script is automatically run by
easy_install or by pip on the user's machine when
she installs the module from pypi.

Here is some documentation for this process.
http://docs.python.org/extending/building.html

Here is some info about the cython part of this process.
http://docs.cython.org/src/quickstart/build.html

Right now I'm sending this to pypi in two steps.
python setup.py build_ext --inplace
python setup.py sdist upload --show-response
The first step is required to cythonize the hello.pyx
file into the hello.c file.
Note that both hello.c and hello.pyx are
included in the MANIFEST.in file.
"""

from distutils.core import setup
from distutils.core import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    print "cython not found, using previously-cython'd .c file."
    hello_sources = ['hello.c']
else:
    hello_sources = ['hello.pyx']

hello = Extension(
        name='hello',
        sources=hello_sources)

hummusc = Extension(
        name='hmmusc',
        sources=['hmmuscmodule.c'])

setup(
        name = 'hmmus',
        version = '0.0.18',
        author = 'Alex Griffing',
        author_email = 'argriffi@ncsu.edu',
        url = 'http://github.com/argriffing/hmmus',
        cmdclass = {'build_ext': build_ext},
        ext_modules = [hummusc, hello],
        packages = ['hmmus'],
        description = 'Hidden Markov model stuff')
