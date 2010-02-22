"""
This is the setup script.
This script is automatically run by
easy_install or by pip on the user's machine when
she installs the module from pypi.

Here is some documentation for this process.
http://docs.python.org/extending/building.html
"""

from distutils.core import setup
from distutils.core import Extension


hummusc = Extension(
        'hmmusc',
        sources=['hmmuscmodule.c'])

setup(
        # required
        name = 'hmmus',
        version = '0.0.8',
        author = 'Alex Griffing',
        author_email = 'argriffi@ncsu.edu',
        url = 'http://github.com/argriffing/hmmus',
        # recommended
        ext_modules = [hummusc],
        packages = ['hmmus'],
        description = 'Hidden Markov model stuff')
