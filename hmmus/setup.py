"""
This is the setup script.
This script is automatically run by
easy_install or by pip on the user's machine when
she installs the module from pypi.

Here is some documentation for this process.
http://docs.python.org/extending/building.html

Send to pypi as follows:
python setup.py sdist upload --show-response
"""

from distutils.core import setup
from distutils.core import Extension

hummusc = Extension(
        name='hmmusc',
        sources=['hmmuscmodule.c', 'hmmguts/hmmguts.c'])

scripts = [
        'bin/create-example-likelihoods-a.py',
        'bin/create-example-likelihoods-b.py',
        'bin/create-example-likelihoods-c.py',
        'bin/view-matrix.py']

setup(
        name = 'hmmus',
        version = '0.0.36',
        author = 'Alex Griffing',
        author_email = 'argriffi@ncsu.edu',
        url = 'http://github.com/argriffing/hmmus',
        ext_modules = [hummusc],
        packages = ['hmmus'],
        scripts = scripts,
        description = 'Hidden Markov model stuff')
