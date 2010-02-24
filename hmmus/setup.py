"""
This is the setup script.
This script is automatically run by
easy_install or by pip on the user's machine when
she installs the module from pypi.

Here is some documentation for this process.
http://docs.python.org/extending/building.html

More info:
http://wiki.python.org/moin/Distutils/Tutorial

Register the metadata with pypi as follows:
python setup.py register

Send to pypi as follows:
python setup.py sdist upload --show-response
"""

from distutils.core import setup
from distutils.core import Extension

hummusc = Extension(
        name='hmmusc',
        sources=['hmmuscmodule.c', 'hmmguts/hmmguts.c'])

classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: Posix :: Linux',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: C',
        'Programming Language :: Unix Shell',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
        'Topic :: Scientific/Engineering :: Information Analysis']

scripts = [
        'bin/create-example-likelihoods-a.py',
        'bin/create-example-likelihoods-b.py',
        'bin/create-example-likelihoods-c.py',
        'bin/view-matrix.py']

setup(
        name = 'hmmus',
        version = '0.1.0',
        author = 'Alex Griffing',
        author_email = 'argriffi@ncsu.edu',
        maintainer = 'Alex Griffing',
        maintainer_email = 'argriffi@ncsu.edu',
        url = 'http://github.com/argriffing/hmmus',
        description = 'Hidden Markov model stuff',
        classifiers = classifiers,
        platforms = ['linux'],
        license = 'http://www.opensource.org/licenses/mit-license.html',
        ext_modules = [hummusc],
        packages = ['hmmus'],
        scripts = scripts)
