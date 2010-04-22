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
import os

from hmmus import hmmus_version


scripts = [
        'bin/hmm-od',
        'bin/hmm-regression-test',
        'bin/hmm-demo',
        'bin/hmm-forward',
        'bin/hmm-backward',
        'bin/hmm-posterior',
        'bin/hmm-fwdbwd-somedisk',
        'bin/hmm-fwdbwd-alldisk']

hummusc = Extension(
        name='hmmusc',
        sources=['hmmuscmodule.c', 'hmmguts/hmmguts.c'])

classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: C',
        'Programming Language :: Unix Shell',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules']

download_url_first = 'http://pypi.python.org/packages/source'
download_url_rest = 'h/hmmus/hmmus-' + hmmus_version + '.tar.gz'
download_url = os.path.join(download_url_first, download_url_rest)

# This seems like a standard method.
long_description = open('README.rst').read()

setup(
        name = 'hmmus',
        version = hmmus_version,
        author = 'Alex Griffing',
        author_email = 'argriffi@ncsu.edu',
        maintainer = 'Alex Griffing',
        maintainer_email = 'argriffi@ncsu.edu',
        url = 'http://github.com/argriffing/hmmus',
        download_url = download_url,
        description = 'Posterior decoding with a hidden Markov model',
        long_description = long_description,
        classifiers = classifiers,
        platforms = ['Linux'],
        license = 'MIT',
        ext_modules = [hummusc],
        packages = ['hmmus'],
        scripts = scripts)
