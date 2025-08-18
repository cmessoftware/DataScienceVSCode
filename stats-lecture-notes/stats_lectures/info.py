""" This file contains defines parameters for stats_lectures that we use to fill
settings in setup.py, the stats_lectures top-level docstring, and for building the docs.
In setup.py in particular, we exec this file, so it cannot import stats_lectures
"""

# stats_lectures version information.  An empty _version_extra corresponds to a
# full release.  '.dev' as a _version_extra string means this is a development
# version
_version_major = 0
_version_minor = 0
_version_micro = 1
_version_extra = '.dev'

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
__version__ = "%s.%s.%s%s" % (_version_major,
                              _version_minor,
                              _version_micro,
                              _version_extra)

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description  = 'A multi-algorithm Python framework for regularized regression'

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = \
"""
==============
stats_lectures
==============

stats_lectures is a small package of helper functions
and data that is used in the notebooks for statistics lectures found at:

http://github.com/jonathan-taylor/stats-lecture-notes

"""

# versions
NUMPY_MIN_VERSION='1.3'
SCIPY_MIN_VERSION = '0.5'
CYTHON_MIN_VERSION = '0.11.1'

NAME                = 'stats_lectures'
MAINTAINER          = "Jonathan Taylor"
MAINTAINER_EMAIL    = ""
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://github.com/jonathan-taylor/stats-lecture-notes"
DOWNLOAD_URL        = ""
LICENSE             = "BSD license"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Jonathan Taylor"
AUTHOR_EMAIL        = ""
PLATFORMS           = "OS Independent"
MAJOR               = _version_major
MINOR               = _version_minor
MICRO               = _version_micro
ISRELEASE           = _version_extra == ''
VERSION             = __version__
STATUS              = 'alpha'
PROVIDES            = ["stats_lectures"]
REQUIRES            = ["numpy (>=%s)" % NUMPY_MIN_VERSION,
                       "scipy (>=%s)" % SCIPY_MIN_VERSION,
                       'ystockquote']
