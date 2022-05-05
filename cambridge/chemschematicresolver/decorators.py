# -*- coding: utf-8 -*-
"""
Decorators
==========

Python decorators used throughout ChemSchematicResolver.

From FigureDataExtractor (<CITATION>) :-
author: Matthew Swain
email: m.swain@me.com

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import functools

import six


log = logging.getLogger(__name__)


def memoized_property(fget):
    """Decorator to create memoized properties."""
    attr_name = '_{}'.format(fget.__name__)

    @functools.wraps(fget)
    def fget_memoized(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
        return getattr(self, attr_name)
    return property(fget_memoized)


def python_2_unicode_compatible(klass):
    """Fix ``__str__``, ``__unicode__`` and ``__repr__`` methods under Python 2.

    Add this decorator to a class, then define ``__str__`` and ``__repr__`` methods that both return unicode strings.
    Under python 2, this will return encoded strings for ``__str__`` (utf-8) and ``__repr__`` (ascii), and add
    ``__unicode__`` and ``_unicode_repr`` to return the original unicode strings. Under python 3, this does nothing.
    """
    if six.PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("Define __str__() on %s to use @python_2_unicode_compatible" % klass.__name__)
        if '__repr__' not in klass.__dict__:
            raise ValueError("Define __repr__() on %s to use @python_2_unicode_compatible" % klass.__name__)
        klass.__unicode__ = klass.__str__
        klass._unicode_repr = klass.__repr__
        klass.__str__ = lambda self: self.__unicode__().encode('utf-8')
        klass.__repr__ = lambda self: self._unicode_repr().encode('ascii', errors='backslashreplace')
    return klass
