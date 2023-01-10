__copyright__ = "Copyright (C) 2011 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np


def f_contiguous_strides(itemsize, shape):
    if shape:
        strides = [itemsize]
        for s in shape[:-1]:
            # NOTE: max(1, s) is used to handle 0-sized axes in `shape`;
            # the stride for `shape[i] <= 1` doesn't matter, but letting it be 0
            # is not a good idea: https://github.com/inducer/arraycontext/pull/91
            strides.append(strides[-1]*max(1, s))
        return tuple(strides)
    else:
        return ()


def c_contiguous_strides(itemsize, shape):
    if shape:
        strides = [itemsize]
        for s in shape[:0:-1]:
            # NOTE: max(1, s) is used to handle 0-sized axes in `shape`;
            # the stride for `shape[i] <= 1` doesn't matter, but letting it be 0
            # is not a good idea: https://github.com/inducer/arraycontext/pull/91
            strides.append(strides[-1]*max(1, s))
        return tuple(strides[::-1])
    else:
        return ()


def equal_strides(strides1, strides2, shape):
    if strides1 == strides2:
        return True

    if len(strides1) != len(strides2) or len(strides2) != len(shape):
        return False

    for s, st1, st2 in zip(shape, strides1, strides2):
        if s != 1 and st1 != st2:
            return False

    return True


def is_f_contiguous_strides(strides, itemsize, shape):
    from pytools import product
    return (
            equal_strides(strides, f_contiguous_strides(itemsize, shape), shape)
            or product(shape) == 0)  # noqa: W503


def is_c_contiguous_strides(strides, itemsize, shape):
    from pytools import product
    return (equal_strides(strides, c_contiguous_strides(itemsize, shape), shape)
            or product(shape) == 0)  # noqa: W503


class ArrayFlags:
    def __init__(self, ary):
        self.f_contiguous = is_f_contiguous_strides(
            ary.strides, ary.dtype.itemsize, ary.shape)
        self.c_contiguous = is_c_contiguous_strides(
            ary.strides, ary.dtype.itemsize, ary.shape)
        self.forc = self.f_contiguous or self.c_contiguous

    def __repr__(self):
        return (
                f"  C_CONTIGUOUS : {self.c_contiguous}\n"
                f"  F_CONTIGUOUS : {self.f_contiguous}"
                )

    def __str__(self):
        return repr(self)


def get_common_dtype(obj1, obj2, allow_double):
    # Yes, numpy behaves differently depending on whether
    # we're dealing with arrays or scalars.

    zero1 = np.zeros(1, dtype=obj1.dtype)

    try:
        zero2 = np.zeros(1, dtype=obj2.dtype)
    except AttributeError:
        zero2 = obj2

    result = (zero1 + zero2).dtype

    if not allow_double:
        if result == np.float64:
            result = np.dtype(np.float32)
        elif result == np.complex128:
            result = np.dtype(np.complex64)

    return result


def bound(a):
    high = a.bytes
    low = a.bytes

    for stri, shp in zip(a.strides, a.shape):
        if stri < 0:
            low += (stri)*(shp-1)
        else:
            high += (stri)*(shp-1)
    return low, high


def may_share_memory(a, b):
    # When this is called with a an ndarray and b
    # a sparse matrix, numpy.may_share_memory fails.
    if a is b:
        return True
    if a.__class__ is b.__class__:
        a_l, a_h = bound(a)
        b_l, b_h = bound(b)
        if b_l >= a_h or a_l >= b_h:
            return False
        return True
    else:
        return False


# {{{ as_strided implementation

try:
    from numpy.lib.stride_tricks import as_strided as _as_strided
    _test_dtype = np.dtype(
            [("a", np.float64), ("b", np.float64)], align=True)
    _test_result = _as_strided(np.zeros(10, dtype=_test_dtype))
    if _test_result.dtype != _test_dtype:
        raise RuntimeError("numpy's as_strided is broken")

    as_strided = _as_strided
except Exception:
    # stolen from numpy to be compatible with older versions of numpy
    class _DummyArray:
        """ Dummy object that just exists to hang __array_interface__ dictionaries
        and possibly keep alive a reference to a base array.
        """
        def __init__(self, interface, base=None):
            self.__array_interface__ = interface
            self.base = base

    def as_strided(x, shape=None, strides=None):
        """ Make an ndarray from the given array with the given shape and strides.
        """
        # work around Numpy bug 1873 (reported by Irwin Zaid)
        # Since this is stolen from numpy, this implementation has the same bug.
        # http://projects.scipy.org/numpy/ticket/1873
        # == https://github.com/numpy/numpy/issues/2466

        # Do not recreate the array if nothing need to be changed.
        # This fixes a lot of errors on pypy since DummyArray hack does not
        # currently (2014/May/17) on pypy.

        if ((shape is None or x.shape == shape)
                and (strides is None or x.strides == strides)):  # noqa: W503
            return x
        if not x.dtype.isbuiltin:
            if shape is None:
                shape = x.shape
            strides = tuple(strides)

            from pytools import product
            if strides is not None and shape is not None \
                    and product(shape) == product(x.shape) \
                    and x.flags.forc:
                # Workaround: If we're being asked to do what amounts to a
                # contiguous reshape, at least do that.

                if strides == f_contiguous_strides(x.dtype.itemsize, shape):
                    result = x.reshape(-1).reshape(*shape, order="F")
                    assert result.strides == strides
                    return result
                elif strides == c_contiguous_strides(x.dtype.itemsize, shape):
                    result = x.reshape(-1).reshape(*shape, order="C")
                    assert result.strides == strides
                    return result

            raise NotImplementedError(
                    "as_strided won't work on non-builtin arrays for now. "
                    "See https://github.com/numpy/numpy/issues/2466")

        interface = dict(x.__array_interface__)
        if shape is not None:
            interface["shape"] = tuple(shape)
        if strides is not None:
            interface["strides"] = tuple(strides)
        return np.asarray(_DummyArray(interface, base=x))

# }}}
