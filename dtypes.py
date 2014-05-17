"""Type mapping helpers."""

from __future__ import division

__copyright__ = "Copyright (C) 2011 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

# {{{ registry

try:
    hash(np.dtype([('s1', np.int8), ('s2', np.int8)]))
    DTypeDict = dict
    dtype_to_key = lambda t: t
    dtype_hashable = True
except:
    import json as _json
    dtype_hashable = False
    dtype_to_key = lambda t: _json.dumps(t.descr, separators=(',', ':'))
    class DTypeDict:
        def __init__(self):
            self.__dict = {}
            self.__type_dict = {}
        def __delitem__(self, key):
            try:
                del self.__dict[key]
            except TypeError:
                del self.__type_dict[dtype_to_key(key)]
        def __setitem__(self, key, val):
            try:
                self.__dict[key] = val
            except TypeError:
                self.__type_dict[dtype_to_key(key)] = val
        def __getitem__(self, key):
            try:
                return self.__dict[key]
            except TypeError:
                return self.__type_dict[dtype_to_key(key)]
        def __contains__(self, key):
            try:
                return key in self.__dict
            except TypeError:
                return dtype_to_key(key) in self.__type_dict

DTYPE_TO_NAME = DTypeDict()
NAME_TO_DTYPE = {}


class TypeNameNotKnown(RuntimeError):
    pass


def get_or_register_dtype(c_names, dtype=None):
    """Get or register a :class:`numpy.dtype` associated with the C type names
    in the string list *c_names*. If *dtype* is `None`, no registration is
    performed, and the :class:`numpy.dtype` must already have been registered.
    If so, it is returned.  If not, :exc:`TypeNameNotKnown` is raised.

    If *dtype* is not `None`, registration is attempted. If the *c_names* are
    already known and registered to identical :class:`numpy.dtype` objects,
    then the previously dtype object of the previously  registered type is
    returned. If the *c_names* are not yet known, the type is registered. If
    one of the *c_names* is known but registered to a different type, an error
    is raised. In this latter case, the type may end up partially registered
    and any further behavior is undefined.

    .. versionadded:: 2012.2
    """

    if isinstance(c_names, str):
        c_names = [c_names]

    if dtype is None:
        from pytools import single_valued
        return single_valued(NAME_TO_DTYPE[name] for name in c_names)

    dtype = np.dtype(dtype)

    # check if we've seen an identical dtype, if so retrieve exact dtype object.
    try:
        existing_name = DTYPE_TO_NAME[dtype]
    except KeyError:
        existed = False
    else:
        existed = True
        existing_dtype = NAME_TO_DTYPE[existing_name]
        assert existing_dtype == dtype
        dtype = existing_dtype

    for nm in c_names:
        try:
            name_dtype = NAME_TO_DTYPE[nm]
        except KeyError:
            NAME_TO_DTYPE[nm] = dtype
        else:
            if name_dtype != dtype:
                raise RuntimeError("name '%s' already registered to "
                        "different dtype" % nm)

    if not existed:
        DTYPE_TO_NAME[dtype] = c_names[0]
    if not str(dtype) in DTYPE_TO_NAME:
        DTYPE_TO_NAME[str(dtype)] = c_names[0]

    return dtype


def register_dtype(dtype, c_names, alias_ok=False):
    from warnings import warn
    warn("register_dtype is deprecated. Use get_or_register_dtype instead.",
            DeprecationWarning, stacklevel=2)

    if isinstance(c_names, str):
        c_names = [c_names]

    dtype = np.dtype(dtype)

    # check if we've seen this dtype before and error out if a) it was seen before
    # and b) alias_ok is False.

    if not alias_ok and dtype in DTYPE_TO_NAME:
        raise RuntimeError("dtype '%s' already registered (as '%s', new names '%s')"
                % (dtype, DTYPE_TO_NAME[dtype], ", ".join(c_names)))

    get_or_register_dtype(c_names, dtype)


def _fill_dtype_registry(respect_windows, include_bool=True):
    from sys import platform
    import struct

    if include_bool:
        # bool is of unspecified size in the OpenCL spec and may in fact be 4-byte.
        get_or_register_dtype("bool", np.bool)

    get_or_register_dtype(["signed char", "char"], np.int8)
    get_or_register_dtype("unsigned char", np.uint8)
    get_or_register_dtype(["short", "signed short",
        "signed short int", "short signed int"], np.int16)
    get_or_register_dtype(["unsigned short",
        "unsigned short int", "short unsigned int"], np.uint16)
    get_or_register_dtype(["int", "signed int"], np.int32)
    get_or_register_dtype(["unsigned", "unsigned int"], np.uint32)

    is_64_bit = struct.calcsize('@P') * 8 == 64
    if is_64_bit:
        if 'win32' in platform and respect_windows:
            i64_name = "long long"
        else:
            i64_name = "long"

        get_or_register_dtype(
                [i64_name, "%s int" % i64_name, "signed %s int" % i64_name,
                    "%s signed int" % i64_name],
                np.int64)
        get_or_register_dtype(
                ["unsigned %s" % i64_name, "unsigned %s int" % i64_name,
                    "%s unsigned int" % i64_name],
                np.uint64)

    # http://projects.scipy.org/numpy/ticket/2017
    if is_64_bit:
        get_or_register_dtype(["unsigned %s" % i64_name], np.uintp)
    else:
        get_or_register_dtype(["unsigned"], np.uintp)

    get_or_register_dtype("float", np.float32)
    get_or_register_dtype("double", np.float64)

# }}}


# {{{ dtype -> ctype

def dtype_to_ctype(dtype):
    if dtype is None:
        raise ValueError("dtype may not be None")

    dtype = np.dtype(dtype)

    try:
        return DTYPE_TO_NAME[dtype]
    except KeyError:
        raise ValueError("unable to map dtype '%s'" % dtype)

# }}}


# {{{ c declarator parsing

def parse_c_arg_backend(c_arg, scalar_arg_factory, vec_arg_factory,
        name_to_dtype=None):
    if name_to_dtype is None:
        name_to_dtype = NAME_TO_DTYPE.__getitem__
    c_arg = c_arg.replace("const", "").replace("volatile", "")

    # process and remove declarator
    import re
    decl_re = re.compile(r"(\**)\s*([_a-zA-Z0-9]+)(\s*\[[ 0-9]*\])*\s*$")
    decl_match = decl_re.search(c_arg)

    if decl_match is None:
        raise ValueError("couldn't parse C declarator '%s'" % c_arg)

    name = decl_match.group(2)

    if decl_match.group(1) or decl_match.group(3) is not None:
        arg_class = vec_arg_factory
    else:
        arg_class = scalar_arg_factory

    tp = c_arg[:decl_match.start()]
    tp = " ".join(tp.split())

    try:
        dtype = name_to_dtype(tp)
    except KeyError:
        raise ValueError("unknown type '%s'" % tp)

    return arg_class(dtype, name)

# }}}




# vim: foldmethod=marker
