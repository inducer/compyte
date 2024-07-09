"""Type mapping helpers."""


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


class TypeNameNotKnown(RuntimeError):  # noqa: N818
    pass


# {{{ registry

class DTypeRegistry:
    def __init__(self):
        self.dtype_to_name = {}
        self.name_to_dtype = {}

    def get_or_register_dtype(self, c_names, dtype=None):
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
            return single_valued(self.name_to_dtype[name] for name in c_names)

        dtype = np.dtype(dtype)

        # check if we've seen an identical dtype, if so retrieve exact dtype object.
        try:
            existing_name = self.dtype_to_name[dtype]
        except KeyError:
            existed = False
        else:
            existed = True
            existing_dtype = self.name_to_dtype[existing_name]
            assert existing_dtype == dtype
            dtype = existing_dtype

        for nm in c_names:
            try:
                name_dtype = self.name_to_dtype[nm]
            except KeyError:
                self.name_to_dtype[nm] = dtype
            else:
                if name_dtype != dtype:
                    raise RuntimeError("name '%s' already registered to "
                            "different dtype" % nm)

        if not existed:
            self.dtype_to_name[dtype] = c_names[0]
        if str(dtype) not in self.dtype_to_name:
            self.dtype_to_name[str(dtype)] = c_names[0]

        return dtype

    def dtype_to_ctype(self, dtype):
        if dtype is None:
            raise ValueError("dtype may not be None")

        dtype = np.dtype(dtype)

        try:
            return self.dtype_to_name[dtype]
        except KeyError:
            raise ValueError("unable to map dtype '%s'" % dtype) from None

# }}}


# {{{ C types

def fill_registry_with_c_types(reg, respect_windows, include_bool=True):
    import struct
    from sys import platform

    if include_bool:
        # bool is of unspecified size in the OpenCL spec and may in fact be
        # 4-byte.
        reg.get_or_register_dtype("bool", np.bool_)

    reg.get_or_register_dtype(["signed char", "char"], np.int8)
    reg.get_or_register_dtype("unsigned char", np.uint8)
    reg.get_or_register_dtype(["short", "signed short",
        "signed short int", "short signed int"], np.int16)
    reg.get_or_register_dtype(["unsigned short",
        "unsigned short int", "short unsigned int"], np.uint16)
    reg.get_or_register_dtype(["int", "signed int"], np.int32)
    reg.get_or_register_dtype(["unsigned", "unsigned int"], np.uint32)

    is_64_bit = struct.calcsize("@P") * 8 == 64
    if is_64_bit:
        if "win32" in platform and respect_windows:
            i64_name = "long long"
        else:
            i64_name = "long"

        reg.get_or_register_dtype(
                [i64_name, "%s int" % i64_name, "signed %s int" % i64_name,
                    "%s signed int" % i64_name],
                np.int64)
        reg.get_or_register_dtype(
                ["unsigned %s" % i64_name, "unsigned %s int" % i64_name,
                    "%s unsigned int" % i64_name],
                np.uint64)

    # http://projects.scipy.org/numpy/ticket/2017
    if is_64_bit:
        reg.get_or_register_dtype(["unsigned %s" % i64_name], np.uintp)
    else:
        reg.get_or_register_dtype(["unsigned"], np.uintp)

    reg.get_or_register_dtype("float", np.float32)
    reg.get_or_register_dtype("double", np.float64)


def fill_registry_with_opencl_c_types(reg):
    reg.get_or_register_dtype(["char", "signed char"], np.int8)
    reg.get_or_register_dtype(["uchar", "unsigned char"], np.uint8)
    reg.get_or_register_dtype(["short", "signed short",
        "signed short int", "short signed int"], np.int16)
    reg.get_or_register_dtype(["ushort", "unsigned short",
        "unsigned short int", "short unsigned int"], np.uint16)
    reg.get_or_register_dtype(["int", "signed int"], np.int32)
    reg.get_or_register_dtype(["uint", "unsigned", "unsigned int"], np.uint32)

    reg.get_or_register_dtype(
            ["long", "long int", "signed long int",
                "long signed int"],
            np.int64)
    reg.get_or_register_dtype(
            ["ulong", "unsigned long", "unsigned long int",
                "long unsigned int"],
            np.uint64)

    reg.get_or_register_dtype(["intptr_t"], np.intp)
    reg.get_or_register_dtype(["uintptr_t"], np.uintp)

    reg.get_or_register_dtype("float", np.float32)
    reg.get_or_register_dtype("double", np.float64)


def fill_registry_with_c99_stdint_types(reg):
    reg.get_or_register_dtype("bool", np.bool_)

    reg.get_or_register_dtype("int8_t", np.int8)
    reg.get_or_register_dtype("uint8_t", np.uint8)
    reg.get_or_register_dtype("int16_t", np.int16)
    reg.get_or_register_dtype("uint16_t", np.uint16)
    reg.get_or_register_dtype("int32_t", np.int32)
    reg.get_or_register_dtype("uint32_t", np.uint32)
    reg.get_or_register_dtype("int64_t", np.int64)
    reg.get_or_register_dtype("uint64_t", np.uint64)
    reg.get_or_register_dtype("uintptr_t", np.uintp)

    reg.get_or_register_dtype("float", np.float32)
    reg.get_or_register_dtype("double", np.float64)


def fill_registry_with_c99_complex_types(reg):
    reg.get_or_register_dtype("float complex", np.complex64)
    reg.get_or_register_dtype("double complex", np.complex128)
    reg.get_or_register_dtype("long double complex", np.clongdouble)

# }}}


# {{{ backward compatibility

TYPE_REGISTRY = DTypeRegistry()

# These are deprecated and should no longer be used
DTYPE_TO_NAME = TYPE_REGISTRY.dtype_to_name
NAME_TO_DTYPE = TYPE_REGISTRY.name_to_dtype

dtype_to_ctype = TYPE_REGISTRY.dtype_to_ctype
get_or_register_dtype = TYPE_REGISTRY.get_or_register_dtype


def _fill_dtype_registry(respect_windows, include_bool=True):
    fill_registry_with_c_types(
            TYPE_REGISTRY, respect_windows, include_bool)

# }}}


# {{{ c declarator parsing

def parse_c_arg_backend(c_arg, scalar_arg_factory, vec_arg_factory,
        name_to_dtype=None):
    if isinstance(name_to_dtype, DTypeRegistry):
        name_to_dtype = name_to_dtype.name_to_dtype__getitem__
    elif name_to_dtype is None:
        name_to_dtype = NAME_TO_DTYPE.__getitem__

    c_arg = (c_arg
            .replace("const", "")
            .replace("volatile", "")
            .replace("__restrict__", "")
            .replace("restrict", ""))

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
        raise ValueError("unknown type '%s'" % tp) from None

    return arg_class(dtype, name)

# }}}


def register_dtype(dtype, c_names, alias_ok=False):
    from warnings import warn
    warn("register_dtype is deprecated. Use get_or_register_dtype instead.",
            DeprecationWarning, stacklevel=2)

    if isinstance(c_names, str):
        c_names = [c_names]

    dtype = np.dtype(dtype)

    # check if we've seen this dtype before and error out if a) it was seen before
    # and b) alias_ok is False.

    if not alias_ok and dtype in TYPE_REGISTRY.dtype_to_name:
        raise RuntimeError("dtype '%s' already registered (as '%s', new names '%s')"
                % (dtype, TYPE_REGISTRY.dtype_to_name[dtype], ", ".join(c_names)))

    TYPE_REGISTRY.get_or_register_dtype(c_names, dtype)


# vim: foldmethod=marker
