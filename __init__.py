#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""
Module dedicated to the description of the spatial boundaries of neuronal
cultures.
This allows for the generation of neuronal networks that are embedded in space.

The `shapely<http://toblerity.org/shapely/index.html>`_ library is used to
generate and deal with the spatial environment of the neurons.


Content
=======
"""

import numpy as np

try:
    import shapely
    from shapely import speedups
    if speedups.available:
        speedups.enable()
    from .shape import Shape
except ImportError:
    from .backup_shape import Shape


__all__ = [
    "Shape",
    "culture_from_file",
]


# ------------------------------------------ #
# Try to import optional SVG and DXF support #
# ------------------------------------------ #

_svg_support = False
_dxf_support = False

try:
    from . import svgtools
    from .svgtools import *
    __all__.extend(svgtools.__all__)
    _svg_support = True
except Exception as e:
    print("Could not import svgtools: {}".format(e))

try:
    from . import dxftools
    from .dxftools import *
    __all__.extend(dxftools.__all__)
    _dxf_support = True
except Exception as e:
    print("Could not import dxftools: {}".format(e))


# ---------------------- #
# Make culture from file #
# ---------------------- #

def culture_from_file(filename, min_x=-5000., max_x=5000., unit='um',
                     parent=None, interpolate_curve=50):
    '''
    Generate a culture from an SVG or DXF file.

    Valid file needs to contain only closed objects among:
    rectangles, circles, ellipses, polygons, and closed curves.
    The objects do not have to be simply connected.
    '''
    shapes, points = None, None

    if filename.endswith(".svg") and _svg_support:
        shapes, points = svgtools.shapes_from_svg(
            filename, parent=parent, interpolate_curve=interpolate_curve,
            return_points=True)
    elif filename.endswith(".dxf") and _dxf_support:
        shapes, points = dxftools.shapes_from_dxf(
            filename, parent=parent, interpolate_curve=interpolate_curve,
            return_points=True)
    else:
        raise ImportError("You do not have support to load '" + filename + \
                          "', please install either 'svg.path' or "
                          "'dxfgrabber' to enable it.")
    idx_main_container = 0
    idx_local = 0
    type_main_container = ''
    count = 0
    min_x_val = np.inf

    # the main container must own the smallest x value
    for elt_type, elements in points.items():
        for i, elt_points in enumerate(elements):
            min_x_tmp = elt_points[:, 0].min()
            if min_x_tmp < min_x_val:
                min_x_val = min_x_tmp
                idx_main_container = count
                idx_local = i
                type_main_container = elt_type
            count += 1

    # make sure that the main container contains all other shapes
    main_container = shapes.pop(idx_main_container)
    exterior = points[type_main_container].pop(idx_local)
    for shape in shapes:
        assert main_container.contains(shape), "Some shapes are not " +\
            "contained in the main container."

    # all remaining shapes are considered as boundaries for the interior
    interiors = [item.coords for item in main_container.interiors]
    for elements in points.values():
        for elt_points in elements:
            interiors.append(elt_points)

    # scale the shape
    if None not in (min_x, max_x):
        exterior = np.array(main_container.exterior.coords)
        leftmost = np.min(exterior[:, 0])
        rightmost = np.max(exterior[:, 0])
        scaling = (max_x - min_x) / (rightmost - leftmost)
        exterior *= scaling
        interiors = [np.multiply(l, scaling) for l in interiors]

    culture = Shape(exterior, interiors, unit=unit, parent=parent)
    return culture
