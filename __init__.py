#!/usr/bin/env python
#-*- coding:utf-8 -*-
#
# This file is part of the PyNCulture project, which aims at providing tools to
# easily generate complex neuronal cultures.
# Copyright (C) 2017 SENeC Initiative
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Principle
=========

Module dedicated to the description of the spatial boundaries of neuronal
cultures.
This allows for the generation of neuronal networks that are embedded in space.

The `shapely <http://toblerity.org/shapely/index.html>`_ library is used to
generate and deal with the spatial environment of the neurons.


Examples
========

Basic features
--------------

The module provides a backup ``Shape`` object, which can be used with only
the `numpy` and `scipy` libraries.
It allows for the generation of simple rectangle, disk and ellipse shapes.

.. literalinclude:: examples/backup_shape.py
   :lines: 23-

All these features are of course still available with the more advanced
``Shape`` object which inherits from :class:`shapely.geometry.Polygon`.


Complex shapes from files
-------------------------

.. literalinclude:: examples/load_culture.py
   :lines: 23-


Content
=======
"""

import logging

import numpy as np

try:
    import shapely
    from shapely import speedups
    if speedups.available:
        speedups.enable()
    from .shape import Shape
    _shapely_support = True
except ImportError:
    from .backup_shape import BackupShape as Shape
    _shapely_support = False


__all__ = ["Shape"]


version = 0.2


if _shapely_support:
    __all__.append("culture_from_file")


# ------------------------------------------ #
# Try to import optional SVG and DXF support #
# ------------------------------------------ #

_logger = logging.getLogger(__name__)

_svg_support = False
_dxf_support = False

try:
    from . import svgtools
    from .svgtools import *
    __all__.extend(svgtools.__all__)
    _svg_support = True
except ImportError as e:
    _logger.info("Could not import svgtools: {}".format(e))

try:
    from . import dxftools
    from .dxftools import *
    __all__.extend(dxftools.__all__)
    _dxf_support = True
except ImportError as e:
    _logger.info("Could not import dxftools: {}".format(e))

try:
    from .plot import plot_shape
    __all__.append('plot_shape')
except ImportError as e:
    _logger.info('Could not import plotting tools: {}'.format(e))


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

    Parameters
    ----------
    filename : str
        Path to the SVG or DXF file.
    min_x : float, optional (default: -5000.)
        Position of the leftmost coordinate of the shape's exterior, in `unit`.
    max_x : float, optional (default: 5000.)
        Position of the rightmost coordinate of the shape's exterior, in
        `unit`.
    , unit : str, optional (default: 'um')
        Unit of the positions, among micrometers ('um'), milimeters ('mm'),
        centimeters ('cm'), decimeters ('dm'), or meters ('m').
     parent : :class:`nngt.Graph` or subclass, optional (default: None)
        Assign a parent graph if working with NNGT.
    interpolate_curve : int, optional (default: 50)
        Number of points by which a curve should be interpolated into segments.

    Returns
    -------
    culture : :class:`Shape` object
        Shape, vertically centred around zero, such that
        :math:`min(y) + max(y) = 0`.
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
        y_center = 0.5*(np.min(exterior[:, 1]) + np.max(exterior[:, 1]))
        scaling = (max_x - min_x) / (rightmost - leftmost)
        y_center *= scaling
        exterior *= scaling
        x_trans = min_x - np.min(exterior[:, 0])
        exterior[:, 0] += x_trans
        exterior[:, 1] -= y_center
        interiors = [np.multiply(l, scaling) for l in interiors]
        for path in interiors:
            path[:, 0] += x_trans
            path[:, 1] -= y_center

    culture = Shape(exterior, interiors, unit=unit, parent=parent)
    return culture
