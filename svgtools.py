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

from xml.dom.minidom import parse
from svg.path import parse_path, CubicBezier, QuadraticBezier, Arc
from itertools import chain

import shapely
from shapely.affinity import scale
from shapely.geometry import Point, Polygon

import numpy as np

from .shape import Shape


'''
Shape generation from SVG files.
'''


__all__ = ["polygons_from_svg"]


# predefined svg shapes and their parameters

_predefined = {
    'path': None,
    'ellipse': ("cx", "cy", "rx", "ry"),
    'circle': ("cx", "cy", "r"),
    'rect': ("x", "y", "width", "height")
}


def polygons_from_svg(filename, interpolate_curve=50, parent=None,
                      return_points=False):
    '''
    Generate :class:`shapely.geometry.Polygon` objects from an SVG file.
    '''
    svg = parse(filename)
    elt_structs = {k: [] for k in _predefined.keys()}
    elt_points = {k: [] for k in _predefined.keys()}

    # get the properties of all predefined elements
    for elt_type, elt_prop in _predefined.items():
        _build_struct(svg, elt_structs[elt_type], elt_type, elt_prop)

    # build all shapes
    shapes = []
    for elt_type, instructions in elt_structs.items():
        for struct in instructions:
            polygon, points = _make_shape(
                elt_type, struct, parent=parent, return_points=True)
            shapes.append(polygon)
            # ~ import matplotlib.pyplot as plt
            # ~ plt.scatter(points[:, 0], points[:, 1])
            # ~ plt.show()
            elt_points[elt_type].append(points)

    if return_points:
        return shapes, elt_points

    return shapes


# ----- #
# Tools #
# ----- #

def _build_struct(svg, container, elt_type, elt_properties):
    for elt in svg.getElementsByTagName(elt_type):
        if elt_type == 'path':
            #~ for s in elt.getAttribute('d').split('z'):
                #~ if s:
                    #~ container.append(s.lstrip() + 'z')
            container.append(elt.getAttribute('d'))
        else:
            struct = {}
            for item in elt_properties:
                struct[item] = float(elt.getAttribute(item))
            container.append(struct)


def _make_shape(elt_type, instructions, parent=None, interpolate_curve=50,
                return_points=False):
    container = None
    shell = []  # outer points defining the polygon's outer shell
    holes = []  # inner points defining holes

    if elt_type == "path":  # build polygons from custom paths
        path_data = parse_path(instructions)
        num_data = len(path_data)
        if not path_data.closed:
            raise RuntimeError("Only closed shapes accepted.")
        start = path_data[0].start
        points = shell  # the first path is the outer shell?
        for j, item in enumerate(path_data):
            if isinstance(item, (Arc, CubicBezier, QuadraticBezier)):
                for frac in np.linspace(0, 1, interpolate_curve):
                    points.append(
                        (item.point(frac).real, -item.point(frac).imag))
            else:
                points.append((item.start.real, -item.start.imag))
            # if the shell is closed, the rest defines holes
            if item.end == start and j < len(path_data) - 1:
                holes.append([])
                points = holes[-1]
                start = path_data[j+1].start
        container = Shape(shell, holes=holes)
        shell = np.array(shell)
    elif elt_type == "ellipse":  # build ellipses
        circle = Point((instructions["cx"], -instructions["cy"])).buffer(1)
        rx, ry = instructions["rx"], instructions["ry"]
        container = Shape.from_polygon(scale(circle, rx, ry), min_x=None)
    elif elt_type == "circle":  # build circles
        container = Shape.from_polygon(Point((instructions["cx"],
            -instructions["cy"])).buffer(instructions["r"]), min_x=None)
    elif elt_type == "rect":  # build rectangles
        x, y = instructions["x"], -instructions["y"]
        w, h = instructions["width"], -instructions["height"]
        shell = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        container = Shape(shell)
    else:
        raise RuntimeError("Unexpected element type: '{}'.".format(elt_type))

    if return_points:
        if len(shell) == 0:
            shell = np.array(container.exterior.coords)
        return container, shell

    return container
