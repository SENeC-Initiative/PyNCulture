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

from copy import deepcopy
from itertools import chain

from xml.dom.minidom import parse

from svg.path import (parse_path, CubicBezier, QuadraticBezier, Arc,
                      Move, Close, Path)

import shapely
from shapely.affinity import scale, affine_transform, translate
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

_valid_nodes = _predefined.keys()


def polygons_from_svg(filename, interpolate_curve=50, parent=None,
                      return_points=False):
    '''
    Generate :class:`shapely.geometry.Polygon` objects from an SVG file.
    '''
    svg = parse(filename)
    elt_structs = {k: [] for k in _valid_nodes}
    elt_points = {k: [] for k in _valid_nodes}

    # get the properties of all predefined elements
    for elt_type, elt_prop in _predefined.items():
        _build_struct(svg, elt_structs[elt_type], elt_type, elt_prop)

    # build all shapes
    polygons = []
    for elt_type, instructions in elt_structs.items():
        for struct in instructions:
            polygon, points = _make_polygon(
                elt_type, struct, parent=parent, return_points=True)
            polygons.append(polygon)
            elt_points[elt_type].append(points)

    if return_points:
        return polygons, elt_points

    return polygons


# ----- #
# Tools #
# ----- #


def _get_closed_subpaths(path):
    '''
    Generates all closed subpaths raises error if open subpaths exist.

    Credit to @tatarize:
    https://github.com/regebro/svg.path/issues/54#issuecomment-570101018
    '''
    segments = None
    for p in path:
        if isinstance(p, Move):
            if segments is not None:
                raise RuntimeError("Only closed shapes accepted.")
            segments = []

        segments.append(p)

        if isinstance(p, Close):
            yield Path(*segments)
            segments = None


def _get_points(path, interpolate_curve=50):
    ''' Get points from path. '''
    points = []

    for item in path:
        if isinstance(item, (Arc, CubicBezier, QuadraticBezier)):
            istart = 1. / interpolate_curve
            for frac in np.linspace(istart, 1, interpolate_curve):
                points.append(
                    (item.point(frac).real, item.point(frac).imag))
        else:
            points.append((item.start.real, item.start.imag))

    return points


def _get_outer_shell(paths_points):
    ''' Returns the index of the container subpath '''
    minx, maxx, miny, maxy = np.inf, -np.inf, np.inf, -np.inf

    container_idx = None

    for i, pp in enumerate(paths_points):
        arr = np.array(pp)

        x_min = np.min(arr[:, 0])
        x_max = np.max(arr[:, 0])
        y_min = np.min(arr[:, 1])
        y_max = np.max(arr[:, 1])

        winner = True

        if x_min <= minx:
            minx = x_min
        else:
            winner = False

        if x_max >= maxx:
            maxx = x_max
        else:
            winner = False

        if y_min <= miny:
            miny = y_min
        else:
            winner = False

        if y_max >= maxy:
            maxy = y_max
        else:
            winner = False

        if winner:
            container_idx = i

    return container_idx


def _build_struct(svg, container, elt_type, elt_properties):
    root    = svg.documentElement

    for elt in root.getElementsByTagName(elt_type):
        struct = {
            "transf": [],
            "transfdata": []
        }

        parent = elt.parentNode

        while parent is not None:
            _get_transform(parent, struct)
            parent = parent.parentNode

        _get_transform(elt, struct)
        
        if elt_type == 'path':
            path, trans = elt.getAttribute('d'), None
            struct["path"] = path
        else:
            for item in elt_properties:
                struct[item] = float(elt.getAttribute(item))

        container.append(struct)


def _make_polygon(elt_type, instructions, parent=None, interpolate_curve=50,
                  return_points=False):
    container = None
    shell     = []  # outer points defining the polygon's outer shell
    holes     = []  # inner points defining holes
    idx_start = 0

    if elt_type == "path":  # build polygons from custom paths
        path_data = parse_path(instructions["path"])
        subpaths  = [subpath for subpath in
                     _get_closed_subpaths(path_data)]
        points    = [_get_points(subpath) for subpath in subpaths]

        # get the container
        idx_container = _get_outer_shell(points)
        shell         = np.array(points[idx_container])

        # get the holes and make the shape
        holes     = [pp for i, pp in enumerate(points) if i!= idx_container]
        container = Polygon(shell, holes=holes)
    elif elt_type == "ellipse":  # build ellipses
        circle = Point((instructions["cx"], instructions["cy"])).buffer(1)
        rx, ry = instructions["rx"], instructions["ry"]
        container = scale(circle, rx, ry)
    elif elt_type == "circle":   # build circles
        r = instructions["r"]
        container = Point((instructions["cx"], instructions["cy"])).buffer(r)
    elif elt_type == "rect":     # build rectangles
        x, y = instructions["x"], instructions["y"]
        w, h = instructions["width"], instructions["height"]
        shell = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        container = Polygon(shell)
    else:
        raise RuntimeError("Unexpected element type: '{}'.".format(elt_type))

    # transforms
    nn, dd = instructions["transf"][::-1], instructions["transfdata"][::-1]
    for name, data in zip(nn, dd):
        if name == "matrix":
            container = affine_transform(container, data)
        elif name == "translate":
            container = translate(container, *data)

    # y axis is inverted in SVG, so make mirror transform
    container = affine_transform(container, (1, 0, 0, -1, 0, 0))
    shell     = np.array(container.exterior.coords)

    if return_points:
        return container, shell

    return container


def _get_transform(obj, tdict):
    ''' Get the transformation properties and name into `tdict` '''
    try:
        if obj.hasAttribute("transform"):
            trans = obj.getAttribute('transform')
            if trans.startswith("translate"):
                start = trans.find("(") + 1
                stop  = trans.find(")")
                tdict["transf"].append("translate")
                tdict["transfdata"].append(
                    [float(f) for f in trans[start:stop].split(",")])
            elif trans.startswith("matrix"):
                start = trans.find("(") + 1
                stop  = trans.find(")")
                trans = [float(f)
                         for f in trans[start:stop].split(",")]
                tdict["transf"].append("matrix")
                tdict["transfdata"].append(trans)
            else:
                raise RuntimeError("Uknown transform: " + trans)
    except:
        pass
