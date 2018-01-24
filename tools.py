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

""" Tools for PyNCulture """

import numpy as np

from . import _shapely_support


def pop_largest(shapes):
    '''
    Returns the largest shape, removing it from the list.
    If `shapes` is a :class:`shapely.geometry.MultiPolygon`, returns the
    largest :class:`shapely.geometry.Polygon` without modifying the object.

    .. versionadded:: 0.3

    Parameters
    ----------
    shapes : list of :class:`Shape` objects or MultiPolygon.
    '''
    MultiPolygon = None
    try:
        from shapely.geometry import MultiPolygon
    except ImportError:
        pass

    max_area = -np.inf
    max_idx  = -1

    for i, s in enumerate(shapes):
        if s.area > max_area:
            max_area = s.area
            max_idx  = i

    if shapes.__class__ == MultiPolygon:
        return shapes[max_idx]

    return shapes.pop(max_idx)


def _insert_area(container, area_name, shape, height, properties):
    '''
    Insert the area into the container, potentially restructuring the existing
    areas.
    In particular, if the shape is composed of multiple polygons, it will
    be inserted as "area_name_X", with X ranging from 0 to N-1, N being the
    number of polygons.

    If `area_name` already exists in `container`, it will be overriden,
    potentially even deleted in favor of numbered subareas if it is
    replaced by a set of polygons.
    The only exception to that rule is the "default_area", which is never
    deleted but replaced by the largest polygon, while the smaller ones are
    numbered from 1.
    '''
    # import
    from .shape import Area
    from shapely.geometry import MultiPolygon
    # check for multiple polygons
    if shape.__class__ == MultiPolygon:
        # behavior differs for default_area (never deleted) and other areas
        if area_name == "default_area":
            largest = pop_largest(shape)
            count   = 1
            for p in shape:
                new_name = area_name
                if p != largest:
                    new_name = area_name + '_' + str(count)
                    count += 1
                container._areas[new_name] = Area.from_shape(
                    p, height=height, name=new_name, properties=properties)
        else:
            for i, p in enumerate(shape):
                new_name = area_name + '_' + str(i)
                container._areas[new_name] = Area.from_shape(
                    p, height=height, name=new_name, properties=properties)
            if area_name in container.areas:
                del container._areas[area_name]
    else:
        container._areas[area_name] = Area.from_shape(
                shape, height=height, name=area_name, properties=properties)
            
