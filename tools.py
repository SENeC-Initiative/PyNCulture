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

from PyNCulture import _shapely_support


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
