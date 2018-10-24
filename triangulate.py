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

""" Triangulation and fast random point generation methods """

from OpenGL.GLU import *
from OpenGL.GL import *

from shapely.geometry import Polygon, MultiPolygon

import numpy as np


def triangulate(polygon):
    """
    Returns a triangulation of `polygon`.

    Parameters
    ----------
    polygon : a :class:`Shape` object or a :class:`~shapely.geometry.Polygon`
        or a :class:`~shapely.geometry.MultiPolygon` which will be decomposed
        into triangles.

    Returns
    -------
    triangles : generator containing triplets of triangle vertices.
    """
    vertices = []
    
    # opengl callbacks
    def _edgeFlagCallback(param1, param2): pass

    def _beginCallback(param=None):
        vertices = []

    def _vertexCallback(vertex, otherData=None):
        vertices.append(vertex[:2])

    def _combineCallback(vertex, neighbors, neighborWeights, out=None):
        out = vertex
        return out

    def _endCallback(data=None): pass

    # init tesselation
    tess = gluNewTess()
    gluTessProperty(tess, GLU_TESS_WINDING_RULE, GLU_TESS_WINDING_ODD)
    # force triangulation of polygons (i.e. GL_TRIANGLES) rather than
    # returning triangle fans or strips
    gluTessCallback(tess, GLU_TESS_EDGE_FLAG_DATA, _edgeFlagCallback)
    gluTessCallback(tess, GLU_TESS_BEGIN, _beginCallback)
    gluTessCallback(tess, GLU_TESS_VERTEX, _vertexCallback)
    gluTessCallback(tess, GLU_TESS_COMBINE, _combineCallback)
    gluTessCallback(tess, GLU_TESS_END, _endCallback)
    gluTessBeginPolygon(tess, 0)

    # first handle the main polygon(s)
    if isinstance(polygon, Polygon):
        _tesselate(tess, polygon)
    elif isinstance(polygon, MultiPolygon):
        for p in polygon.geoms:
            _tesselate(tess, p)

    # finish polygon and remove tesselator
    gluTessEndPolygon(tess)
    gluDeleteTess(tess)

    return ((vertices[i], vertices[i+1], vertices[i+2])
            for i in range(0, len(vertices), 3))


def rnd_pts_in_tr(triangles, num_points):
    '''
    Generate random points in a set of triangles.

    Parameters
    ----------
    triangles : list of :class:`shapely.geometry.Polygon` triangles
    num_points : number of points to generate.

    Returns
    -------
    points : np.array of shape (`num_points`, 2)
    '''
    # normalized areas
    areas = [t.area for t in triangles]
    areas = np.array(areas) / np.sum(areas)
    # vertices and triangle ids
    verts = np.array([t.exterior.coords for t in triangles])
    idx   = np.arange(0, len(triangles), dtype=int)

    # choose triangle based on its area
    chosen_idx = np.random.choice(idx, size=num_points, p=areas)
    chosen     = verts[chosen_idx]

    # generate random points inside these triangles
    r1, r2 = np.random.rand(2, num_points)

    As = chosen[:, 0, :]
    Bs = chosen[:, 1, :]
    Cs = chosen[:, 2, :]

    points = (As.T*(1 - np.sqrt(r1))).T \
             + (Bs.T*(np.sqrt(r1)*(1 - r2))).T \
             + (Cs.T*(np.sqrt(r1)*r2)).T

    return points


# polygon tesselation

def _tesselate(tess, polygon):
    gluTessBeginContour(tess)
    for point in polygon.exterior.coords:
        point3d = (point[0], point[1], 0)
        gluTessVertex(tess, point3d, point3d)
    gluTessEndContour(tess)
    # then handle each of the holes, if applicable
    for hole in polygon.interiors:
        gluTessBeginContour(tess)
        for point in hole.coords:
            point3d = (point[0], point[1], 0)
            gluTessVertex(tess, point3d, point3d)
        gluTessEndContour(tess)
