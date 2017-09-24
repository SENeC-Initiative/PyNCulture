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

'''
Shape implementation using the
`shapely <http://toblerity.org/shapely/index.html>`_ library.
'''

import weakref

import shapely
from shapely.wkt import loads
from shapely.affinity import scale
from shapely.geometry import Point, Polygon

import numpy as np
from numpy.random import uniform

from .geom_utils import conversion_magnitude


__all__ = ["Shape"]


class Shape(Polygon):
    """
    Class containing the shape of the area where neurons will be distributed to
    form a network.

    Attributes
    ----------
    area : double
        Area of the shape in the :class:`Shape`'s
        :func:`Shape.unit` squared (:math:`\mu m^2`,
        :math:`mm^2`, :math:`cm^2`, :math:`dm^2` or :math:`m^2`).
    centroid : tuple of doubles
        Position of the center of mass of the current shape in `unit`.

    See also
    --------
    Parent class: :class:`shapely.geometry.Polygon`
    """

    @staticmethod
    def from_svg(filename, min_x=-5000., max_x=5000., unit='um', parent=None,
                 nterpolate_curve=50):
        '''
        Create a shape from an SVG file.

        Parameters
        ----------
        filename : str
            Path to the file that should be loaded.
        min_x : float, optional (default: -5000.)
            Absolute horizontal position of the leftmost point in the
            environment in `unit` (default: 'um'). If None, no rescaling
            occurs.
        max_x : float, optional (default: 5000.)
            Absolute horizontal position of the rightmost point in the
            environment in `unit`. If None, no rescaling occurs.
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'.
        parent : :class:`nngt.Graph` object
            The parent which will become a :class:`nngt.SpatialGraph`.
        interpolate_curve : int, optional (default: 50)
            Number of points that should be used to interpolate a curve.
        '''
        try:
            from .svgtools import culture_from_svg
            return culture_from_svg(
                filename,  min_x=min_x, max_x=max_x, unit=unit, parent=parent,
                interpolate_curve=interpolate_curve)
        except ImportError:
            raise ImportError("Install 'svg.path' to use this feature.")

    @staticmethod
    def from_dxf(filename, min_x=-5000., max_x=5000., unit='um', parent=None,
                 nterpolate_curve=50):
        '''
        Create a shape from an SVG file.

        Parameters
        ----------
        filename : str
            Path to the file that should be loaded.
        min_x : float, optional (default: -5000.)
            Absolute horizontal position of the leftmost point in the
            environment in `unit` (default: 'um'). If None, no rescaling
            occurs.
        max_x : float, optional (default: 5000.)
            Absolute horizontal position of the rightmost point in the
            environment in `unit`. If None, no rescaling occurs.
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'.
        parent : :class:`nngt.Graph` object
            The parent which will become a :class:`nngt.SpatialGraph`.
        interpolate_curve : int, optional (default: 50)
            Number of points that should be used to interpolate a curve.
        '''
        try:
            from .dxftools import culture_from_dxf
            return culture_from_dxf(
                filename,  min_x=min_x, max_x=max_x, unit=unit, parent=parent,
                interpolate_curve=interpolate_curve)
        except ImportError:
            raise ImportError("Install 'dxfgrabber' to use this feature.")


    @classmethod
    def from_polygon(cls, polygon, min_x=-5000., max_x=5000., unit='um',
                     parent=None):
        '''
        Create a shape from a :class:`shapely.geometry.Polygon`.

        Parameters
        ----------
        polygon : :class:`shapely.geometry.Polygon`
            The initial polygon.
        min_x : float, optional (default: -5000.)
            Absolute horizontal position of the leftmost point in the
            environment in `unit` If None, no rescaling occurs.
        max_x : float, optional (default: 5000.)
            Absolute horizontal position of the rightmost point in the
            environment in `unit` If None, no rescaling occurs.
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        parent : :class:`nngt.Graph` object
            The parent which will become a :class:`nngt.SpatialGraph`.
        '''
        assert isinstance(polygon, Polygon), "Expected a Polygon object."
        # find the scaling factor
        scaling = 1.
        if None not in (min_x, max_x):
            ext = np.array(polygon.exterior.coords)
            leftmost = np.min(ext[:, 0])
            rightmost = np.max(ext[:, 0])
            scaling = (max_x - min_x) / (rightmost - leftmost)
        # create the newly scaled polygon and convert it to Shape
        p2 = scale(polygon, scaling, scaling)
        p2.__class__ = cls
        p2._parent = parent
        p2._unit = unit
        p2._geom_type = 'Polygon'
        return p2

    @classmethod
    def from_wtk(cls, wtk, min_x=-5000., max_x=5000., unit='um', parent=None):
        '''
        Create a shape from a WTK string.

        .. versionadded:: 0.2

        Parameters
        ----------
        wtk : str
            The WTK string.

        See also
        --------
        :func:`Shape.from_polygon` for details about the other arguments.
        '''
        p = loads(wtk)
        return cls.from_polygon(
            p, min_x=min_x, max_x=max_x, unit=unit, parent=parent)

    @classmethod
    def rectangle(cls, height, width, centroid=(0., 0.), unit='um',
                  parent=None):
        '''
        Generate a rectangle of given height, width and center of mass.

        Parameters
        ----------
        height : float
            Height of the rectangle in `unit`
        width : float
            Width of the rectangle in `unit`
        centroid : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass in `unit`
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        parent : :class:`nngt.Graph` or subclass, optional (default: None)
            The parent container.

        Returns
        -------
        shape : :class:`Shape`
            Rectangle shape.
        '''
        half_w = 0.5 * width
        half_h = 0.5 * height
        centroid = np.array(centroid)
        points = [centroid + [half_w, half_h],
                  centroid + [half_w, -half_h],
                  centroid - [half_w, half_h],
                  centroid - [half_w, -half_h]]
        shape = cls(points, unit=unit, parent=parent)
        shape._geom_type = "Rectangle"
        return shape

    @classmethod
    def disk(cls, radius, centroid=(0.,0.), unit='um', parent=None):
        '''
        Generate a disk of given radius and center (`centroid`).

        Parameters
        ----------
        radius : float
            Radius of the disk in `unit`
        centroid : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass in `unit`
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        parent : :class:`nngt.Graph` or subclass, optional (default: None)
            The parent container.

        Returns
        -------
        shape : :class:`Shape`
            Rectangle shape.
        '''
        centroid = np.array(centroid)
        minx = centroid[0] - radius
        maxx = centroid[0] + radius
        disk = cls.from_polygon(
            Point(centroid).buffer(radius), min_x=minx, max_x=maxx, unit=unit,
            parent=parent)
        disk._geom_type = "Disk"
        disk.radius = radius
        return disk

    @classmethod
    def ellipse(cls, radii, centroid=(0.,0.), unit='um', parent=None):
        '''
        Generate a disk of given radius and center (`centroid`).

        Parameters
        ----------
        radii : tuple of floats
            Couple (rx, ry) containing the radii of the two axes in `unit`
        centroid : tuple of floats, optional (default: (0., 0.))
            Position of the rectangle's center of mass in `unit`
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'
        parent : :class:`nngt.Graph` or subclass, optional (default: None)
            The parent container.

        Returns
        -------
        shape : :class:`Shape`
            Rectangle shape.
        '''
        centroid = np.array(centroid)
        rx, ry = radii
        minx = centroid[0] - rx
        maxx = centroid[0] + rx
        ellipse = cls.from_polygon(
            scale(Point(centroid).buffer(1.), rx, ry), min_x=minx, max_x=maxx,
            unit=unit, parent=parent)
        ellipse._geom_type = "Ellipse"
        ellipse.radii = radii
        return ellipse

    def __init__(self, shell, holes=None, unit='um', parent=None):
        '''
        Initialize the :class:`Shape` object and the underlying
        :class:`shapely.geometry.Polygon`.

        Parameters
        ----------
        exterior : array-like object of shape (N, 2)
            List of points defining the external border of the shape.
        interiors : array-like, optional (default: None)
            List of array-like objects of shape (M, 2), defining empty regions
            inside the shape.
        unit : string (default: 'um')
            Unit in the metric system among 'um' (:math:`\mu m`), 'mm', 'cm',
            'dm', 'm'.
        parent : :class:`nngt.Graph` or subclass
            The graph which is associated to this Shape.
        '''
        self._parent = weakref.proxy(parent) if parent is not None else None
        self._unit = unit
        self._geom_type = 'Polygon'
        super(Shape, self).__init__(shell, holes=holes)

    @property
    def parent(self):
        ''' Return the parent of the :class:`Shape`. '''
        return self._parent

    @property
    def unit(self):
        '''
        Return the unit for the :class:`Shape` coordinates.
        '''
        return self._unit

    def set_parent(self, parent):
        ''' Set the parent :class:`nngt.Graph`. '''
        self._parent = weakref.proxy(parent) if parent is not None else None

    def add_subshape(self, subshape, position, unit='um'):
        """
        Add a :class:`Shape` to the current one.

        Parameters
        ----------
        subshape : :class:`Shape`
            Subshape to add.
        position : tuple of doubles
            Position of the subshape's center of gravity in space.
        unit : string (default: 'um')
            Unit in the metric system among 'um', 'mm', 'cm', 'dm', 'm'

        Returns
        -------
        None
        """
        raise NotImplementedError("To be implemented.")

    def seed_neurons(self, neurons=None, container=None, xmin=None, xmax=None,
                     ymin=None, ymax=None, soma_radius=0, unit=None):
        '''
        Return the positions of the neurons inside the
        :class:`Shape`.

        Parameters
        ----------
        neurons : int, optional (default: None)
            Number of neurons to seed. This argument is considered only if the
            :class:`Shape` has no `parent`, otherwise, a position is generated
            for each neuron in `parent`.
        container : :class:`Shape`, optional (default: None)
            Subshape acting like a mask, in which the neurons must be
            contained. The resulting area where the neurons are generated is
            the :func:`~shapely.Shape.intersection` between of the current
            shape and the `container`.
        xmin : double, optional (default: lowest abscissa of the Shape)
            Limit the area where neurons will be seeded to the region on the
            right of `xmin`.
        xmax : double, optional (default: highest abscissa of the Shape)
            Limit the area where neurons will be seeded to the region on the
            left of `xmax`.
        ymin : double, optional (default: lowest ordinate of the Shape)
            Limit the area where neurons will be seeded to the region on the
            upper side of `ymin`.
        ymax : double, optional (default: highest ordinate of the Shape)
            Limit the area where neurons will be seeded to the region on the
            lower side of `ymax`.
        unit : string (default: None)
            Unit in which the positions of the neurons will be returned, among
            'um', 'mm', 'cm', 'dm', 'm'.

        Returns
        -------
        positions : array of double with shape (N, 2)
        '''
        positions = None
        if self._parent is not None:
            neurons = self._parent.node_nb()
        if neurons is None:
            raise ValueError("`neurons` cannot be None if `parent` is None.")

        custom_shape = False
        if container is None:
            # set min/max
            if xmin is None:
                xmin = -np.inf
            if ymin is None:
                ymin = -np.inf
            if xmax is None:
                xmax = np.inf
            if ymax is None:
                ymax = np.inf
            min_x, min_y, max_x, max_y = self.bounds
            min_x = max(xmin, min_x)
            min_y = max(ymin, min_y)
            max_x = min(xmax, max_x)
            max_y = min(ymax, max_y)
            # remaining tests
            if self._geom_type == "Rectangle":
                xx = uniform(
                    min_x + soma_radius, max_x - soma_radius, size=neurons)
                yy = uniform(
                    min_y + soma_radius, max_y - soma_radius, size=neurons)
                positions = np.vstack((xx, yy)).T
            elif (self._geom_type == "Disk"
                  and (xmin, ymin, xmax, ymax) == self.bounds):
                theta = uniform(0, 2*np.pi, size=neurons)
                # take some precaution to stay inside the shape
                r = (self.radius - soma_radius) *\
                    np.sqrt(uniform(0, 0.99, size=neurons))
                positions = np.vstack(
                    (r*np.cos(theta) + self.centroid[0],
                     r*np.sin(theta) + self.centroid[1])).T
            else:
                custom_shape = True
                container = Polygon([(min_x, min_y), (min_x, max_y),
                                     (max_x, max_y), (max_x, min_y)])
        else:
            custom_shape = True
        # enter here only if Polygon or `container` is not None
        if custom_shape:
            seed_area = self.intersection(container)
            seed_area = Shape.from_polygon(
                seed_area.buffer(-soma_radius), min_x=min_x+soma_radius,
                max_x=max_x-soma_radius)
            if not isinstance(seed_area, Polygon):
                raise ValueError("Invalid boundary value for seed region; "
                                 "check that the min/max values you requested "
                                 "are inside the shape.")
            points = []
            p = Point()
            while len(points) < neurons:
                new_x = uniform(min_x, max_x, neurons-len(points))
                new_y = uniform(min_y, max_y, neurons-len(points))
                for x, y in zip(new_x, new_y):
                    p.coords = (x, y)
                    if seed_area.contains(p):
                        points.append((x, y))
            positions = np.array(points)

        if unit is not None and unit != self._unit:
            positions *= conversion_magnitude(unit, self._unit)

        return positions
