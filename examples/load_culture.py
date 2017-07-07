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

""" Loading a culture from an SVG or DXF file """

import matplotlib.pyplot as plt

import PyNCulture as pnc


''' Choose a file '''
culture_file = "culture_from_filled_polygons.svg"
# culture_file = "culture_with_holes.svg"
# culture_file = "culture.dxf"

shapes = None

if culture_file.endswith(".dxf"):
    shapes = pnc.shapes_from_dxf(culture_file)
else:
    shapes = pnc.shapes_from_svg(culture_file)

''' Plot the shapes '''
fig, ax = plt.subplots()
plt.title("shapes")

for shape in shapes:
    pnc.plot_shape(shape, ax)

''' Make a culture '''
fig2, ax2 = plt.subplots()
plt.title("culture")

culture = pnc.culture_from_file(culture_file)

pnc.plot_shape(culture, ax2)

''' Add neurons '''
fig3, ax3 = plt.subplots()
plt.title("culture with neurons")

culture_bis = pnc.culture_from_file(culture_file)
pos = culture_bis.seed_neurons(neurons=1000, xmax=0)

pnc.plot_shape(culture_bis, ax3)
ax3.scatter(pos[:, 0], pos[:, 1], s=2, zorder=3)

plt.show()
