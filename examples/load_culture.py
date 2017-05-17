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

from descartes.patch import PolygonPatch

import matplotlib
import matplotlib.pyplot as plt

import PyNCulture as pync


# ------------- #
# Plot function #
# ------------- #

def shape2patch(ax, shape):
    x, y = shape.exterior.xy
    ax.plot(x, y, 'o', color='#999999', zorder=1)
    patch = PolygonPatch(shape, alpha=0.5, zorder=2)
    ax.add_patch(patch)


# ------- #
# Example #
# ------- #

# chose a file

# culture_file = "culture_with_holes.svg"
# culture_file = "culture.dxf"
culture_file = "culture_from_filled_polygons.svg"

shapes = None

if culture_file.endswith(".dxf"):
    shapes = pync.shapes_from_dxf(culture_file)
else:
    shapes = pync.shapes_from_svg(culture_file)

# --------------- #
# Plot the shapes #
# --------------- #

fig, ax = plt.subplots()
plt.title("shapes")

for shape in shapes:
    shape2patch(ax, shape)

ax.set_aspect(1)


# -------------- #
# Make a culture #
# -------------- #

fig2, ax2 = plt.subplots()
plt.title("culture")

culture = pync.culture_from_file(culture_file)

shape2patch(ax2, culture)
ax2.set_aspect(1)


# ----------- #
# Add neurons #
# ----------- #

fig3, ax3 = plt.subplots()
plt.title("culture with neurons")

culture_bis = pync.culture_from_file(culture_file)
pos = culture_bis.seed_neurons(neurons=1000)

shape2patch(ax3, culture_bis)
ax3.scatter(pos[:, 0], pos[:, 1], s=2, zorder=3)

ax3.set_aspect(1)
plt.show()
