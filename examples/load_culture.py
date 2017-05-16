from descartes.patch import PolygonPatch

import matplotlib
import matplotlib.pyplot as plt

from PyNCulture import shapes_from_svg, shapes_from_dxf, culture_from_file


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

culture_file = "culture_with_holes.svg"
# culture_file = "culture.dxf"
# culture_file = "culture_from_filled_polygons.svg"

shapes = None

if culture_file.endswith(".dxf"):
    shapes = shapes_from_dxf(culture_file)
else:
    shapes = shapes_from_svg(culture_file)

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

culture = culture_from_file(culture_file)

shape2patch(ax2, culture)
ax2.set_aspect(1)


# ----------- #
# Add neurons #
# ----------- #

fig3, ax3 = plt.subplots()
plt.title("culture with neurons")

culture_bis = culture_from_file(culture_file)
pos = culture_bis.seed_neurons(neurons=1000)

shape2patch(ax3, culture_bis)
ax3.scatter(pos[:, 0], pos[:, 1], s=2, zorder=3)

ax3.set_aspect(1)
plt.show()
