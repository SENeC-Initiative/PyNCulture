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

try:
    import pint
    from pint import UnitRegistry, set_application_registry

    # check for the registry

    ureg = pint._APP_REGISTRY

    if ureg == pint._DEFAULT_REGISTRY:
        ureg = UnitRegistry()
        set_application_registry(ureg)

    Q_   = ureg.Quantity


    # length

    m   = ureg.meter
    cm  = ureg.cm
    mm  = ureg.mm
    um = ureg.micrometer


    # time

    day    = ureg.day
    hour   = ureg.hour
    minute = ureg.min
    second = ureg.second


    # frequency

    cps = ureg.count / ureg.second
    cpm = ureg.count / ureg.minute
    cph = ureg.count / ureg.hour


    # concentration

    M   = ureg.mol / ureg.L
    mM  = ureg.millimol / ureg.L
    uM = ureg.micromol / ureg.L


    # angles

    rad = ureg.rad
    deg = ureg.deg
    
    _unit_support = True
except:
    _unit_support = False
