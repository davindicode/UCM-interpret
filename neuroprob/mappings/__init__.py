#! /usr/bin/env python
# -*- coding: utf-8 -*-

from . import means

from .GP import CVI_VSSGP, SVGP, TT_SVGP, SKI_SVGP, ST_SVGP
from .nonparametric import histogram
from .parametric import GLM, FFNN
from .mixture import mixture_model, product_model, mixture_composition, HMM_model