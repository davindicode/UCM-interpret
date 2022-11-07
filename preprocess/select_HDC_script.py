%load_ext autoreload
%autoreload 2

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

import pickle

import sys
sys.path.append("..")

import neuroprob as mdl
import neuroprob.utils as utils



dev = utils.pytorch.get_device()

mice_channels = {'Mouse12': {'ANT': [1, 2, 3, 4, 5, 6, 7, 8],
                        'mPFC': [9, 10, 11, 12], 
                        'CA1': [13]},
            'Mouse17': {'ANT': [1, 2, 3, 4, 5, 6, 7, 8],
                        'CA1': [9]},
            'Mouse20': {'ANT': [1, 3, 4, 5, 6, 7, 8]},
            'Mouse24': {'PoS': [1, 2, 3, 4],
                        'ANT': [5, 6, 7, 8]},
            'Mouse25': {'PoS': [1, 2, 3, 4],
                        'ANT': [5, 6, 7, 8]},
            'Mouse28': {'PoS': [1, 2, 3, 4, 5, 6, 7], 
                        'ANT': [8, 9, 10, 11]}
           }
mice_sessions = {'Mouse12': ['120806', '120807', '120809', '120810' ], # '120808' is missing position files
                'Mouse17': ['130125', '130128', '130129', '130130', '130131', '130201', '130202', '130203', '130204'],
                'Mouse20': ['130514', '130515', '130516', '130517', '130520'],
                'Mouse24': ['131213', '131216', '131217','131218'],
                'Mouse25': ['140123', '140124', '140128', '140129', '140130', '140131', '140203', '140204', '140205', '140206'],
                'Mouse28': ['140310', '140311', '140312', '140313', '140317', '140318']
               } 