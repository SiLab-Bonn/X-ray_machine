#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

''' This script creates calibration plots from analyzed beam profiles,
    using saved data in pickle format and plot functions from the xray_plotting class
'''

import os
import pickle
import xray

path = os.path.join('data', 'calibration')
filename = 'calibration.pkl'
os.chdir(path)

xplt = xray.plotting()

with open(filename, 'rb') as f:
    data = pickle.load(f)
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
    xplt.plot_calibration_curves(sorted_data)
