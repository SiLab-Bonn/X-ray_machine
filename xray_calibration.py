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
import logging
import coloredlogs
import xray

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

path = os.path.join('data', 'calibration')
calibration_filename = 'calibration.pkl'
os.chdir(path)

xplt = xray.plotting()

filelist = []
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f.endswith('.csv')]:
        filelist.append(os.path.join(dirpath, filename))

calibration = {}

for filename in filelist:
    try:
        distance = int(filename.split('/')[1].split('cm')[0])
        peak_intensity, beam_diameter = xplt.plot_data(filename=filename, background='auto', unit='rad', chip='none', distance=distance)
        calibration.update({distance: {'peak_intensity': peak_intensity, 'beam_diameter': beam_diameter}})
        logger.info('Processed "{}"'.format(filename))
    except RuntimeError as e:
        logger.error('Error loading {}/n{}'.format(filename, e))

# Save the analyzed calibration data
with open(calibration_filename, 'wb') as f:
    pickle.dump(calibration, f, pickle.HIGHEST_PROTOCOL)

with open(calibration_filename, 'rb') as f:
    data = pickle.load(f)
    sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: item[0])}
    xplt.plot_calibration_curves(sorted_data)
