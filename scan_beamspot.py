import logging
import coloredlogs
import xray
from xray import logger

local_logger = logging.getLogger('scan')
local_logger.setLevel('INFO')
coloredlogs.install(level=local_logger.level, logger=local_logger)

_local_config = {
    'directory': 'data/calibration/2023.11',
    'filename': 'profile',
    'distance': '14',
    'factor': 9.76,
    'xray_use_remote': True,
    'xray_voltage': 40,
    'xray_current': 50,
    'smu_init' : True,
    'smu_use_bias': True,
    'smu_diode_bias': -50,
    'smu_current_limit': 1.0E-04,
    'steps_per_mm': 55555.555556,
    'address_x': 1,
    'address_y': 2,
    'address_z': 3,
    'invert_x': True,
    'invert_y': True,
    'invert_z': False
}

# Simple script to measure a beam profile
scan = xray.utils(**_local_config)
filename = scan.init(x_range=32, y_range=32, stepsize=0.2, **_local_config)

scan.xray_control(shutter='close')
scan.goto_home_position(('x', 'y'))

background, std = scan.smu_get_current(10)
logger.info('Background current=%.3e A' % background)

scan.xray_control(shutter='open')
scan.step_scan(factor=_local_config['factor'], background=background)
scan.xray_control(shutter='close')

background_after, std_after = scan.smu_get_current(10)
logger.info('Background current after scan=%.3e A' % background_after)

scan.goto_home_position(('x', 'y'))

# filename = "data/calibration/2023.09/profile_14cm_32mm_32mm_0d2mm_40kV_50mA_2023-11-26_16-12-04 (copy 3).csv"
# background = 5.966323e-06

# generate plots
plot = xray.plotting()
try:
    plot.plot_data(filename=filename,
                    distance=float(_local_config['distance']),
                    factor=_local_config['factor'],
                    background=background,
                    unit='rad')
except RuntimeError as e:
    logger.error('Error loading' + filename, e)
