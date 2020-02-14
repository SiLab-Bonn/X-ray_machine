import logging
import coloredlogs
import numpy as np

from basil.dut import Dut
import xray
from xray import logger
import xray_plotting


local_logger = logging.getLogger('scan')
local_logger.setLevel('INFO')
coloredlogs.install(level='INFO', logger=local_logger)

_local_config = {
    'directory': 'data/',
    'filename': 'profile',
    'distance': '7',
    'factor': 9.76,
    'xray_use_remote': True,
    'xray_voltage': 40,
    'xray_current': 50,
    'smu_use_bias': True,
    'smu_diode_bias': 50,
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
filename = scan.init(x_range=10, y_range=10, stepsize=1, **_local_config)

scan.xray_control(shutter='close')

scan.goto_home_position(('x', 'y'))
# scan._ms_move_abs('x', -0.5)
# scan._ms_move_rel('x', 1.5)
# scan._ms_move_rel('y', 1)
# scan.set_home_position(('x','y'))
# scan.goto_home_position(('x', 'y'))

background, std = scan.smu_get_current(10)
logger.info('Background current=%.3e' % background)

scan.xray_control(shutter='open')
scan.step_scan()
scan.xray_control(shutter='close')

background_after, std_after = scan.smu_get_current(10)
logger.info('Background current after scan=%.3e' % background_after)

scan.goto_home_position(('x', 'y'))

# generate plots
plot = xray_plotting.plot()
try:
    plot.plot_data(filename=filename, background=background, factor=_local_config['factor'], unit='rad')
except RuntimeError as e:
    logger.error('Error loading' + filename, e)
