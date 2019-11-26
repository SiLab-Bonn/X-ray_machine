import logging
import coloredlogs
import numpy as np

from basil.dut import Dut
import xray
import xray_plotting

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

_local_config = {
    'directory': 'data/',
    'filename': 'test',
    'factor': 9.76,
    'xray_use_remote': True,
    'xray_voltage': 40,
    'xray_current': 5,
    'smu_use_bias': True,
    'smu_diode_bias': 50,
    'smu_current_limit': 1.000000E-04,
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
filename = scan.init(x_range=8, y_range=8, stepsize=0.4, **_local_config)

background, std = scan.smu_get_current(10)
logger.info('background current=%s' % background)

scan.goto_home_position(('x', 'y'))

# scan._ms_move_abs('x', 5)
# scan._ms_move_rel('x', 4)
# scan._ms_move_rel('y', 10)

scan.xray_control(shutter='open')
scan.step_scan()
scan.xray_control(shutter='close')

scan.goto_home_position(('x', 'y'))

# generate plots
plot = xray_plotting.plot()
try:
    plot.plot_data(filename=filename, background=background, factor=_local_config['factor'])
except RuntimeError as e:
    logger.error('Error loading' + filename, e)
