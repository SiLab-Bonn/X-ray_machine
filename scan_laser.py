import logging
import coloredlogs
import xray
from xray import logger
import xray_plotting


local_logger = logging.getLogger('scan')
local_logger.setLevel('INFO')
coloredlogs.install(level='INFO', logger=local_logger)

_local_config = {
    'directory': 'data/calibration',
    'filename': 'laser',
    'distance': '20',
    'factor': 9.76,
    'xray_use_remote': False,
    'xray_voltage': 40,
    'xray_current': 3,
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

# Simple script to calibrate the alignment laser
scan = xray.utils(**_local_config)
filename = scan.init(x_range=10, y_range=10, stepsize=1, **_local_config)

scan.goto_home_position(('x', 'y'))
# scan._ms_move_abs('x', 0)
# scan._ms_move_rel('x', 2)
# scan._ms_move_rel('y', -2)
# scan.set_home_position(('x', 'y'))
# scan.goto_home_position(('x', 'y'))

scan.step_scan()

scan.goto_home_position(('x', 'y'))

# generate plots
plot = xray_plotting.plot()
try:
    plot.plot_data(filename=filename, unit='A')
except RuntimeError as e:
    logger.error('Error loading' + filename, e)
