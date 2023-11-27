import logging
import coloredlogs
from matplotlib.pyplot import sca
import xray
from xray import logger

local_logger = logging.getLogger('scan')
local_logger.setLevel('INFO')
coloredlogs.install(level='INFO', logger=local_logger)

_local_config = {
    'directory': 'data/tests',
    'filename': 'laser',
    'distance': '14',
    'factor': 9.76,
    'xray_use_remote': False,
    'xray_voltage': 40,
    'xray_current': 0,
    'smu_init' : True,
    'smu_use_bias': True,
    'smu_diode_bias': -5,
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
filename = scan.init(x_range=5, y_range=5, stepsize=1, **_local_config)

# input("Press Enter to home...")
scan.goto_home_position(('x', 'y'))

# input("Press Enter to start the scan...")
scan.step_scan(factor=_local_config['factor'],
                background='minimum')

# input("Press Enter to home again...")
scan.goto_home_position(('x', 'y'))

# generate plots
plot = xray.plotting()
try:
    plot.plot_data(filename=filename, unit='A')
except RuntimeError as e:
    logger.error('Error loading' + filename, e)
