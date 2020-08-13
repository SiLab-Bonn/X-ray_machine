import logging
import coloredlogs
import numpy as np

from basil.dut import Dut
import xray

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

_local_config = {
    'directory': 'data/',
    'filename': 'debug',
    'distance': '8',
    'factor': 10,
    'steps_per_mm': 55555.555556,
    'address_x': 1,
    'address_y': 2,
    'address_z': 3,
    'invert_x': True,
    'invert_y': True,
    'invert_z': False,

    'xray_use_remote': False,
    'xray_voltage': 40,
    'xray_current': 50
}

# Example (debug mode): Create a new config, specify scanning range and step size
scan = xray.utils(debug=True, **_local_config)
filename = scan.init(x_range=10, y_range=10, stepsize=1, **_local_config)

# Move to home position (motor movement is simulated) and start the scan
scan.goto_home_position(('x', 'y'))
scan.step_scan()

# Create beam profile plots
plot = xray.plotting()
plot.plot_data(filename=filename, background=0, factor=1)
