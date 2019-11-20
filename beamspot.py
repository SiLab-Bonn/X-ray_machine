#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#
from basil.dut import Dut
import time
import datetime
import math
import numpy as np
import tables as tb
import csv
import logging
import coloredlogs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import progressbar

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

_local_config = {
    'directory': 'data/',
    'filename': 'fake',
    'smu_use_bias': True,
    'smu_diode_bias': 5.0,
    'smu_current_limit': 1.000000E-04,
    'steps_per_mm': 55555.555556,
    'backlash': 0.0,
    'address_x': 1,
    'address_y': 2,
    'address_z': 3,
    'invert_x': True,
    'invert_y': True,
    'invert_z': False
}


class scan_beamspot():
    '''
        Simple beam profile scan
        - Stores the measured current values and the corresponding coordinates
        - Has a debug mode which generates fake data
    '''

    def __init__(self):
        self.debug = False
        self.debug_motor_res = 0.1
        self.debug_delay = 0
        self.devices = Dut('config.yaml')
        if self.debug is False:
            self.devices.init()

    def init(self, x_range=0, y_range=0, z_height=False, stepsize=0, **kwargs):
        self.axis = {
            'x': kwargs['address_x'],
            'y': kwargs['address_y'],
            'z': kwargs['address_z']
            }
        self.invert = {
            'x': kwargs['invert_x'],
            'y': kwargs['invert_y'],
            'z': kwargs['invert_z']
            }
        self.x_range, self.y_range, self.z_height, self.stepsize = x_range, y_range, z_height, stepsize
        self.backlash = kwargs['backlash']
        self.steps_per_mm = kwargs['steps_per_mm']
        self.debug_pos_mm = {'x': 0, 'y': 0, 'z': 0}
        self.filename = kwargs['directory']+kwargs['filename']+'_'+str(x_range)+'_'+str(y_range)+'_'+str(stepsize).replace('.','d')+'_'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"+'.csv')
        logger.info('Filename: '+self.filename)
        if(kwargs['smu_use_bias'] is True and self.debug is False):
            self.init_smu(voltage=kwargs['smu_diode_bias'], current_limit=kwargs['smu_current_limit'])

    def init_smu(self, voltage=0, current_limit=0):
        logger.info(self.devices['SMU'].get_name())
        self.devices['SMU'].source_volt()
        self.devices['SMU'].set_current_limit(current_limit)
        if (abs(voltage) <= 55):
            self.devices['SMU'].set_voltage(voltage)
            self.devices['SMU'].on()
        else:
            logger.exception('SMU Voltage %s out of range' % voltage)

    # def _ms_write_read(self, command=None, address=None):
    #     self.devices['MS'].write(bytearray.fromhex("01%d" % (address + 30)) + command.encode())
    #     ret = self.devices['MS'].read()
    #     return ret

    # def _ms_get_status(self, axis=None):
    #     status = self.devices['MS'].get_channel(address=self.axis[axis])
    #     logger.debug('Motor stage controller %s status: %s' % (axis, status))
    #     return status

    def _ms_get_position(self, axis=None):
        if self.debug is True:
            position_mm = self.debug_pos_mm[axis]
            position = position_mm*self.steps_per_mm
        else:
            position = int(self.devices['MS'].get_position(address=self.axis[axis]))
            #position = int(self._ms_write_read("TP", address=self.axis[axis])[2:-3])
            position_mm = position/self.steps_per_mm
        logger.debug('Motor stage controller %s position: %s \t %.3f mm' % (axis, position, position_mm))
        return position, position_mm

    def _ms_move_rel(self, axis=None, value=0, precision=0.02, wait=True):
        if self.invert[axis] is True:
            value = -value
        logger.debug('_ms_move_rel(axis=%s, value=%s, precision=%s)' % (axis, value, precision))
        if self.debug is False:
            self.devices['MS'].move_relative(value*self.steps_per_mm, address=self.axis[axis])
        if wait is True:
            self._wait_pos(axis=axis, target=self._ms_get_position(axis=axis)[1]+value)

    def _ms_move_abs(self, axis=None, value=0, precision=0.02, wait=True):
        if self.invert[axis] is True:
            value = -value
        logger.debug('_ms_move_abs(axis=%s, value=%s, precision=%s)' % (axis, value, precision))
        if self.debug is False:
            self.devices['MS'].set_position(value*self.steps_per_mm, address=self.axis[axis])
        if wait is True:
            self._wait_pos(axis=axis, target=value)

    def _ms_stop(self, axis=None):
        self.devices['MS']._write_command('AB', self.axis[axis])

    def _wait_pos(self, axis=None, precision=0.02, target=0):
        logger.debug('_wait_pos(axis=%s, precision=%s, target=%s)' % (axis, precision, target))
        logger.debug('Moving motor %s: %.2f mm -> %.2f mm' % (axis, self._ms_get_position(axis=axis)[1], target))
        done = False
        pos_mm = 0
        while done is False:
            if self.debug is True:
                e_pos = self.debug_pos_mm[axis] - target
                if abs(e_pos) > precision:
                    self.debug_pos_mm[axis] = self.debug_pos_mm[axis] - math.copysign(self.debug_motor_res, e_pos)
                    pos_mm = self.debug_pos_mm[axis]
                    logger.debug('Moving motor %s: %.3f -> %.3f' % (axis, pos_mm, target))
                    time.sleep(self.debug_delay)
                else:
                    done = True
                pos = pos_mm * self.steps_per_mm
            else:
                pos, pos_mm = self._ms_get_position(axis=axis)
                logger.debug('Moving motor %s: %.2f mm -> %.2f mm' % (axis, pos_mm, target))
                if abs(round(pos_mm, 2) - round(target, 2)) <= precision:
                    done = True
                else:
                    time.sleep(0.2)
        return pos, pos_mm

    def goto_end_position(self, axis=None, negative=True, wait=True):
        logger.debug('goto_end_position(axis=%s, negative=%s, wait=%s)' % (axis, negative, wait))
        if negative is True:
            self.devices['MS']._write_command('FE1', self.axis[axis])
        else:
            self.devices['MS']._write_command('FE0', self.axis[axis])
        if wait is True:
            self._wait_pos(axis=axis, target=0)

    def goto_home_position(self, axis=None, wait=True):
        logger.info('Moving to home position')
        if self.debug is False:
            self.devices['MS']._write_command('GH', self.axis[axis])
        if wait is True:
            self._wait_pos(axis)

    def set_home_position(self, axis=None):
        logger.warning('Set new home position for %s-axis)' % axis)
        self._ms_write_read('DH', self.axis[axis])

    def step_scan(self, precision=0.02, wait=True):
        x_range, y_range, z_height, stepsize = self.x_range, self.y_range, self.z_height, self.stepsize
        logger.debug('step_scan(x_range=%s, y_range=%s, z_height=%s, stepsize=%s, precision=%s, wait=%s):' % (x_range, y_range, z_height, stepsize, precision, wait))
        x_position = self._ms_get_position(axis='x')[1]
        y_position = self._ms_get_position(axis='y')[1]
        x_start = x_position - x_range/2
        x_stop = x_position + x_range/2
        y_start = y_position - y_range/2
        y_stop = y_position + y_range/2
        y_steps = int((y_stop - y_start)/stepsize)
        logger.info('Scanning range: x=%s mm, y=%s mm, z_height: %s, stepsize=%s mm' % (x_range, y_range, z_height, stepsize))
        logger.debug('x_start=%s, x_stop=%s, y_start=%s, y_stop=%s, y_steps=%s):' % (x_start, x_stop, y_start, y_stop, y_steps))

        self._ms_move_abs('x', value=x_start, wait=True)
        self._ms_move_abs('y', value=y_start, wait=True)
        if z_height is not False:
            self._ms_move_abs('z', value=z_height, wait=False)

        data = []
        done = False
        backlash = self.backlash

        while done is False:
            # move y
            for indx_y, y_move in enumerate(np.arange(y_start, y_stop+stepsize, stepsize)):
                logger.info('row %s of %s' % (indx_y, y_steps))
                self._ms_move_abs('y', value=y_move, wait=True)
                # move x
                if indx_y % 2:
                    x_start_line = x_stop
                    x_stop_line = x_start-stepsize
                    stepsize_line = -stepsize
                else:
                    x_start_line = x_start
                    x_stop_line = x_stop+stepsize
                    stepsize_line = stepsize

                widgets = [progressbar.Percentage(), progressbar.Bar()]
                bar = progressbar.ProgressBar(self.x_range/self.stepsize, widgets=widgets).start()
                for indx_x, x_move in enumerate(np.arange(x_start_line, x_stop_line, stepsize_line)):
                    self._ms_move_abs('x', value=x_move+backlash, wait=True)
                    bar.update(indx_y)
                    if self.debug is False:                    
                        time.sleep(0.1)
                        rawdata = self.devices['SMU'].get_current()
                        current = float(rawdata[15:-43])
                    else:
                        current = (1/(2*np.pi*5*10) * np.exp(-(self.debug_pos_mm['x']**2/(2*5**2) + self.debug_pos_mm['y']**2/(2*5**2))))
                    data.append([x_move, y_move, current])
                    with open(self.filename, mode='a') as csv_file:
                        file_writer = csv.writer(csv_file)
                        file_writer.writerow([x_move, y_move, current])
                bar.finish()
            done = True

        data = np.array(data).T
        N = int(len(data[2])**.5)
        z = data[2].reshape(N, N)

        z_reshaped = []
        for idx, row in enumerate(z):
            if idx % 2:
                z_reshaped.append(np.flip(row))
            else:
                z_reshaped.append(row)

        plt.imshow(np.flip(z_reshaped, 0), extent=(np.amin(data[0]), np.amax(data[0]), np.amin(data[1]), np.amax(data[1])), aspect = '1')
        plt.colorbar()
        plt.savefig('last.png')
        plt.savefig(fname=self.filename[:-4], dpi=200)
        plt.show()


if __name__ == '__main__':
    scan = scan_beamspot()
    scan.init(x_range=2, y_range=2, z_height=False, stepsize=1, **_local_config)

    scan._ms_get_position('x')
    scan._ms_get_position('y')
#    scan._ms_get_position('z')

    scan.goto_home_position('x')
    scan.goto_home_position('y')

#    scan._ms_move_rel('x', -4)
#    scan._ms_move_rel('y', 10)

#    scan.set_home_position('x')
#    scan.set_home_position('y')

    scan.step_scan()

    scan.goto_home_position('x')
    scan.goto_home_position('y')
