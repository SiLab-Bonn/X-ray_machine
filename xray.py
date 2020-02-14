#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import os
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
from tqdm import tqdm

from basil.dut import Dut
import xray_plotting

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class utils():
    '''
        Simple beam profile scan
        - Stores the measured current values and the corresponding coordinates
        - Has a debug mode which generates fake data
    '''
    def __init__(self, debug=False, **kwargs):
        self.debug = debug
        self.devices = Dut('config.yaml')
        if self.debug is False:
            self.devices.init()
            if kwargs['xray_use_remote'] is True:
                self.xraymachine = Dut('xray.yaml')
                self.xraymachine.init()

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
        self.steps_per_mm = kwargs['steps_per_mm']
        self.debug_motor_res = 0.1
        self.debug_delay = 0
        self.debug_pos_mm = {'x': 0, 'y': 0, 'z': 0}
        self.filename = os.path.join(
            kwargs['directory'], kwargs['filename'] + '_' +
            kwargs['distance'] + 'cm_' + str(x_range) + '_' +
            str(y_range) + '_' + str(stepsize).replace('.', 'd') + '_' + 
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S" + '.csv'))

        logger.info('Filename: ' + self.filename)
        fh = logging.FileHandler(self.filename[:-4] + '.log')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(name)-10s] - %(levelname)-7s %(message)s'))

        if self.debug is False:
            fh.setLevel(logging.DEBUG)
            self.use_xray_control = kwargs['xray_use_remote']
            self.voltage = kwargs['xray_voltage']
            self.current = kwargs['xray_current']   
            if(kwargs['smu_use_bias'] is True and self.debug is False):
                self.init_smu(voltage=kwargs['smu_diode_bias'],
                              current_limit=kwargs['smu_current_limit'])
        logger.addHandler(fh)

        return self.filename

    def init_smu(self, voltage=0, current_limit=0):
        logger.info(self.devices['SMU'].get_name())
        self.devices['SMU'].source_volt()
        self.devices['SMU'].set_avg_on()
        self.devices['SMU'].set_avg_10()
        self.devices['SMU'].set_current_limit(current_limit)
        self.devices['SMU'].set_current_sense_range(1E-6)   # 1e-6 is the lowest possible range
        logger.debug(self.devices['SMU'].get_current_sense_range())
        if (abs(voltage) <= 55):
            self.devices['SMU'].set_voltage(voltage)
            self.devices['SMU'].on()
        else:
            logger.exception('SMU Voltage %s out of range' % voltage)

    def smu_get_current(self, n=None):
        ''' Returns the mean and sigma of n consecutive measurements
            For a single measurement, leave n empty
        '''
        if n is None:
            return float(self.devices['SMU'].get_current()[15:-43])
        else:
            rawdata = []
            for _ in range(n):
                rawdata.append(float(self.devices['SMU'].get_current()[15:-43]))
            logger.debug(rawdata)
            return np.mean(rawdata), np.std(rawdata)

    def xray_control(self, voltage=0, current=0, shutter='close'):
        ''' Sets high voltage and current, operate the shutter
            If no values are given for voltage and current,
            the local_configuration values are used
        '''
        if self.use_xray_control is False:
            logger.warning('X-ray control is not activated')
            return

        if voltage == 0 and current == 0:
            try:
                voltage = self.voltage
                current = self.current
            except RuntimeError as e:
                logger.error('X-ray voltage/current control failed. %s', e)

        if shutter == 'close':
            self.xraymachine["xray_tube"].close_shutter()
        if shutter == 'open':
            self.xraymachine["xray_tube"].open_shutter()

        self.xraymachine["xray_tube"].set_voltage(voltage)
        for _ in range(10):
            time.sleep(1)
            xray_v = self.xraymachine["xray_tube"].get_actual_voltage()
            logger.warning('X-ray voltage adjusing: %s kV' % xray_v)
            if xray_v == voltage:
                break
        if xray_v != voltage:
            logger.error('X-ray voltage is %s but expected %s' % (xray_v, voltage))
        else:
            logger.info('X-ray voltage set to %s kV' % xray_v)

        self.xraymachine["xray_tube"].set_current(current)
        for _ in range(10):
            time.sleep(1)
            xray_i = self.xraymachine["xray_tube"].get_actual_current()
            logger.warning('X-ray current ramping up: %s mA' % xray_i)
            if xray_i == current:
                break
        if xray_i != current:
            logger.error('X-ray current is %s but expected %s' % (xray_i, current))
        else:
            logger.info('X-ray current set to %s mA' % xray_i)

    def _ms_get_position(self, axis=None):
        for ax in axis:
            if self.debug is True:
                position_mm = self.debug_pos_mm[ax]
                position = position_mm * self.steps_per_mm
            else:
                position = int(self.devices['MS'].get_position(address=self.axis[ax]))
                # position = int(self._ms_write_read("TP", address=self.axis[ax])[2:-3])
                position_mm = position / self.steps_per_mm
            logger.debug('Motor stage controller %s position: %s \t %.3f mm' % (ax, position, position_mm))
            return position, position_mm

    def _ms_move_rel(self, axis=None, value=0, precision=0.02, wait=True):
        if self.invert[axis] is True:
            value = -value
        logger.debug('_ms_move_rel(axis=%s, value=%s, precision=%s)' % (axis, value, precision))
        if self.debug is False:
            self.devices['MS'].move_relative(value * self.steps_per_mm, address=self.axis[axis])
        if wait is True:
            self._wait_pos(axis=axis, target=self._ms_get_position(axis=axis)[1] + value)

    def _ms_move_abs(self, axis=None, value=0, precision=0.02, wait=True):
        if self.invert[axis] is True:
            value = -value
        logger.debug('_ms_move_abs(axis=%s, value=%s, precision=%s)' % (axis, value, precision))
        if self.debug is False:
            self.devices['MS'].set_position(value * self.steps_per_mm, address=self.axis[axis])
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

    def goto_home_position(self, axis=[], wait=True):
        logger.info('Moving to home position')
        if self.debug is False:
            for ax in axis:
                self.devices['MS']._write_command('GH', self.axis[ax])
        if wait is True:
            for ax in axis:
                self._wait_pos(ax)

    def set_home_position(self, axis=[]):
        for ax in axis:
            logger.warning('Set new home position for %s-axis)' % ax)
            self.devices['MS']._write_command('DH', self.axis[ax])

    def step_scan(self, precision=0.02, wait=True):
        x_range, y_range, z_height, stepsize = self.x_range, self.y_range, self.z_height, self.stepsize
        logger.debug('step_scan(x_range=%s, y_range=%s, z_height=%s, stepsize=%s, precision=%s, wait=%s):' % (x_range, y_range, z_height, stepsize, precision, wait))
        x_position = self._ms_get_position(axis='x')[1]
        y_position = self._ms_get_position(axis='y')[1]
        x_start = x_position - x_range / 2
        x_stop = x_position + x_range / 2
        y_start = y_position - y_range / 2
        y_stop = y_position + y_range / 2
        y_steps = int((y_stop - y_start) / stepsize)
        logger.info('Scanning range: x=%s mm, y=%s mm, z_height: %s, stepsize=%s mm' % (x_range, y_range, z_height, stepsize))
        logger.debug('x_start=%s, x_stop=%s, y_start=%s, y_stop=%s, y_steps=%s):' % (x_start, x_stop, y_start, y_stop, y_steps))

        self._ms_move_abs('x', value=x_start, wait=True)
        self._ms_move_abs('y', value=y_start, wait=True)
        if z_height is not False:
            self._ms_move_abs('z', value=z_height, wait=False)

        data = []
        done = False

        while done is False:
            outerpbar = tqdm(total=self.y_range / self.stepsize, desc='row', position=0)
            # move y
            for indx_y, y_move in enumerate(np.arange(y_start, y_stop + stepsize, stepsize)):
                logger.debug('row %s of %s' % (indx_y, y_steps))
                self._ms_move_abs('y', value=y_move, wait=True)

                if indx_y % 2:
                    x_start_line = x_stop
                    x_stop_line = x_start - stepsize
                    stepsize_line = -stepsize
                else:
                    x_start_line = x_start
                    x_stop_line = x_stop + stepsize
                    stepsize_line = stepsize

                data = []
                innerpbar = tqdm(total=self.x_range / self.stepsize, desc='column', position=1)

                # move x
                for indx_x, x_move in enumerate(np.arange(x_start_line, x_stop_line, stepsize_line)):
                    self._ms_move_abs('x', value=x_move, wait=True)
                    if self.debug is False:
                        time.sleep(0.1)
                        current = self.smu_get_current()
                    else:
                        x0, y0, fwhm = 0, 0, (x_stop - x_start) / 3
                        current = np.exp(-4 * np.log(2) * ((self.debug_pos_mm['x'] - x0)**2 + (self.debug_pos_mm['y'] - y0)**2) / fwhm**2) * 1e-6
                    data.append([round(x_move, 2), round(y_move, 2), current])
                    innerpbar.update()
                # write the last row and invert order for even rows
                if indx_y % 2:
                    data.reverse()
                with open(self.filename, mode='a') as csv_file:
                    file_writer = csv.writer(csv_file)
                    for entr in data:
                        file_writer.writerow(entr)

                outerpbar.update(1)

            done = True


if __name__ == '__main__':
    pass
