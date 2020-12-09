#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#

import csv
import os
import math
import logging
import coloredlogs
import time
import datetime
import numpy as np
from basil.dut import Dut
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

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
            kwargs['distance'] + 'cm_' + str(x_range) + 'mm_' + str(y_range) + 'mm_' +
            str(stepsize).replace('.', 'd') + 'mm_' +
            str(kwargs['xray_voltage']) + 'kV_' + str(kwargs['xray_current']) + 'mA_' +
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
        self.devices['SMU'].set_avg_en(1)
        self.devices['SMU'].set_avg_n(10)
        self.devices['SMU'].set_current_limit(current_limit)
        self.devices['SMU'].set_current_sense_range(1E-6)   # 1e-6 is the lowest possible range
        self.devices['SMU'].set_beeper(0)
        logger.debug(self.devices['SMU'].get_current_sense_range())
        if (abs(voltage) <= 55):
            self.devices['SMU'].set_voltage(voltage)
        else:
            logger.exception('SMU Voltage %s out of range' % voltage)

    def smu_get_current(self, n=1):
        ''' Returns the mean and sigma of n consecutive measurements
            For a single measurement, leave n empty
        '''
        rawdata = []
        for _ in range(n):
            self.devices['SMU'].on()
            time.sleep(.1)
            rawdata.append(float(self.devices['SMU'].get_current()))
            self.devices['SMU'].off()
            time.sleep(.1)
        
        logger.debug(rawdata)
        if n >1:
            return np.mean(rawdata), np.std(rawdata)
        else:
            return rawdata[0]

    def xray_control(self, voltage=0, current=0, shutter='close'):
        ''' Sets high voltage and current, operate the shutter
            If no values are given for voltage and current,
            the local_configuration values are used
        '''
        if self.use_xray_control is False:
            logger.warning('X-ray control is not activated')
            return

        if self.voltage > 40:
            raise RuntimeError('Voltage above safe limit of 40 kV')
        if self.current > 50:
            raise RuntimeError('Current above safe limit of 50 mA')

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
            logger.warning('X-ray current ramping up/down: %s mA' % xray_i)
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
                    csv_file.close()
                outerpbar.update(1)

            done = True


class plotting(object):
    ''' Analysis and plotting functions for beam profiles etc.
        Can also be run standalone. In this case, all files in the given folder will be plotted and a calibration file along with calibration plots will be generated
    '''

    # A few examples for DUT dimensions
    _chip_outlines = {
        'rd53a': {'pos_x': 0, 'pos_y': 0, 'size_x': 11.6, 'size_y': 20},
        'itkpix': {'pos_x': -5, 'pos_y': 0, 'size_x': 20, 'size_y': 21},
        'croc': {'pos_x': 0, 'pos_y': 0, 'size_x': 21.6, 'size_y': 18.6},
    }

    def __init__(self, filename=None):
        pass

    def load_data(self, filename=None):
        data = []
        with open(filename, mode='r') as csv_file:
            file_reader = csv.reader(csv_file)
            data = []
            for row in file_reader:
                el = []
                row_float = []
                for el in row:
                    row_float.append(float(el))
                data.append(row_float)
            data = np.array(data).T
            csv_file.close()
        return data

    def convert_data(self, data, background=0, factor=1, scale=0, unit='rad'):
        ''' Converts the raw data in units of [A] or [rad]
            - background: dark current in [A]
            - factor: diode calibration factor
            - scale: adjusts the axis labels
        '''
        if background == 'auto':
            background = np.min(data[2])
        N = int(len(data[2])**.5)
        if unit == 'A':
            z = np.array(scale * (data[2].reshape(N, N) - background))
        else:
            # calculate the dose rate based on the measured delta current in uA and the diode calibration factor
            z = np.array(scale * (data[2].reshape(N, N) - background) * factor / 1000)

        return data, N, z

    def _conv_to_mm(self, pos=0, min=0, max=0, N=1):
        ''' Map indexed position to mm
        '''
        return pos * (max - min) / (N - 1) - abs(min)

    def get_beam_parameters(self, data, z, N):
        (peak_y, peak_x) = np.where(z == np.amax(z))
        xmin, xmax, ymin, ymax = (np.amin(data[0]), np.amax(data[0]), np.amin(data[1]), np.amax(data[1]))
        return peak_x, peak_y, xmin, xmax, ymin, ymax

    def create_profile_plot(self, data, N, z, name='test', unit='rad', distance=0):
        ''' Simple raw data 2D plot of the diode current or dose rate vs. position
        '''
        _, _, xmin, xmax, ymin, ymax = self.get_beam_parameters(data, z, N)
        extent = np.round((xmin, xmax, ymin, ymax), decimals=1)

        fig, ax = plt.subplots()
        im = ax.imshow(np.flip(z, 0), extent=extent, aspect='1', alpha=1)
        ax.set_title('Beam profile (' + name + ')')

        cbar = fig.colorbar(im)
        if unit == 'A':
            label = '$\Delta$ diode current [nA]'
        if unit == 'rad':
            label = 'dose rate in Si$O_2$ [Mrad/h]'
        cbar.ax.set_ylabel(label, fontsize=8)

        plt.tight_layout()
        plt.savefig(name + '_raw.pdf', dpi=200)
        plt.savefig(name + '_raw.png', dpi=200)
        plt.close('all')

    def create_fancy_profile_plot(self, data, N, z, name='test', unit='rad', chip='', distance=0):
        ''' Full beam profile plot with analysis
        '''
        left, width = 0.1, 0.65
        bottom, height = 0.13, 0.65
        bottom_h = left_h = left + width + 0.07
        cbarHeight = 0.02

        rect_color = [left, bottom, width, height]
        rect_histx = [left, bottom_h + 0.02, width, 0.15]
        rect_histy = [left_h, bottom, 0.15, height]

        # get limits from raw data fields
        peak_x, peak_y, xmin, xmax, ymin, ymax = self.get_beam_parameters(data, z, N)
        extent = np.round((xmin, xmax, ymin, ymax), decimals=1)
        histBin = np.linspace(xmin, xmax, N)

        fig = plt.figure(figsize=(9, 9))
        axColor = fig.add_axes(rect_color)

        sumx = z[int((N - 1) / 2)]  # p.sum(z, 0)
        sumxn = sumx / np.amax(sumx)
        sumy = z.T[int((N - 1) / 2)]  # np.sum(z, 1)
        sumyn = sumy / np.amax(sumy)

        # plot image and contour
        peak_intensity = np.amax(z)
        im = plt.imshow(np.flip(z, 0), cmap='viridis', extent=extent, interpolation="bicubic")
        cset = plt.contour(z / peak_intensity, linewidths=.8, cmap='cividis_r', extent=extent)
        axColor.clabel(cset, inline=True, fmt="%1.1f", fontsize=8)
        axColor.set(xlabel='x position [mm]', ylabel='y position [mm]', title='Beam profile (' + name + ')')
        axColor.title.set_position([0.5, 1.01])

        # draw a circle at the peak value
        center_x = 0
        center_y = 0
        radius = (ymax - ymin) / (N - 1) / 2
        peak_xx = (xmax - xmin) / N * (peak_x + 0.5) + xmin
        peak_yy = (ymax - ymin) / N * (peak_y + 0.5) + ymin

        circle = Circle((peak_xx, peak_yy), radius, color='red', fill=False)
        if unit == 'rad':
            label = 'peak: %s Mrad/h \nat x=%.1f mm y=%.1f mm'
        if unit == 'A':
            label = 'peak: %s nA \nat x=%.1f mm y=%.1f mm'
        legend_helper = axColor.plot([], marker='o', markerfacecolor='none', markersize=10, linestyle='', color=circle.get_edgecolor())
        axColor.legend(legend_helper, [label % (np.round(peak_intensity, 2), peak_xx, peak_yy)])
        axColor.add_artist(circle)

        # draw a cross hair, indicating the laser position
        plt.axhline(y=center_y, linewidth=0.5, linestyle='dashed', color='#d62728')
        plt.axvline(x=center_x, linewidth=0.5, linestyle='dashed', color='#d62728')

        # plot cuts
        major_ticks = np.arange(0, 1.1, 0.5)
        minor_ticks = np.arange(0, 1.1, 0.25)

        axHistx = fig.add_axes(rect_histx, xlim=(xmin, xmax), ylim=(-0.05, 1.05))
        axHistx.plot(histBin, sumxn)
        axHistx.set(ylabel='rel. instensity in x')
        axHistx.title.set_position([0.4, 1.05])
        axHistx.set_yticks(major_ticks)
        axHistx.set_yticks(minor_ticks, minor=True)
        axHistx.grid(which='minor', alpha=0.2)
        axHistx.grid(which='major', alpha=0.5)

        thrshld_list = [0.5, 0.2]
        peaks, _ = find_peaks(sumxn, prominence=1, threshold=0.01)
        for thrshld in thrshld_list:
            try:
                results_half = peak_widths(sumxn, peaks, rel_height=thrshld)
                leng = len(results_half[0])
                if leng > 1:
                    ret = (results_half[i][-1] for i in range(len(results_half)))
                    results_half = list(ret)
                axHistx.plot(histBin[peaks], sumxn[peaks], "x", color="C1")
                fwhm_line = float(results_half[1:][0]), self._conv_to_mm(float(results_half[1:][1]), ymin, ymax, N), self._conv_to_mm(float(results_half[1:][2]), ymin, ymax, N)
                axHistx.hlines(*fwhm_line, color="C2")
            except Exception as e:
                print(e)

        axHisty = fig.add_axes(rect_histy, ylim=(ymin, ymax), xlim=(-0.05, 1.05))
        axHisty.plot(sumyn, histBin)
        axHisty.set(xlabel='rel. instensity in y')
        axHisty.title.set_position([0.4, 1.015])
        axHisty.set_xticks(major_ticks)
        axHisty.set_xticks(minor_ticks, minor=True)
        axHisty.grid(which='minor', alpha=0.2)
        axHisty.grid(which='major', alpha=0.5)

        beam_diameter = {}
        peaks, _ = find_peaks(sumyn, distance=10)
        for thrshld in thrshld_list:
            try:
                results_half = peak_widths(sumyn, peaks, rel_height=thrshld)
                leng = len(results_half[0])
                if leng > 1:
                    ret = (results_half[i][-1] for i in range(len(results_half)))
                    results_half = list(ret)
                axHisty.plot(sumyn[peaks], histBin[peaks], "x", color="C1")
                fwhm_line = float(results_half[1:][0]), self._conv_to_mm(float(results_half[1:][1]), ymin, ymax, N), self._conv_to_mm(float(results_half[1:][2]), ymin, ymax, N)
                axHisty.vlines(*fwhm_line, color="C2")
                beam_diameter.update({1 - thrshld: (fwhm_line[2] - fwhm_line[1])})
            except Exception as e:
                print(e)

        axCbar = fig.add_axes([left, 0.05, width, cbarHeight])
        if unit == 'A':
            label = '$\Delta$ diode current [nA]'
        if unit == 'rad':
            label = 'dose rate in Si$O_2$ [Mrad/h]'
        plt.colorbar(im, cax=axCbar, label=label, orientation='horizontal')
        # cbar.set_ticks(np.arange(np.floor(np.amin(z)), np.ceil(np.amax(z)) + 0.1, 0.1))

        # draw the chip outline
        # TODO: Implement chip rotation
        if chip in self._chip_outlines:
            sty = {'xy': (self._chip_outlines[chip]['pos_x'] - self._chip_outlines[chip]['size_x'] / 2,
                          self._chip_outlines[chip]['pos_y'] - self._chip_outlines[chip]['size_y'] / 2),
                   'width': self._chip_outlines[chip]['size_x'],
                   'height': self._chip_outlines[chip]['size_y'],
                   'color': 'red'}

            axColor.add_artist(Rectangle(sty['xy'], sty['width'], sty['height'], color=sty['color'], fill=False))
            axHistx.add_artist(Rectangle(sty['xy'], sty['width'], sty['height'], color=sty['color'], fill=True, alpha=0.2))
            axHisty.add_artist(Rectangle(sty['xy'], sty['width'], sty['height'], color=sty['color'], fill=True, alpha=0.2))

        # Summary
        textstr = '\n'.join((
            r'distance=%.1f cm' % distance,
            r'peak=%.2f Mrad/h' % peak_intensity,
            r'' + '\n'.join("d@{:.0f}%: {:.1f}mm".format(100 * k, v) for k, v in beam_diameter.items()),
            r'DUT={}'.format(chip)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.78, 0.92, textstr, fontsize=13, ha="left", va="center", bbox=props)

        # save the plot
        plt.savefig(name + '.pdf', dpi=200)
        plt.savefig(name + '.png', dpi=200)
        plt.close('all')

        return peak_intensity, beam_diameter

    def plot_data(self, filename=None, background=0, factor=10, scale=1e9, unit='rad', chip='', distance=0):
        ''' Converts rawdata and creates the specified plots
            - background: Measured dark current in [A] or 'auto' to use the minimum value as background
            - factor: diode calibration factor in [Mrad/h/uA]
            - scale: scaling factor for plotting
            - unit: 'rad' or 'A', in case of 'A', the current is plotted
        '''
        # Plot raw data in [A] and before subtracting the background
        data, N, z = self.convert_data(self.load_data(filename), background=0, factor=factor, scale=scale, unit='A')
        self.create_profile_plot(data, N, z, name=filename[:-4], unit='A', distance=distance)
        # Plot the interpolated data in [unit] after subtracting the background
        data, N, z = self.convert_data(self.load_data(filename), background=background, factor=factor, scale=scale, unit=unit)
        peak_intensity, beam_diameter = self.create_fancy_profile_plot(data, N, z, name=filename[:-4], unit=unit, chip=chip, distance=distance)
        return peak_intensity, beam_diameter

    def model_func(self, x, a, b, c):
        # return a / (b * np.power(x, 2)) + c
        return a / np.power((x + b), 2) + c

    def plot_calibration_curves(self, data=None):
        ''' Plots the extracted peak intensity and beam diameter values
            - data: Generated by create_fancy_profile_plot()
        '''
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        distance = []
        peak = []
        diameters = {}

        for el in data:
            try:
                distance.append(el)
                peak.append(data[el]['peak_intensity'])
                diameters[el] = data[el]['beam_diameter']
            except Exception as e:
                print(e)

        plt.title('Intensity and beam diameter as a function of distance')
        ax.set_xlabel('distance from collimator holder [cm]', fontsize=10)
        ax.set_ylabel('peak dose rate in Si$O_2$ [Mrad/h]', fontsize=10)
        ax2.set_xlabel('distance from collimator holder [cm]', fontsize=10)
        ax2.set_ylabel('beam diameter [mm]', fontsize=10)

        # Plot the values for different thresholds
        diameter = []
        lns = {}
        fits = {}
        for it in [0.5, 0.8]:
            temp = []
            for val in diameters:
                temp.append(diameters[val][it])
            diameter.append(temp)
            # Plot diameter vs. distance
            coef = np.polyfit(distance, temp, 1)
            fits.update({it: np.poly1d(coef)})
            ax2.plot(distance, fits[it](distance), linestyle='--', linewidth=0.75)
            lns.update({it: ax2.plot(distance, temp, linestyle='None', marker='o', markersize=5, color=plt.gca().lines[-1].get_color(), label='d @ {:.0f}% peak intensity ({:.1f}x + {:.1f})'.format((it * 100), coef[0], coef[1]))})

        # Plot intensity vs. distance
        popt, pcov = curve_fit(self.model_func, distance, peak, p0=[0, -2, 1], maxfev=1000)
        a, b, c = popt
        logger.info("Optimal parameters are a=%g, b=%g, and c=%g" % (a, b, c))
        dist_range = np.arange(min(distance), max(distance), 0.1)
        intensity_color = ax2._get_lines.get_next_color()
        ax.plot(dist_range, self.model_func(dist_range, a, b, c), linestyle='--', linewidth=0.75, color=intensity_color)
        fits.update({'peak': [a, b, c]})

        lns.update({'peak': ax.plot(distance, peak, linestyle='None', marker='o', markersize=5, color=intensity_color, label='peak intensity ({:.1f}/(x + {:.1f})^2 + {:.2f})'.format(popt[0], popt[1], popt[2]))})
        plt.fill_between(distance, diameter[0], diameter[1], alpha=0.2)

        # Extract the labels and show them in a combined legend
        lines = [ln[0] for ln in lns.values()]
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, loc='upper center', fontsize=9)
        ax.grid()
        plt.tight_layout()
        plt.savefig('calibration' + '.pdf', dpi=200)
        plt.savefig('calibration' + '.png', dpi=200)
        plt.close('all')


if __name__ == '__main__':
    # create plots for all files in the given folder and its subfolders
    path = os.path.join('data', 'calibration')
    os.chdir(path)
    filelist = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith('.csv')]:
            filelist.append(os.path.join(dirpath, filename))

    beamplot = plotting()

    # Optionally, draw the DUT outline
    chip = 'none'

    for filename in filelist:
        try:
            distance = int(filename.split('_')[1].split('cm')[0])
            beamplot.plot_data(filename=filename, background='auto', unit='rad', chip=chip, distance=distance)
            logger.info('Processed "{}"'.format(filename))
        except RuntimeError as e:
            logger.error('Error loading {}/n{}'.format(filename, e))
