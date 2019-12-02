#
# ------------------------------------------------------------
# Copyright (c) All rights reserved
# SiLab, Institute of Physics, University of Bonn
# ------------------------------------------------------------
#
import time
import datetime
import math
import numpy as np
import tables as tb
import csv
import os
import random
import glob
import logging
import coloredlogs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from matplotlib.backends.backend_pdf import PdfPages
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.interpolate import interp2d
from scipy.stats import norm


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


class plot(object):
    ''' Plotting functions for beam profiles etc.
        Can also be run standalone. In this case, all files in the given folder will be plotted
    '''

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

        return data

    def convert_data(self, data, background=0, factor=1, scale=0, unit='rad'):
        ''' Converts the raw data in units of [A] or [rad]
                background: dark current in [A]
                factor: diode calibration factor
                scale: adjusts the axis labels
        '''
        if background == 'auto':
            background = np.min(data[2])
        N = int(len(data[2])**.5)
        if unit == 'A':
            z = scale * (data[2].reshape(N, N) - background)
        else:
            # calculate the dose rate based on the measured delta current in uA and the diode calibration factor
            z = scale * (data[2].reshape(N, N) - background) * factor / 1000

        return data, N, z

    def find_beam_parameters(self, z, N, dim):
        try:
            # x = np.linspace(dim[0], dim[1], N)
            # y = np.linspace(dim[2], dim[3], N)
            # f = interp2d(x, y, z, kind='linear')
            # x2 = np.linspace(dim[0], dim[1], N*4)
            # y2 = np.linspace(dim[2], dim[3], N*4)
            # z2 = f(x2, y2)
#            fig, ax = plt.subplots()
#            ax.imshow(np.flip(z2, 0))
#            plt.show()
            # Find edges
            cut = np.amax(z) * 0.1
            edges = canny(z)
            # Perform a Hough transform
            hough_radii = range(1, N, 1)
            result = hough_circle(edges, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(result, hough_radii, total_num_peaks=1)
        except RuntimeError as e:
            logger.error(e)
        return cx, cy, radii

    def create_profile_plot(self, data, N, z, name='test', unit='rad'):
        fig, ax = plt.subplots()

        extent = (np.amin(data[0]), np.amax(data[0]),
                  np.amin(data[1]), np.amax(data[1]))

        im = ax.imshow(np.flip(z, 0), extent=extent, aspect='1', alpha=1)
#        imc = ax.contour(np.flip(z,0), extent=extent, cmap=cm.cividis_r,
#                         interpolation="bilinear", aspect='1', alpha=1)

        ax.set_title('Beam profile ('+name+')')
        ax.set_xlabel("x position [mm]")
        ax.set_ylabel("y position [mm]")

        cbar = fig.colorbar(im)
        if unit == 'A':
            label = '$\Delta$ diode current [nA]'
        if unit == 'rad':
            label = 'dose rate in Si$O_2$ [Mrad/h]'
        cbar.ax.set_ylabel(label)
#        ax.clabel(im,inline=True,fmt="%1.1f",fontsize=8)
        plt.savefig(name+'_raw', dpi=200)
        plt.close('all')

    def create_fancy_profile_plot(self, data, N, z, name='test', unit='rad'):
        left, width = 0.1, 0.65
        bottom, height = 0.13, 0.65
        bottom_h = left_h = left + width + 0.07
        cbarHeight = 0.02

        rect_color = [left, bottom, width, height]
        rect_histx = [left, bottom_h + 0.02, width, 0.15]
        rect_histy = [left_h, bottom, 0.15, height]

        # get limits from raw data fields
        (peak_x, peak_y) = np.where(z == np.amax(z))
        extent = np.round((np.amin(data[0]), np.amax(data[0]),
                           np.amin(data[1]), np.amax(data[1])), decimals=1)
        xmin, xmax, ymin, ymax = extent[0], extent[1], extent[2], extent[3]
        histBin = np.linspace(xmin, xmax, N)

        fig = plt.figure(figsize=(9, 9))
        axColor = fig.add_axes(rect_color)

        sumx = np.sum(z, 0)
        sumxn = sumx / np.amax(sumx)
        meanx, stdx = norm.fit(sumxn)

        sumy = np.sum(z, 1)
        sumyn = sumy / np.amax(sumy)
        meany, stdy = norm.fit(sumyn)

        # plot image and contour
        im = plt.imshow(np.flip(z, 0), extent=extent, cmap='viridis',
                        interpolation="bicubic")
        cset = plt.contour(z/np.amax(z), linewidths=.8, cmap='cividis_r', extent=extent)
        axColor.clabel(cset, inline=True, fmt="%1.1f", fontsize=8)
        axColor.set(xlabel='x position [mm]', ylabel='y position [mm]',
                    title='Beam profile ('+name+')')
        axColor.title.set_position([0.5, 1.01])
        # axColor.annotate('peak:%s Mrad/h' %np.round(np.amax(z),2), xy=((xmax-xmin)/N*peak_y+xmin+1, (ymax-ymin)/N*peak_x+ymin+1))

        center_x, center_y, radius = self.find_beam_parameters(z, N, extent)
        radius = (xmax - xmin) * radius/N
        center_x = 0
        center_y = 0
        circle = Circle(((xmax-xmin)/N*peak_y+xmin, (ymax-ymin)/N*peak_x+ymin), 1, color='red', fill=False)
        axColor.legend([circle], ['peak:%s Mrad/h' % np.round(np.amax(z), 2)])
        axColor.add_artist(circle)

        plt.axhline(y=center_y, linewidth=0.5, linestyle='dashed',
                    color='#d62728')
        plt.axvline(x=center_x, linewidth=0.5, linestyle='dashed',
                    color='#d62728')

        # plot cuts
        major_ticks = np.arange(0, 1.1, 0.5)
        minor_ticks = np.arange(0, 1.1, 0.25)

        axHistx = fig.add_axes(rect_histx, xlim=(xmin, xmax), ylim=(0, 1))
        axHistx.plot(histBin, sumxn)
        axHistx.set(ylabel='rel. instensity in x')
        axHistx.title.set_position([0.4, 1.05])
        axHistx.set_yticks(major_ticks)
        axHistx.set_yticks(minor_ticks, minor=True)
        axHistx.grid(which='minor', alpha=0.2)
        axHistx.grid(which='major', alpha=0.5)

        axHisty = fig.add_axes(rect_histy, ylim=(ymin, ymax), xlim=(0, 1))
        axHisty.plot(sumyn, histBin)
        axHisty.set(xlabel='rel. instensity in y')
        axHisty.title.set_position([0.4, 1.015])
        axHisty.set_xticks(major_ticks)
        axHisty.set_xticks(minor_ticks, minor=True)
        axHisty.grid(which='minor', alpha=0.2)
        axHisty.grid(which='major', alpha=0.5)

        axCbar = fig.add_axes([left, 0.05, width, cbarHeight])
        if unit == 'A':
            label = '$\Delta$ diode current [nA]'
        if unit == 'rad':
            label = 'dose rate in Si$O_2$ [Mrad/h]'
        cbar = plt.colorbar(im, cax=axCbar, label=label,
                            orientation='horizontal')
        #cbar.set_ticks(np.arange(np.floor(np.amin(z)), np.ceil(np.amax(z))+0.1, 0.1))

        # save and show the plot
        plt.savefig(name, dpi=200)
        plt.close('all')

    def plot_data(self, filename=None, background=0, factor=10, scale=1e9, unit='rad'):
        ''' Converts rawdata and creates the specified plots
            background: Measured dark current in [A] or 'auto' to use the minimum value as background
            factor: diode calibration factor in [Mrad/h uA]
            scale: scaling factor for plotting
            unit: 'rad' or 'A', in case of 'A', the current is plotted
        '''
        # Plot raw data in [A] and before subtracting the background
        data, N, z = self.convert_data(self.load_data(filename), background=0, factor=factor, scale=scale, unit='A')
        self.create_profile_plot(data, N, np.array(z), name=filename[:-4], unit='A')
        # Plot the interpolated data in [unit] after subtracting the background
        data, N, z = self.convert_data(self.load_data(filename), background=background, factor=factor, scale=scale, unit=unit)
        self.create_fancy_profile_plot(data, N, np.array(z), name=filename[:-4], unit=unit)


if __name__ == '__main__':
    beamplot = plot()

    # create plots for all files in the given folder
    path = 'data/calibration/45cm'
    extension = 'csv'
    os.chdir(path)
    filelist = glob.glob('*.{}'.format(extension))

    for filename in filelist:
        logger.info('Processing '+filename+"'")
        try:
            beamplot.plot_data(filename=filename, background='auto', unit='rad')
        except RuntimeError as e:
            logger.error('Error loading '+filename+"'", e)
