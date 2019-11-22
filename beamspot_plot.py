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


class utils(object):
    '''
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

    def convert_data(self, data, background=0, scale=0):
        N = int(len(data[2])**.5)
        z = scale * (data[2].reshape(N, N) - background)
        # z_reshaped = []

        # for idx, row in enumerate(z):
        # #     if idx % 2:
        # #         z_reshaped.append(np.flip(row))
        # #     else:
        #     z_reshaped.append(row)

        return data, N, z

    def plot_intensity_hist(self, z):
        # TODO:
        pass

    def find_beam_parameter(self, z, N, dim):
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
            # Perform a Hough Transform
            hough_radii = range(1, N, 1)
            result = hough_circle(edges, hough_radii)
            accums, cx, cy, radii = hough_circle_peaks(result, hough_radii, total_num_peaks=1)
        except RuntimeError as e:
            logger.error(e)
        return cx, cy, radii

    def create_profile_plot(self, data, N, z, name='test'):
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
        cbar.ax.set_ylabel('$\Delta$ diode current [$n$A]')
#        ax.clabel(im,inline=True,fmt="%1.1f",fontsize=8)
        plt.savefig(name+'_raw', dpi=200)
        plt.savefig('last', dpi=200)
        plt.close('all')

    def create_fancy_profile_plot(self, data, N, z, name='test'):
        left, width = 0.1, 0.65
        bottom, height = 0.13, 0.65
        bottom_h = left_h = left + width + 0.07
        cbarHeight = 0.02

        rect_color = [left, bottom, width, height]
        rect_histx = [left, bottom_h + 0.02, width, 0.15]
        rect_histy = [left_h, bottom, 0.15, height]

        # get limits from raw data fields
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
        im = plt.imshow(np.flip(z, 0), extent=extent, cmap=cm.viridis,
                        interpolation="bicubic")
        cset = plt.contour(z/np.amax(z), linewidths=.8, cmap=cm.cividis_r, extent=extent)
        axColor.clabel(cset, inline=True, fmt="%1.1f", fontsize=8)
        axColor.set(xlabel='x position [mm]', ylabel='y position [mm]',
                    title='Beam profile ('+name+')')
        axColor.title.set_position([0.5, 1.01])

        # also plot a circle fitted to the 10% max intensity contour
        center_x, center_y, radius = self.find_beam_parameter(z, N, extent)
        radius = (xmax - xmin) * radius/N
        # center_x = (xmax - xmin) * center_x/(N-1) + xmin
        # center_y = (ymax - ymin) * center_y/(N-1) + ymin
        center_x = meanx
        center_y = meany
        # circle = Circle((center_x, center_y), radius, color='red', fill=False)
        # axColor.add_artist(circle)

        plt.axhline(y=center_y, linewidth=0.5, linestyle='dashed',
                    color='#d62728')
        plt.axvline(x=center_x, linewidth=0.5, linestyle='dashed',
                    color='#d62728')

        # plot cuts
        axHistx = fig.add_axes(rect_histx, xlim=(xmin, xmax), ylim=(0, 1))
        axHistx.plot(histBin, sumxn)
        axHistx.set(ylabel='rel. instensity in x')
        axHistx.title.set_position([0.4, 1.05])
        axHistx.grid(True, alpha=0.5)

        axHisty = fig.add_axes(rect_histy, ylim=(ymin, ymax), xlim=(0, 1))
        axHisty.plot(sumyn, histBin)
        axHisty.set(xlabel='rel. instensity in y')
        axHisty.title.set_position([0.4, 1.015])
        axHisty.grid(True, alpha=0.5)

        axCbar = fig.add_axes([left, 0.05, width, cbarHeight])
        cbar = plt.colorbar(im, cax=axCbar, label='$\Delta$ diode current [nA]',
                            orientation='horizontal')
#        cbar.set_ticks(np.arange(np.floor(z), np.ceil(z), 0.1))

        # save and show the plot
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(name, dpi=200)
        plt.savefig('last_fancy', dpi=200)
        plt.close('all')
#        plt.show()

    def plot_data(self, filename=None, data=[], z=[], background=0):
        data, N, z = self.convert_data(self.load_data(filename), background=background, scale=10e9)
        #self.find_beam_parameter(np.array(z))
        self.create_profile_plot(data, N, np.array(z), name=filename[:-4])
        self.create_fancy_profile_plot(data, N, np.array(z), name=filename[:-4])


if __name__ == '__main__':
    bs = utils()

    # find all files
    path = 'data/'
    extension = 'csv'
    os.chdir(path)
    filelist = glob.glob('*.{}'.format(extension))

    for filename in filelist:
        logger.info('Processing '+filename+"'")
        try:
            bs.plot_data(filename=filename, background=5.2e-9)
        except RuntimeError as e:
            logger.error('Error loading '+filename+"'", e)

#    bs.plot_data(filename='data/tests/xray_2ma_laser_2019-11-13-19-26.csv')
#    bs.plot_data(filename='data/11cm/xray_2ma_2019-11-13-18-31.csv')
