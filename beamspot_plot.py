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
import logging
import coloredlogs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import random
import os
import glob

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

    def convert_data(self, data):
        N = int(len(data[2])**.5)
        z = data[2].reshape(N, N)
        z_reshaped = []

        for idx, row in enumerate(z):
            if idx % 2:
                z_reshaped.append(np.flip(row))
            else:
                z_reshaped.append(row)

        return data, N, z_reshaped

    def create_profile_plot(self, data, N, z, name='test'):
        fig, ax = plt.subplots()

        extent = (np.amin(data[0]), np.amax(data[0]),
                  np.amin(data[1]), np.amax(data[1]))

        im = ax.imshow(np.flip(z, 0), extent=extent, aspect='1', alpha=1)
#        imc = ax.contour(np.flip(z,0), extent=extent, cmap=cm.cividis_r,
#                         interpolation="bilinear", aspect='1', alpha=1)

        ax.set_title('Beam profile ('+name+')')
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('diode current [$n$A]')

#        plt.clabel(imc,inline=True,fmt="%1.1f",fontsize=8)

        plt.savefig(name+'_raw', dpi=200)
        plt.close('all')
#        plt.savefig(fname=self.filename[:-4], dpi=200)
#        plt.show()

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
        histBin = np.linspace(extent[0], extent[1], N)

        fig = plt.figure(figsize=(9, 9))
        axColor = fig.add_axes(rect_color)

        # plot image and contour
        im = plt.imshow(np.flip(z, 0), extent=extent, cmap=cm.viridis, interpolation="bicubic")
        cset = plt.contour(z, linewidths=.8, cmap=cm.cividis_r, extent=extent)  #ToDO: Why flip?
        axColor.clabel(cset, inline=True, fmt="%1.1f", fontsize=8)
        axColor.set(xlabel='x position [mm]', ylabel='y position [mm]', title='Beam profile ('+name+')')
        axColor.title.set_position([0.5, 1.01])

        # plot cuts
        axHistx = fig.add_axes(rect_histx, xlim=(extent[0], extent[1]))
        sumx = np.sum(z, 0)
        sumxn = sumx / np.amax(sumx)
        axHistx.plot(histBin, sumxn)
        axHistx.set(ylabel='rel. instensity')  # title='x distribution'
        axHistx.title.set_position([0.4, 1.05])
        axHistx.grid(True, alpha=0.5)

        sumy = np.sum(z, 1)
        sumyn = sumy / np.amax(sumy)
        axHisty = fig.add_axes(rect_histy, ylim=(extent[2], extent[3]))
        axHisty.plot(sumyn, histBin)
        axHisty.set(xlabel='rel. instensity')  # title='y distribution'
        axHisty.title.set_position([0.4, 1.015])
        axHisty.grid(True, alpha=0.5)

        axCbar = fig.add_axes([left, 0.05, width, cbarHeight])
        cbar = plt.colorbar(im, cax=axCbar, label='diode current [A]', orientation='horizontal')
#        cbar.set_ticks(np.arange(np.floor(z), np.ceil(z), 0.1))

        # save and show the plot
#        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig(name, dpi=200)
        plt.close('all')
#        plt.show()

    def plot_data(self, filename=None, data=[], z=[]):
        data, N, z = self.convert_data(self.load_data(filename))
        self.create_profile_plot(data, N, np.array(z)*10e9, name=filename[:-4])
        self.create_fancy_profile_plot(data, N, np.array(z)*10e9, name=filename[:-4])


if __name__ == '__main__':
    bs = utils()

    # find all files
    path = 'data/27cm'
    extension = 'csv'
    os.chdir(path)
    filelist = glob.glob('*.{}'.format(extension))

    for filename in filelist:
        logger.info('Processing '+filename+"'")
        try:
            bs.plot_data(filename=filename)
        except:
            logger.error('Error loading '+filename+"'")

#    bs.plot_data(filename='data/tests/xray_2ma_laser_2019-11-13-19-26.csv')
#    bs.plot_data(filename='data/11cm/xray_2ma_2019-11-13-18-31.csv')
