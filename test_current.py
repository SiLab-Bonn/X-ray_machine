import csv
import logging
import coloredlogs

import xray

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

class scan_beamspot(object):
    pass


if __name__ == '__main__':
    scan = xray.utils()

    with open('data/logger.csv', mode='a') as csv_file:
        file_writer = csv.writer(csv_file)
        for i in range(3600):
            background, std = scan.smu_get_current(10)
            logger.info('background current = %s %s' % (background, std))
            file_writer.writerow([background, std])