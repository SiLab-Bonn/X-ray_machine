import csv
import logging
import coloredlogs
import xray

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

_local_config = {
    'xray_use_remote': False,
    'smu_use_bias': True,
    'smu_diode_bias': -50,
    'smu_current_limit': 1.0E-05
}

class test_cuuent(object):
    pass


if __name__ == '__main__':
    scan = xray.utils(**_local_config)
    scan.init_smu(voltage=_local_config['smu_diode_bias'], current_limit=_local_config['smu_current_limit'])

    with open('data/logger.csv', mode='a') as csv_file:
        file_writer = csv.writer(csv_file)
        for i in range(3600):
            background, std = scan.smu_get_current(10)
            logger.info('current = {:.3e} ({:.3e})'.format(background, std))
            file_writer.writerow([background, std])
