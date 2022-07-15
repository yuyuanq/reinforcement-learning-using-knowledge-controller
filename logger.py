from colorlog import ColoredFormatter
import colorlog

handler = colorlog.StreamHandler()
formatter = ColoredFormatter(
    fmt='%(log_color)s' + '%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%',
)
handler.setFormatter(formatter)
logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel('DEBUG')
