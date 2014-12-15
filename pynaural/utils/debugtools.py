import logging, os.path, time, traceback

__all__ = ['log_debug', 'log_info', 'log_warn',
           #'logging', 'setup_logging', 'logger',
           'debug_level', 'info_level', 'warning_level']

def setup_logging(level):
    if level == logging.DEBUG:
        logging.basicConfig(level=level,
                            format="%(asctime)s,%(msecs)03d  %(message)s",
                            datefmt='%H:%M:%S')
    else:
        logging.basicConfig(level=level,
                            format="%(message)s")
    return logging.getLogger(__name__)

def get_caller():
    tb = traceback.extract_stack()[-3]
    module = os.path.splitext(os.path.basename(tb[0]))[0].strip()
    line = str(tb[1]).ljust(4)
    func = tb[2].ljust(18)
#    return "L:%s  %s  %s" % (line, module, func)
    return "line %s in %s.%s\n............  " % (line, module,func)

def log_debug(obj):
    # HACK: bug fix for deadlocks when logger level is not debug
    time.sleep(.0005)
    if level == logging.DEBUG:
        string = str(obj)
        string = get_caller() + string
        logger.debug(string)

def log_info(str):
    if level == logging.DEBUG:
        str = get_caller() + str
    logger.info(str)

def log_warn(str):
    if level == logging.DEBUG:
        str = get_caller() + str
    logger.warn(str)

def debug_level():
    logger.setLevel(logging.DEBUG)

def info_level():
    logger.setLevel(logging.INFO)

def warning_level():
    logger.setLevel(logging.WARNING)

# Set logging level to INFO by default
level = eval('logging.%s' % 'DEBUG')
logger = setup_logging(level)

if __name__ == '__main__':
    log_debug("hello world")
