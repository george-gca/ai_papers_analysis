import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


SUPPORTED_CONFERENCES = [
    'aaai',
    'acl',
    'coling',
    'cvpr',
    'eacl',
    'eccv',
    'emnlp',
    'findings',
    'iccv',
    'iclr',
    'icml',
    'ijcai',
    'ijcnlp',
    'kdd',
    'naacl',
    'neurips',
    # 'neurips_workshop',
    'sigchi',
    'sigdial',
    'siggraph',
    'siggraph-asia',
    'tacl',
    'wacv',
]


CONFERENCES_PDFS = [c for c in SUPPORTED_CONFERENCES if not c.startswith(('kdd', 'sigchi', 'siggraph', 'siggraph-asia'))]


NOT_INFORMATIVE_WORDS = {
    'base',
    'data',
    'deep',
    'learning',
    'method',
    'model',
    'network',
    'problem',
    'result',
    'setting',
    'task',
    'training',
}


def _recreate_url(url_str: str, conference: str, year: int, is_abstract: bool = False) -> str:
    if url_str is None or len(url_str) == 0:
        return url_str

    if url_str.startswith('http://') or url_str.startswith('https://'):
        return url_str

    conference_lower = conference.lower()
    assert conference_lower in set(SUPPORTED_CONFERENCES + ['arxiv']), f'conference is {conference} and url_str is {url_str}'

    if conference_lower == 'aaai':
        if year <= 2018:
            return f'https://www.aaai.org/ocs/index.php/AAAI/AAAI{year % 2000}/paper/viewPaper/{url_str}'
        else:
            return f'https://ojs.aaai.org/index.php/AAAI/article/view/{url_str}'

    # acl conferences
    elif conference_lower in {'acl', 'coling', 'eacl', 'emnlp', 'findings', 'ijcnlp', 'naacl', 'sigdial', 'tacl'}:
        return f'https://aclanthology.org/{url_str}'

    # arxiv
    elif conference_lower == 'arxiv':
        if is_abstract:
            url_type = 'abs'
            url_ext = ''
        else:
            url_type = 'pdf'
            url_ext = '.pdf'

        return f'https://arxiv.org/{url_type}/{url_str}{url_ext}'

    # thecvf conferences
    elif conference_lower in {'cvpr', 'iccv', 'wacv'}:
        return f'https://openaccess.thecvf.com/{url_str}'

    elif conference_lower == 'eccv':
        if is_abstract:
            url_type = 'html'
            url_ext = '.php'
        else:
            url_type = 'papers'
            url_ext = '.pdf'

        return f'https://www.ecva.net/papers/eccv_{year}/papers_ECCV/{url_type}/{url_str}{url_ext}'

    elif conference_lower in {'iclr', 'neurips_workshop'} or \
        (conference_lower == 'neurips' and 2022 <= year <= 2023) or \
        (conference_lower == 'icml' and year == 2024):

        if is_abstract:
            url_type = 'forum'
        else:
            url_type = 'pdf'

        return f'https://openreview.net/{url_type}?id={url_str}'

    elif conference_lower == 'icml':
        if is_abstract:
            url_ext = '.html'
        else:
            url_ext = f'/{url_str.split("/")[1]}.pdf'

        return f'http://proceedings.mlr.press/{url_str}{url_ext}'

    elif conference_lower == 'ijcai':
        return f'https://www.ijcai.org/proceedings/{year}/{url_str}'

    elif conference_lower == 'kdd':
        if year == 2017:
            return f'https://www.kdd.org/kdd{year}/papers/view/{url_str}'
        elif year == 2018 or year == 2020:
            return f'https://www.kdd.org/kdd{year}/accepted-papers/view/{url_str}'
        else: # if year == 2021:
            return f'https://dl.acm.org/doi/abs/{url_str}'

    elif conference_lower == 'neurips':
        if is_abstract:
            url_type = 'hash'
        else:
            url_type = 'file'

        return f'https://papers.nips.cc/paper/{year}/{url_type}/{url_str}'

    elif conference_lower == 'sigchi' or conference_lower in {'siggraph', 'siggraph-asia'}:
        return f'https://dl.acm.org/doi/abs/{url_str}'

    return url_str


def recreate_url_from_code(url_str: str, code: int, conference: str, year: int, is_abstract: bool = False) -> str:
    if url_str is None or len(url_str) == 0 or url_str.startswith(('http://', 'https://')):
        return url_str

    if code < 0:
        return _recreate_url(url_str, conference, year, is_abstract)

    conference_lower = conference.lower()
    assert conference_lower in SUPPORTED_CONFERENCES, f'conference is {conference} and url_str is {url_str}'

    if code == 1:
        if year <= 2018:
            return f'https://www.aaai.org/ocs/index.php/AAAI/AAAI{year % 2000}/paper/viewPaper/{url_str}'
        else:
            return f'https://ojs.aaai.org/index.php/AAAI/article/view/{url_str}'

    # acl conferences
    elif code == 2:
        return f'https://aclanthology.org/{url_str}'

    # arxiv
    elif code == 10:
        if is_abstract:
            url_type = 'abs'
            url_ext = ''
        else:
            url_type = 'pdf'
            url_ext = '.pdf'

        return f'https://arxiv.org/{url_type}/{url_str}{url_ext}'

    # thecvf conferences
    elif code == 9:
        return f'https://openaccess.thecvf.com/{url_str}'

    elif code == 3:
        if is_abstract:
            url_type = 'html'
            url_ext = '.php'
        else:
            url_type = 'papers'
            url_ext = '.pdf'

        return f'https://www.ecva.net/papers/eccv_{year}/papers_ECCV/{url_type}/{url_str}{url_ext}'

    elif code == 0:
        if is_abstract:
            url_type = 'forum'
        else:
            url_type = 'pdf'

        return f'https://openreview.net/{url_type}?id={url_str}'

    elif code == 6:
        if is_abstract:
            url_ext = '.html'
        else:
            url_ext = f'/{url_str.split("/")[1]}.pdf'

        return f'http://proceedings.mlr.press/{url_str}{url_ext}'

    elif code == 4:
        return f'https://www.ijcai.org/proceedings/{year}/{url_str}'

    elif code == 5:
        if year == 2017:
            return f'https://www.kdd.org/kdd{year}/papers/view/{url_str}'
        elif year == 2018 or year == 2020:
            return f'https://www.kdd.org/kdd{year}/accepted-papers/view/{url_str}'
        else: # if year == 2021:
            return f'https://dl.acm.org/doi/abs/{url_str}'

    elif code == 7:
        if is_abstract:
            url_type = 'hash'
        else:
            url_type = 'file'

        return f'https://papers.nips.cc/paper/{year}/{url_type}/{url_str}'

    elif code == 8 or code == 11:
        return f'https://dl.acm.org/doi/abs/{url_str}'

    return url_str


def setup_log(
        log_level: str = 'warning',
        log_file: str | Path = Path('run.log'),
        file_log_level: str = 'info',
        logs_to_silence: list[str] = [],
        ) -> None:
    """
    Setup the logging.

    Args:
        log_level (str): stdout log level. Defaults to 'warning'.
        log_file (str | Path): file where the log output should be stored. Defaults to 'run.log'.
        file_log_level (str): file log level. Defaults to 'info'.
        logs_to_silence (list[str]): list of loggers to be silenced. Useful when using log level < 'warning'. Defaults to [].
    """
    # TODO: fix this according to this
    # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
    # https://www.electricmonk.nl/log/2017/08/06/understanding-pythons-logging-module/
    logging.PRINT = 60
    logging.addLevelName(60, 'PRINT')

    def log_print(self, message, *args, **kws):
        if self.isEnabledFor(logging.PRINT):
            # Yes, logger takes its '*args' as 'args'.
            self._log(logging.PRINT, message, args, **kws)

    logging.Logger.print = log_print

    # convert log levels to int
    int_log_level = {
        'debug': logging.DEBUG,  # 10
        'info': logging.INFO,  # 20
        'warning': logging.WARNING,  # 30
        'error': logging.ERROR,  # 40
        'critical': logging.CRITICAL,  # 50
        'print': logging.PRINT,  # 60
    }

    log_level = int_log_level[log_level]
    file_log_level = int_log_level[file_log_level]

    # create a handler to log to stderr
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(log_level)

    # create a logging format
    if log_level >= logging.WARNING:
        stderr_formatter = logging.Formatter('{message}', style='{')
    else:
        stderr_formatter = logging.Formatter(
            # format:
            # <10 = pad with spaces if needed until it reaches 10 chars length
            # .10 = limit the length to 10 chars
            '{name:<10.10} [{levelname:.1}] {message}', style='{')
    stderr_handler.setFormatter(stderr_formatter)

    # create a file handler that have size limit
    if isinstance(log_file, str):
        log_file = Path(log_file).expanduser()

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)  # ~ 5 MB
    file_handler.setLevel(file_log_level)

    # https://docs.python.org/3/library/logging.html#logrecord-attributes
    file_formatter = logging.Formatter(
        '{asctime} - {name:<12.12} {levelname:<8} {message}', datefmt='%Y-%m-%d %H:%M:%S', style='{')
    file_handler.setFormatter(file_formatter)

    # add the handlers to the root logger
    logging.basicConfig(handlers=[file_handler, stderr_handler], level=logging.DEBUG)

    # change logger level of logs_to_silence to warning
    for other_logger in logs_to_silence:
        logging.getLogger(other_logger).setLevel(logging.WARNING)

    # create logger
    logger = logging.getLogger(__name__)

    logger.info(f'Saving logs to {log_file.absolute()}')
    logger.info(f'Log level: {logging.getLevelName(log_level)}')
