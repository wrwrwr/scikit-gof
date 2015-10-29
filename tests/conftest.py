from unittest import TestCase

from pytest import skip


def pytest_addoption(parser):
    parser.addoption('--slow', action='store_true',
                     help="run tests that may take a few minutes each")
    parser.addoption('--veryslow', action='store_true',
                     help="run tests that may take a few hours each")
    parser.addoption('--veryveryslow', action='store_true',
                     help="run tests that may take a few days each")
    parser.addoption('--veryveryveryslow', action='store_true',
                     help="run tests that are hardly doable")


def pytest_runtest_setup(item):
    for mark in ('slow', 'veryslow', 'veryveryslow', 'veryveryveryslow'):
        # Workaround: Tests marked as (very)+slow get a slow marker too.
        if mark == 'slow' and (item.config.getoption('--veryslow') or
                               item.config.getoption('--veryveryslow') or
                               item.config.getoption('--veryveryveryslow')):
            continue
        option = '--' + mark
        if mark in item.keywords and not item.config.getoption(option):
            skip("only run with the {} option".format(option))


# Standarize some unittest.TestCase method names.
def set_up(self, *args, **kwargs):
    if hasattr(self, 'set_up'):
        self.set_up(*args, **kwargs)
    self._old_set_up(*args, **kwargs)


def tear_down(self, *args, **kwargs):
    if hasattr(self, 'tear_down'):
        self.tear_down(*args, **kwargs)
    self._old_tear_down(*args, **kwargs)


TestCase._old_set_up = TestCase.setUp
TestCase._old_tear_down = TestCase.tearDown
TestCase.setUp = set_up
TestCase.tearDown = tear_down
