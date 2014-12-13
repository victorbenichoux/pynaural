from unittest import TestCase

__author__ = 'victorbenichoux'


class TestImports(TestCase):
    def test_global_import(self):
        exec('''from pynaural import *''')

    def test_signal_import(self):
        exec('''from pynaural.signal import *''')

    def test_raytracer_import(self):
        exec('''from pynaural.raytracer import *;''')

    def test_binaural_import(self):
        exec('''from pynaural.binaural import *;''')

    def test_utils_import(self):
        exec('''from pynaural.utils import *''')
