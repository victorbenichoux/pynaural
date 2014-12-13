from unittest import TestCase

__author__ = 'victorbenichoux'


class TestImports(TestCase):
    def test_import(self):
        exec('''from pynaural.signal import *''')
        exec('''from pynaural.raytracer import *; from pynaural.binaural import *; from pynaural.utils import *''')
        exec('''from pynaural import *''')