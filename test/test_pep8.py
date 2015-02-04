import unittest
from glob import glob

import pep8


class TestCodeFormat(unittest.TestCase):
    ERROR_MSG = "Please fix your code style errors."

    def test_pep8_conformance_core(self):
        files = glob('bernet/*.py') + glob('test/*.py')

        pep8style = pep8.StyleGuide()
        result = pep8style.check_files(files)
        self.assertEqual(result.total_errors, 0, self.ERROR_MSG)
