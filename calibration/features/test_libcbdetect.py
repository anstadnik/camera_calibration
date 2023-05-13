import unittest


class TestImports(unittest.TestCase):
    def test_imports(self):
        try:
            from cbdetect_py import (  # noqa: unused-imports
                CornerType,
                Params,
                boards_from_corners,
                find_corners,
            )

            self.assertTrue(True)
        except ImportError:
            self.fail("ImportError: One or more imports failed")
