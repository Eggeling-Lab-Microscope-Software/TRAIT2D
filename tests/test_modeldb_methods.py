import unittest

from trait2d.analysis import ModelDB

class TestModelDBMethods(unittest.TestCase):

    # Test whether all predefined models can be imported.
    def test_import_models(self):
        from trait2d.analysis.models import ModelBrownian
        from trait2d.analysis.models import ModelConfined
        from trait2d.analysis.models import ModelHop
        from trait2d.analysis.models import ModelImmobile
        from trait2d.analysis.models import ModelHopModified

    def test_model_db(self):
        from trait2d.analysis.models import ModelConfined
        from trait2d.analysis.models import ModelHop
        ModelDB().add_model(ModelConfined)
        ModelDB().add_model(ModelHop)
        self.assertTrue(isinstance(ModelDB().get_model(ModelConfined), ModelConfined))
        self.assertTrue(isinstance(ModelDB().get_model(ModelHop), ModelHop))
        ModelDB().remove_model(ModelConfined)
        with self.assertRaises(ValueError):
            ModelDB().get_model(ModelConfined)
        self.assertTrue(isinstance(ModelDB().get_model(ModelHop), ModelHop))
        ModelDB().cleanup()
        with self.assertRaises(ValueError):
            ModelDB().get_model(ModelHop)

if __name__ == '__main__':
    unittest.main()