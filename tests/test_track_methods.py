import unittest

import os
import pickle
import numpy as np

from trait2d.analysis import Track

with open(os.path.dirname(__file__) + '/test_data/track.pickle', 'rb') as handle:
    TRACK_DATA = pickle.load(handle)

class TestTrackMethods(unittest.TestCase):

    # Test whether a Track object can be constructed from a dictionary.
    def test_from_dict(self):
        track = Track.from_dict(TRACK_DATA['track'])
        self.assertEqual(track.get_size(), len(TRACK_DATA['track']['x']))

    # Test whether the MSD and the error is calculated correctly for artificial dataset.
    def test_calculate_msd(self):
        track = Track.from_dict(TRACK_DATA['track'])
        track.calculate_msd()
        self.assertTrue(track.is_msd_calculated())
        self.assertTrue(np.isclose(track.get_msd(), TRACK_DATA['msd']).all())
        self.assertTrue(np.isclose(track.get_msd_error(), TRACK_DATA['msd_error']).all())

    # Test whether ADC analysis produces results for artificial dataset.
    # (Note: Does not check for actual results of analysis as these might change between versions.)
    def test_adc_analysis(self):
        track = Track.from_dict(TRACK_DATA['track'])
        track.adc_analysis()
        self.assertTrue(track.get_adc_analysis_results() != None)

    # Test whether MSD analysis produces results for artificial dataset.
    # (Note: Does not check for actual results of analysis as these might change between versions.)
    def test_msd_analysis(self):
        track = Track.from_dict(TRACK_DATA['track'])
        track.msd_analysis()
        self.assertTrue(track.get_msd_analysis_results() != None)

if __name__ == '__main__':
    unittest.main()