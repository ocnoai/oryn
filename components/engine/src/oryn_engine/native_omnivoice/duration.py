# Copyright 2026 Xiaomi Corp.; Copyright 2025 Boson AI and HuggingFace.
# Adapted under Apache-2.0; see LICENSE.txt and provenance.json.
"""Text duration estimation for TTS generation.

Provides ``RuleDurationEstimator``, which estimates audio duration from text
using character phonetic weights across 600+ languages. Used by
``OmniVoice.generate()`` to determine output length when no duration is specified.
"""
import bisect
import unicodedata
from functools import lru_cache
from typing import Optional

class RuleDurationEstimator:

    def __init__(self):
        self.weights = {'cjk': 3.0, 'hangul': 2.5, 'kana': 2.2, 'ethiopic': 3.0, 'yi': 3.0, 'indic': 1.8, 'thai_lao': 1.5, 'khmer_myanmar': 1.8, 'arabic': 1.5, 'hebrew': 1.5, 'latin': 1.0, 'cyrillic': 1.0, 'greek': 1.0, 'armenian': 1.0, 'georgian': 1.0, 'punctuation': 0.5, 'space': 0.2, 'digit': 3.5, 'mark': 0.0, 'default': 1.0}
        self.ranges = [(687, 'latin'), (1023, 'greek'), (1327, 'cyrillic'), (1423, 'armenian'), (1535, 'hebrew'), (1919, 'arabic'), (2207, 'arabic'), (2303, 'arabic'), (2431, 'indic'), (2559, 'indic'), (2687, 'indic'), (2815, 'indic'), (2943, 'indic'), (3071, 'indic'), (3199, 'indic'), (3327, 'indic'), (3455, 'indic'), (3583, 'indic'), (3839, 'thai_lao'), (4095, 'indic'), (4255, 'khmer_myanmar'), (4351, 'georgian'), (4607, 'hangul'), (4991, 'ethiopic'), (5023, 'ethiopic'), (5119, 'default'), (5759, 'default'), (5791, 'default'), (5887, 'default'), (5919, 'default'), (5951, 'default'), (5983, 'default'), (6015, 'default'), (6143, 'khmer_myanmar'), (6319, 'default'), (6399, 'default'), (6479, 'indic'), (6623, 'indic'), (6655, 'khmer_myanmar'), (6687, 'indic'), (6831, 'indic'), (7039, 'indic'), (7103, 'indic'), (7167, 'indic'), (7247, 'indic'), (7295, 'indic'), (7311, 'cyrillic'), (7359, 'georgian'), (7375, 'indic'), (7423, 'indic'), (7551, 'latin'), (7615, 'latin'), (7679, 'default'), (7935, 'latin'), (12447, 'kana'), (12543, 'kana'), (12591, 'cjk'), (12687, 'hangul'), (40959, 'cjk'), (42191, 'yi'), (42239, 'default'), (42559, 'default'), (42655, 'cyrillic'), (42751, 'default'), (43007, 'latin'), (43055, 'indic'), (43135, 'default'), (43231, 'indic'), (43263, 'indic'), (43311, 'indic'), (43359, 'indic'), (43391, 'hangul'), (43487, 'indic'), (43519, 'khmer_myanmar'), (43615, 'indic'), (43647, 'khmer_myanmar'), (43743, 'indic'), (43775, 'indic'), (43823, 'ethiopic'), (43887, 'latin'), (43967, 'default'), (44031, 'indic'), (55215, 'hangul'), (64255, 'cjk'), (65023, 'arabic'), (65135, 'default'), (65279, 'arabic'), (65519, 'latin')]
        self.breakpoints = [r[0] for r in self.ranges]

    @lru_cache(maxsize=4096)
    def _get_char_weight(self, char):
        """Determines the weight of a single character."""
        code = ord(char)
        if 65 <= code <= 90 or 97 <= code <= 122:
            return self.weights['latin']
        if code == 32:
            return self.weights['space']
        if code == 1600:
            return self.weights['mark']
        category = unicodedata.category(char)
        if category.startswith('M'):
            return self.weights['mark']
        if category.startswith('P') or category.startswith('S'):
            return self.weights['punctuation']
        if category.startswith('Z'):
            return self.weights['space']
        if category.startswith('N'):
            return self.weights['digit']
        idx = bisect.bisect_left(self.breakpoints, code)
        if idx < len(self.ranges):
            script_type = self.ranges[idx][1]
            return self.weights.get(script_type, self.weights['default'])
        if code > 131072:
            return self.weights['cjk']
        return self.weights['default']

    def calculate_total_weight(self, text):
        """Sums up the normalized weights for a string."""
        return sum((self._get_char_weight(c) for c in text))

    def estimate_duration(self, target_text: str, ref_text: str, ref_duration: float, low_threshold: Optional[float]=50, boost_strength: float=3) -> float:
        """

        Args:
            target_text (str): The text for which we want to estimate the duration.
            ref_text (str): The reference text that was used to measure
                the ref_duration.
            ref_duration (float): The actual duration it took
                to speak the ref_text.
            low_threshold (float): The minimum duration threshold below which the
                estimation will be considered unreliable.
            boost_strength (float): Controls the power-curve boost for short durations.
                Higher values boost small durations more aggressively.
                1 = no boost (linear), 2 = sqrt-like

        Returns:
            float: The estimated duration for the target_text based
                on the ref_text and ref_duration.
        """
        if ref_duration <= 0 or not ref_text:
            return 0.0
        ref_weight = self.calculate_total_weight(ref_text)
        if ref_weight == 0:
            return 0.0
        speed_factor = ref_weight / ref_duration
        target_weight = self.calculate_total_weight(target_text)
        estimated_duration = target_weight / speed_factor
        if low_threshold is not None and estimated_duration < low_threshold:
            alpha = 1.0 / boost_strength
            return low_threshold * (estimated_duration / low_threshold) ** alpha
        else:
            return estimated_duration
