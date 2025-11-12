from typing import Tuple

from .alphabet_model import AlphabetModel
from .diacritics_model import DiacriticsModel


class PredictionModel:
    '''Encapsulates the entire prediction combining alphabet and diacritics models.'''
    IDX_WRIST = 0

    def __init__(self):
        self._alphabet_model = AlphabetModel()
        self._diacritics_model = DiacriticsModel()

        self._previous_wrist_landmarks = []

    def predict(self, mp_results) -> Tuple[str, int]:
        '''Predict letter and diacritics based on MP results

        :param mp_results: results from MPH
        :return: letter and diacritic
        '''
        hand_l = []
        wrist_diff = []

        if mp_results.multi_hand_landmarks:
            hand_l, wrist_diff = self._preprocess_landmarks(mp_results)

        letter = self._alphabet_model.process(hand_l)
        diacritic = self._diacritics_model.process(wrist_diff, letter)
        return letter, diacritic

    def _preprocess_landmarks(self, results):
        hand_l = []
        wrist_l = [results.multi_hand_landmarks[0].landmark[self.IDX_WRIST].x,
                   results.multi_hand_landmarks[0].landmark[self.IDX_WRIST].y,
                   results.multi_hand_landmarks[0].landmark[self.IDX_WRIST].z]

        for landmark in results.multi_hand_landmarks[0].landmark:
            hand_l.append(landmark.x - wrist_l[0])
            hand_l.append(landmark.y - wrist_l[1])
            hand_l.append(landmark.z - wrist_l[2])

        if 'Right' in str(results.multi_handedness[0].classification):
            for coordinate in range(len(hand_l)):
                if coordinate % 3 == 0:
                    hand_l[coordinate] *= -1

        wrist_diff = self._preprocess_wrist(wrist_l)

        return hand_l, wrist_diff

    def _preprocess_wrist(self, wrist_l):
        if self._previous_wrist_landmarks:
            wrist_diff = [wrist_l[idx] - self._previous_wrist_landmarks[idx] for idx in range(2)]
        else:
            wrist_diff = [0.0, 0.0]

        self._previous_wrist_landmarks = wrist_l[0:-1]
        return wrist_diff
