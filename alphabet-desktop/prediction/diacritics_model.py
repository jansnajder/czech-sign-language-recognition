import numpy as np
import tensorflow as tf


class DiacriticsModel:
    '''Encapsulates the tensorflow model for predicting sign alphabet diacritics.'''

    def __init__(self):
        self._model = tf.keras.models.load_model(r'models\diacritics-v1.0-32-8-D01-switch.model', compile=False)
        self._wrist_diff_list = []
        self._prev_letter = ''

    def process(self, wrist_diff, letter):
        '''Process the wrist difference by the model

        :param wrist_diff: wrist difference
        :param letter: predicted letter
        :return: predicted diacritic
        '''
        diacritic = 0

        if self._prev_letter and letter != self._prev_letter:
            if self._wrist_diff_list:
                wrist_diff_tf = tf.convert_to_tensor([self._wrist_diff_list])
                diacritic_raw = self._model(wrist_diff_tf)
                if np.max(diacritic_raw) > 0.7:
                    diacritic = np.argmax(diacritic_raw[0])

            if wrist_diff:
                self._wrist_diff_list = [wrist_diff]
            else:
                self._wrist_diff_list = []
        else:
            if wrist_diff:
                self._wrist_diff_list.append(wrist_diff)

        self._prev_letter = letter

        return diacritic
