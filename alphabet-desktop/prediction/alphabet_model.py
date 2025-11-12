import numpy as np
import tensorflow as tf


class AlphabetModel:
    '''Encapsulates the tensorflow model for predicting sign alphabet letters.'''

    LETTERS = ['A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    OUTPUTS_COUNT = 27  # number of classes
    MOVING_AVG_LENGTH = 10  # length of moving average window

    def __init__(self):
        self._model = tf.keras.models.load_model(r'models\xyz_060_b4096_d10_512-128.model')
        self._prediction_matrix = np.zeros(shape=(self.MOVING_AVG_LENGTH, self.OUTPUTS_COUNT))

    def process(self, hand_landmarks):
        '''Process the hand landmarks by the model

        :param hand_landmarks: landmarks to process
        '''
        letter = ''

        if hand_landmarks:
            raw_prediction = self._predict(hand_landmarks)
        else:
            raw_prediction = np.zeros(shape=(1, self.OUTPUTS_COUNT))

        self._prediction_matrix = np.insert(self._prediction_matrix[0:-1], 0, raw_prediction, axis=0)
        moving_avg = np.sum(self._prediction_matrix, axis=0) / self.MOVING_AVG_LENGTH

        if np.max(moving_avg) > 0.75:
            letter = self.LETTERS[int(np.argmax(moving_avg))]

        return letter

    def _predict(self, hand_landmarks):
        hand_landmarks_tf = tf.convert_to_tensor([hand_landmarks])
        raw_prediction = self._model(hand_landmarks_tf)
        return raw_prediction
