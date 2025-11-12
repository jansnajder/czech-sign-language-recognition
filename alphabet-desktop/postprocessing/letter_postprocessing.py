from typing import Optional


class LetterPostprocessing:
    '''Object encapsulating the postprocessing of both the alphabet and diacritis model.'''

    DIACRITICS_MAP = {1: {'C': 'Č', 'D': 'Ď', 'E': 'Ě', 'N': 'Ň', 'R': 'Ř', 'S': 'Š', 'T': 'Ť', 'U': 'Ů',  'Z': 'Ž'},
                      2: {'A': 'Á', 'E': 'É', 'I': 'Í', 'O': 'Ó', 'U': 'Ú', 'Y': 'Ý'}}
    LETTERS = ['A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self):
        self.predicted_string = ''
        self._prev_letter = ' '

    def process(self, letter: str, diacritic: Optional[int]):
        '''Process the outputs of models. If process outputs a leter, append it to predicted_string

        :param letter: letter predicted by alphabet model
        :param diacritic: diacritic predicted by diacritics model
        '''
        if diacritic in self.DIACRITICS_MAP:
            if self._prev_letter in self.DIACRITICS_MAP[diacritic]:
                self.predicted_string = self.predicted_string[:-1] + self.DIACRITICS_MAP[diacritic][self._prev_letter]

        if letter in self.LETTERS:
            if letter != self._prev_letter:
                self.predicted_string += letter
                self._prev_letter = letter
