from collections import defaultdict
from typing import Dict, List

import numpy as np



class Vocabulary:
    '''
    Object representing vocabulary. Initially the vocabulary is build from sequences of texts. It then can be stored
    into dict to dump to file (see from_dict, to_dict).

    The class itself handles mainly the encoding and decoding from text to integers. Two maps are used for that,
    stoi and itos. These maps also include special tokens.
    '''

    BOS_TOKEN = '<s>'  # beginning of sequence
    EOS_TOKEN = '</s>'  # end of sequence
    EOB_TOKEN = '</b>'  # end of sentence/block
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    DEFAULT_PAD_ID = 0
    DEFAULT_UNK_ID = lambda x: -1
    DEFAULT_UNK_TOKEN = lambda x: Vocabulary.UNK_TOKEN

    def __init__(self):
        self.stoi: Dict[str, int] = defaultdict(self.DEFAULT_UNK_ID)
        self.itos: Dict[int, str] = defaultdict(self.DEFAULT_UNK_TOKEN)
        self.stoi[self.PAD_TOKEN] = self.DEFAULT_PAD_ID
        self.itos[self.DEFAULT_PAD_ID] = self.PAD_TOKEN
        self.word_counter: Dict[str, int] = defaultdict(self.DEFAULT_UNK_ID)
        self.size = 1

    def to_dict(self):
        '''Dump the class to a dict, so it can be easily stored in file for later use'''
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "word_counter": self.word_counter,
            "size": self.size
        }

    @staticmethod
    def from_dict(val: Dict):
        '''Create Vocabulary object from previously saved dictionary'''
        vocab = Vocabulary()

        for k, v in val["stoi"].items():
            vocab.stoi[k] = v
        for k, v in val["itos"].items():
            vocab.itos[int(k)] = v
        for k, v in val["word_counter"].items():
            vocab.word_counter[k] = v

        vocab.size = val["size"]

        return vocab

    @staticmethod
    def build(texts: List[List[str]]):
        '''Build Vocabulary from sequences of strings'''
        vocab = Vocabulary()

        for text in texts:
            vocab.add_tokens(text)

        return vocab

    def add_tokens(self, tokens: List[str]) -> None:
        '''Add list of tokens to the vocabulary. It also keeps track of how many of given token is already in vocab.'''
        for token in tokens:
            index = len(self.itos)

            if token not in self.stoi:
                self.itos[index] = token
                self.stoi[token] = index
                self.word_counter[token] = 1
                self.size += 1
            else:
                self.word_counter[token] += 1

    def decode(self, array: np.array, eos_cut: bool = True) -> List[str]:
        '''Decode integer array to text.'''
        sentence = []

        for idx in array:
            token = self.itos[idx]

            if token == self.PAD_TOKEN:
                continue

            sentence.append(token)

            if eos_cut and token == self.EOS_TOKEN:
                break

        return sentence

    def encode(self, sentence: List[str]) -> np.array:
        '''Encode text to array of integers'''
        return np.array([self.stoi[token] for token in sentence])
