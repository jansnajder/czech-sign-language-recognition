from __future__ import annotations
import csv
import json

from data_handling.vocabulary import Vocabulary


def load_text_data(path: str) -> list[str]:
    words = []

    with open(path, 'r', encoding='utf-8') as fh:
        reader = csv.reader(fh)

        for line in reader:
            words.extend(line[1:])

    words = list(set(words))
    return words


if __name__ == "__main__":
    words = load_text_data("output_word_list.csv")
    vocab = Vocabulary.build([words])
    
    with open("vocab_full.json", "w", encoding="utf-8") as fh:
        json.dump(vocab.to_dict(), fh)

