import evaluate

from typing import List
from nltk.translate.bleu_score import corpus_bleu


def calculate_rouge(refs: List[str], preds: List[str]) -> float:
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=preds, references=refs)
    return results["rougeL"]


def calculate_bleu(refs: List[str], preds: List[str]) -> List[float]:
    weights = [(1.,), (1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)]
    references = [[ref.split(' ')] for ref in refs]
    predictions = [pred.split(' ') for pred in preds]
    score = corpus_bleu(references, predictions, weights=weights)
    return score
