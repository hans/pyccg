"""
Structured perceptron algorithm for learning CCG weights.
"""

from collections import Counter
import logging
import numpy as np

from pyccg import chart
from pyccg.util import softmax

L = logging.getLogger(__name__)


def update_perceptron_batch(lexicon, data, learning_rate=0.1, parser=None):
  """
  Execute a batch perceptron weight update with the given training data.

  Args:
    lexicon: CCGLexicon with weights
    data: List of `(x, y)` tuples, where `x` is a list of string
      tokens and `y` is an LF string.
    learning_rate:

  Returns:
    l2 norm of total weight updates
  """

  if parser is None:
    parser = chart.WeightedCCGChartParser(lexicon)

  norm = 0.0
  for x, y in data:
    weighted_results = parser.parse(x, return_aux=True)

    max_result, max_score, _ = weighted_results[0]
    correct_result, correct_score = None, None

    for result, score, _ in weighted_results:
      root_token, _ = result.label()
      if str(root_token.semantics()) == y:
        correct_result, correct_score = result, score
        break
    else:
      raise ValueError("no valid parse derived")

    if correct_score < max_score:
      for result, sign in zip([correct_result, max_result], [1, -1]):
        for _, leaf_token in result.pos():
          delta = sign * learning_rate
          norm += delta ** 2
          leaf_token._weight += delta

  return norm


def update_perceptron_distant(lexicon, sentence, model, answer,
                              learning_rate=10, parser=None):
  if parser is None:
    parser = chart.WeightedCCGChartParser(lexicon,
                                          ruleset=chart.DefaultRuleSet)

  norm = 0.0
  weighted_results = parser.parse(sentence, return_aux=True)
  if not weighted_results:
    raise ValueError("No successful parses computed.")

  max_score, max_incorrect_score = -np.inf, -np.inf
  correct_results, incorrect_results = [], []

  L.debug("Desired answer: %s", answer)
  for result, score, _ in weighted_results:
    root_token, _ = result.label()
    try:
      if model.evaluate(root_token.semantics()) == answer:
        if score > max_score:
          max_score = score
          correct_results = [(score, result)]
        elif score == max_score:
          correct_results.append((score, result))
      else:
        raise ValueError()
    except:
      if score > max_incorrect_score:
        max_incorrect_score = score
        incorrect_results = [(score, result)]
      elif score == max_incorrect_score:
        incorrect_results.append((score, result))
  else:
    if not correct_results:
      raise ValueError("No parses derived have the correct answer.")
    elif not incorrect_results:
      L.info("No incorrect parses. Skipping update.")
      return weighted_results, 0.0

  # Sort results by descending parse score.
  correct_results = sorted(correct_results, key=lambda r: r[0], reverse=True)
  incorrect_results = sorted(incorrect_results, key=lambda r: r[0], reverse=True)

  # TODO margin?

  # Update to separate max-scoring parse from max-scoring correct parse if
  # necessary.
  positive_mass = 1 / len(correct_results)
  negative_mass = 1 / len(incorrect_results)

  token_deltas = Counter()
  observed_leaf_sequences = set()
  for results, delta in zip([correct_results, incorrect_results],
                             [positive_mass, -negative_mass]):
    for _, result in results:
      leaf_seq = tuple(leaf_token for _, leaf_token in result.pos())
      if leaf_seq not in observed_leaf_sequences:
        observed_leaf_sequences.add(leaf_seq)
        for leaf_token in leaf_seq:
          token_deltas[leaf_token] += delta

  for token, delta in token_deltas.items():
    delta *= learning_rate
    norm += delta ** 2

    L.info("Applying delta: %+.03f %s", delta, token)
    token._weight += delta

  return weighted_results, norm


def update_perceptron_distant_v2(lexicon, sentence, model, answer,
                                 learning_rate=10, parser=None):
  if parser is None:
    parser = chart.WeightedCCGChartParser(lexicon,
                                          ruleset=chart.DefaultRuleSet)

  norm = 0.0
  weighted_results = parser.parse(sentence, return_aux=True)
  if not weighted_results:
    raise ValueError("No successful parses computed.")

  max_score, max_incorrect_score = -np.inf, -np.inf
  correct_results, incorrect_results = [], []

  L.debug("Desired answer: %s", answer)
  for result, score, _ in weighted_results:
    root_token, _ = result.label()
    try:
      if hasattr(model, 'evaluate_and_score'):
        correct, answer_score = model.evaluate_and_score(root_token.semantics(), answer)
        if correct:
          score += answer_score
      else:
        correct = model.evaluate(root_token.semantics()) == answer
      if correct:
        if score > max_score:
          max_score = score
          correct_results = [(score, result)]
        elif score == max_score:
          correct_results.append((score, result))
      else:
        raise ValueError()
    except:
      if score > max_incorrect_score:
        max_incorrect_score = score
        incorrect_results = [(score, result)]
      elif score == max_incorrect_score:
        incorrect_results.append((score, result))
  else:
    if not correct_results:
      raise ValueError("No parses derived have the correct answer.")
    elif not incorrect_results:
      L.info("No incorrect parses. Skipping update.")
      return weighted_results, 0.0

  # Sort results by descending parse score.
  correct_results = sorted(correct_results, key=lambda r: r[0], reverse=True)
  incorrect_results = sorted(incorrect_results, key=lambda r: r[0], reverse=True)

  # TODO margin?

  # Update to separate max-scoring parse from max-scoring correct parse if
  # necessary.
  positive_mass = 1 / len(correct_results)
  negative_mass = 1 / len(incorrect_results)

  token_deltas = Counter()
  observed_leaf_sequences = set()
  for results, delta in zip([correct_results, incorrect_results],
                             [1, -1]):
    parsing_results = [r[1] for r in results]
    parsing_scores = delta * softmax(np.array([r[0] for r in results]))

    for normed_delta, result in zip(parsing_scores, parsing_results):
      leaf_seq = tuple(leaf_token for _, leaf_token in result.pos())
      if leaf_seq not in observed_leaf_sequences:
        observed_leaf_sequences.add(leaf_seq)
        for leaf_token in leaf_seq:
          token_deltas[leaf_token] += normed_delta

  for token, delta in token_deltas.items():
    delta *= learning_rate
    norm += delta ** 2

    L.info("Applying delta: %+.03f %s", delta, token)
    token._weight += delta

  return weighted_results, norm
