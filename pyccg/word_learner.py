import logging

from pyccg.lexicon import augment_lexicon_distant, predict_zero_shot, \
    get_candidate_categories, get_semantic_arity


L = logging.getLogger(__name__)


class WordLearner(object):

  def __init__(self, lexicon, bootstrap=True,
               learning_rate=10.0, beta=3.0, negative_samples=5,
               total_negative_mass=0.1, syntax_prior_smooth=1e-3,
               meaning_prior_smooth=1e-3, bootstrap_alpha=0.25,
               update_perceptron_algo='margin'):

    """
    Args:
      lexicon:
      compressor:
      bootstrap: If `True`, enable syntactic bootstrapping.
    """
    self.lexicon = lexicon

    self.bootstrap = bootstrap

    # Learning hyperparameters
    self.learning_rate = learning_rate
    self.beta = beta
    self.negative_samples = negative_samples
    self.total_negative_mass = total_negative_mass
    self.syntax_prior_smooth = syntax_prior_smooth
    self.meaning_prior_smooth = meaning_prior_smooth
    self.bootstrap_alpha = bootstrap_alpha

    if update_perceptron_algo == 'margin':
      from pyccg.perceptron import update_perceptron_distant
      self.update_perceptron_distant = update_perceptron_distant
    elif update_perceptron_algo == 'reinforce':
      from pyccg.perceptron import update_perceptron_distant_reinforce
      self.update_perceptron_distant = update_perceptron_distant_reinforce
    else:
      raise ValueError('Unknown update_perceptron algorithm: {}.'.format(update_perceptron_algo))

  @property
  def ontology(self):
    return self.lexicon.ontology

  def prepare_lexical_induction(self, sentence):
    """
    Find the tokens in a sentence which need to be updated such that the
    sentence will parse.

    Args:
      sentence: Sequence of tokens

    Returns:
      query_tokens: List of tokens which need to be updated
      query_token_syntaxes: Dict mapping tokens to weighted list of candidate
        syntaxes (as returned by `get_candidate_categoies`)
    """
    query_tokens = [word for word in sentence
                    if not self.lexicon._entries.get(word, [])]
    if len(query_tokens) > 0:
      # Missing lexical entries -- induce entries for words which clearly
      # require an entry inserted
      L.info("Novel words: %s", " ".join(query_tokens))
      query_token_syntaxes = get_candidate_categories(
          self.lexicon, query_tokens, sentence,
          smooth=self.syntax_prior_smooth)

      return query_tokens, query_token_syntaxes

    # Lexical entries are present for all words, but parse still failed.
    # That means we are missing entries for one or more wordforms.
    # For now: blindly try updating each word's entries.
    #
    # TODO: Does not handle case where multiple words need an update.
    query_tokens, query_token_syntaxes = [], []
    for token in sentence:
      query_tokens = [token]
      query_token_syntaxes = get_candidate_categories(
          self.lexicon, query_tokens, sentence,
          smooth=self.syntax_prior_smooth)

      if query_token_syntaxes:
        # Found candidate parses! Let's try adding entries for this token,
        # then.
        return query_tokens, query_token_syntaxes

    raise ValueError(
        "unable to find new entries which will make the sentence parse: %s" % sentence)

  def predict_zero_shot(self, sentence):
    """
    Yield zero-shot predictions on the syntax and meaning of words in the
    sentence requiring novel lexical entries.

    Args:
      sentence: List of token strings

    Returns:
      syntaxes: Dict mapping tokens to posterior distributions over syntactic
        categories
      joint_candidates: Dict mapping tokens to posterior distributions over
        tuples `(syntax, lf)`
    """
    # Find tokens for which we need to insert lexical entries.
    query_tokens, query_token_syntaxes = self.prepare_lexical_induction(sentence)
    candidates, _, _ = predict_zero_shot(
        self.lexicon, query_tokens, query_token_syntaxes,
        sentence, self.ontology,
        bootstrap=self.bootstrap,
        meaning_prior_smooth=self.meaning_prior_smooth,
        alpha=self.bootstrap_alpha)
    return query_token_syntaxes, candidates

  def update_with_example(self, sentence, model, answer, verbose=True):
    """
    Observe a new `sentence -> answer` pair in the context of some `model` and
    update learner weights.

    Args:
      sentence: List of token strings
      model: `Model` instance
      answer: Desired result from `model.evaluate(lf_result(sentence))`

    Returns:
      weighted_results: List of weighted parse results for the example.
    """

    try:
      weighted_results, _ = self.update_perceptron_distant(
          self.lexicon, sentence, model, answer,
          learning_rate=self.learning_rate)
    except ValueError as e:
      if verbose:
        # No parse succeeded -- attempt lexical induction.
        L.warning("Parse failed for sentence '%s'", " ".join(sentence))
        L.warning(e)

      # Find tokens for which we need to insert lexical entries.
      query_tokens, query_token_syntaxes = \
          self.prepare_lexical_induction(sentence)
      if verbose:
        L.info("Inducing new lexical entries for words: %s", ", ".join(query_tokens))

      # Augment the lexicon with all entries for novel words which yield the
      # correct answer to the sentence under some parse. Restrict the search by
      # the supported syntaxes for the novel words (`query_token_syntaxes`).
      self.lexicon = augment_lexicon_distant(
          self.lexicon, query_tokens, query_token_syntaxes, sentence,
          self.ontology, model, answer,
          bootstrap=self.bootstrap,
          meaning_prior_smooth=self.meaning_prior_smooth,
          alpha=self.bootstrap_alpha, beta=self.beta,
          negative_samples=self.negative_samples,
          total_negative_mass=self.total_negative_mass)

      if verbose:
        self.lexicon.debug_print()

      # Attempt a new parameter update.
      weighted_results, _ = self.update_perceptron_distant(
          self.lexicon, sentence, model, answer,
          learning_rate=self.learning_rate)

    prune_count = self.lexicon.prune()
    if verbose:
      L.info("Pruned %i entries from lexicon.", prune_count)

    return weighted_results
