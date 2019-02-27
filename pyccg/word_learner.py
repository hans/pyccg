import logging

import numpy as np

from pyccg import chart
from pyccg.lexicon import predict_zero_shot, \
    get_candidate_categories, get_semantic_arity, \
    augment_lexicon_distant, augment_lexicon_cross_situational, augment_lexicon_2afc, \
    build_bootstrap_likelihood
from pyccg.perceptron import \
    update_perceptron_distant, update_perceptron_cross_situational, update_perceptron_2afc
from pyccg.util import Distribution, NoParsesError


L = logging.getLogger(__name__)


class WordLearner(object):

  def __init__(self, lexicon, compressor, bootstrap=True,
               learning_rate=10.0, beta=3.0, negative_samples=5,
               total_negative_mass=0.1, syntax_prior_smooth=1e-3,
               meaning_prior_smooth=1e-3, bootstrap_alpha=0.25,
               prune_entries=3, zero_shot_limit=5,
               limit_induction=False):

    """
    Args:
      lexicon:
      compressor:
      bootstrap: If `True`, enable syntactic bootstrapping.
    """
    self.lexicon = lexicon
    self.compressor = compressor

    self.bootstrap = bootstrap

    # Learning hyperparameters
    self.learning_rate = learning_rate
    self.beta = beta
    self.negative_samples = negative_samples
    self.total_negative_mass = total_negative_mass
    self.syntax_prior_smooth = syntax_prior_smooth
    self.meaning_prior_smooth = meaning_prior_smooth
    self.bootstrap_alpha = bootstrap_alpha
    self.prune_entries = prune_entries
    self.zero_shot_limit = zero_shot_limit
    self.limit_induction = limit_induction

  @property
  def ontology(self):
    return self.lexicon.ontology

  def make_parser(self, ruleset=chart.DefaultRuleSet):
    """
    Construct a CCG parser from the current learner state.
    """
    return chart.WeightedCCGChartParser(self.lexicon, ruleset=ruleset)

  def compress_lexicon(self):
    if self.compressor is None:
      return

    try:
      # Run EC compression on the entries of the induced lexicon. This may create
      # new inventions, updating both the `ontology` and the provided `lex`.
      new_lex, affected_entries = self.compressor.make_inventions(self.lexicon)
    except Exception as e:
      L.error("Compression failed: %s", e)
      return

    # Create derived categories following new inventions.
    to_propagate = []
    for invention_name, tokens in affected_entries.items():
      if invention_name in new_lex._derived_categories_by_source:
        # TODO merge possibly new tokens with existing invention token groups
        continue

      affected_syntaxes = set(t.categ() for t in tokens)
      if len(affected_syntaxes) == 1:
        # Just one syntax is involved. Create a new derived category.
        L.debug("Creating new derived category for tokens %r", tokens)

        derived_name = new_lex.add_derived_category(tokens, source_name=invention_name)
        to_propagate.append((derived_name, next(iter(affected_syntaxes))))


    # Propagate derived categories, beginning with the largest functional
    # categories. The ordering allows us to support hard-propagating both e.g.
    # a new root category and a new argument category at the same time.
    #
    # (We may have derived new categories for entries with types `PP/NP` and # `S/NP/PP` -- in this case, we want to first make available a new category
    # `D0{S}/NP/PP` such that we can propagate the derived `D1{PP}` onto it,
    # yielding `D0{S}/NP/D1{PP}`.)
    to_propagate = sorted(
        to_propagate, key=lambda proposal: get_semantic_arity(proposal[1]),
        reverse=True)
    for derived_cat, base in to_propagate:
      new_lex.propagate_derived_category(derived_cat)
      cat_obj, _ = new_lex._derived_categories[derived_cat]
      L.info("Propagated derived category %s (source %s)", cat_obj, cat_obj.source_name)

    self.lexicon = new_lex

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

    L.info("No novel words; searching for new entries for known wordforms.")
    # Lexical entries are present for all words, but parse still failed.
    # That means we are missing entries for one or more wordforms.
    # For now: blindly try updating each word's entries.
    #
    # TODO: Does not handle case where multiple words need a joint update.
    query_tokens, query_token_syntaxes = list(set(sentence)), {}
    for token in sentence:
      query_token_syntaxes.update(
          get_candidate_categories(self.lexicon, [token], sentence,
                                   smooth=self.syntax_prior_smooth))

    # Sort query token list by increasing maximum weight of existing lexical
    # entry. This is a little hack to help the learner prefer to try to infer
    # new meanings for words it is currently more uncertain about.
    query_tokens = sorted(query_tokens,
        key=lambda tok: max(self.lexicon._entries[tok], key=lambda entry: entry.weight()).weight())
    return query_tokens, query_token_syntaxes

    raise ValueError(
        "unable to find new entries which will make the sentence parse: %s" % sentence)

  def do_lexical_induction(self, sentence, model, augment_lexicon_fn,
                           **augment_lexicon_args):
    """
    Perform necessary lexical induction such that `sentence` can be parsed
    under `model`.

    Returns:
      aug_lexicon: augmented lexicon, a modified copy of `self.lexicon`
    """
    if "queue_limit" not in augment_lexicon_args:
      augment_lexicon_args["queue_limit"] = self.zero_shot_limit

    # Find tokens for which we need to insert lexical entries.
    query_tokens, query_token_syntaxes = \
        self.prepare_lexical_induction(sentence)
    L.info("Inducing new lexical entries for words: %s", ", ".join(query_tokens))

    # Augment the lexicon with all entries for novel words which yield the
    # correct answer to the sentence under some parse. Restrict the search by
    # the supported syntaxes for the novel words (`query_token_syntaxes`).
    #
    # HACK: For now, only induce one word meaning at a time.
    lex = self.lexicon
    for query_token in query_tokens:
      try:
        lex = augment_lexicon_fn(
            lex, [query_token], query_token_syntaxes, sentence,
            self.ontology, model, self._build_likelihood_fns(sentence, model),
            beta=self.beta,
            negative_samples=self.negative_samples,
            total_negative_mass=self.total_negative_mass,
            **augment_lexicon_args)
      except NoParsesError:
        # No way to fix this token. That might be okay -- let's keep going.
        pass
      else:
        # Something got added.
        if self.limit_induction:
          break

    return lex

  def _build_likelihood_fns(self, sentence, model):
    ret = []
    if self.bootstrap:
      ret.append(build_bootstrap_likelihood(
        self.lexicon, sentence, self.ontology,
        alpha=self.bootstrap_alpha,
        meaning_prior_smooth=self.meaning_prior_smooth))

    return ret

  def predict_zero_shot_tokens(self, sentence, model):
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
        self.lexicon, query_tokens, query_token_syntaxes, sentence,
        self.ontology, model, self._build_likelihood_fns(sentence, model))
    return query_token_syntaxes, candidates

  def predict_zero_shot_2afc(self, sentence, model1, model2):
    """
    Yield zero-shot predictions on a 2AFC sentence, marginalizing over possible
    novel lexical entries required to parse the sentence.

    TODO explain marginalization process in more detail

    Args:
      sentence: List of token strings
      models:

    Returns:
      model_scores: `Distribution` over scene models (with support `models`),
        `p(referred scene | sentence)`
    """
    parser = chart.WeightedCCGChartParser(self.lexicon)
    weighted_results = parser.parse(sentence, True)
    if len(weighted_results) == 0:
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))

      aug_lexicon = self.do_lexical_induction(sentence, (model1, model2),
                                              augment_lexicon_fn=augment_lexicon_2afc,
                                              queue_limit=50)
      parser = chart.WeightedCCGChartParser(aug_lexicon)
      weighted_results = parser.parse(sentence, True)

    dist = Distribution()

    for result, score, _ in weighted_results:
      semantics = result.label()[0].semantics()
      try:
        model1_pass = model1.evaluate(semantics) == True
      except: pass
      else:
        if model1_pass:
          dist[model1] += np.exp(score)

      try:
        model2_pass = model2.evaluate(semantics) == True
      except: pass
      else:
        if model2_pass:
          dist[model2] += np.exp(score)

    return dist.ensure_support((model1, model2)).normalize()

  def _update_with_example(self, sentence, model,
                           augment_lexicon_fn, update_perceptron_fn,
                           augment_lexicon_args=None,
                           update_perceptron_args=None):
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

    augment_lexicon_args = augment_lexicon_args or {}
    update_perceptron_args = update_perceptron_args or {}

    try:
      weighted_results, _ = update_perceptron_fn(
          self.lexicon, sentence, model,
          learning_rate=self.learning_rate,
          **update_perceptron_args)
    except NoParsesError as e:
      # No parse succeeded -- attempt lexical induction.
      L.warning("Parse failed for sentence '%s'", " ".join(sentence))
      L.warning(e)

      self.lexicon = self.do_lexical_induction(sentence, model, augment_lexicon_fn,
                                               **augment_lexicon_args)
      self.lexicon.debug_print()

      # Compress the resulting lexicon.
      self.compress_lexicon()

      # Attempt a new parameter update.
      try:
        weighted_results, _ = update_perceptron_fn(
            self.lexicon, sentence, model,
            learning_rate=self.learning_rate,
            **update_perceptron_args)
      except NoParsesError:
        return []

    prune_count = self.lexicon.prune(max_entries=self.prune_entries)
    L.info("Pruned %i entries from lexicon.", prune_count)

    return weighted_results

  def update_with_distant(self, sentence, model, answer):
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
    kwargs = {"answer": answer}
    return self._update_with_example(
        sentence, model,
        augment_lexicon_fn=augment_lexicon_distant,
        update_perceptron_fn=update_perceptron_distant,
        augment_lexicon_args=kwargs,
        update_perceptron_args=kwargs)

  def update_with_cross_situational(self, sentence, model):
    """
    Observe a new `sentence` in the context of a scene reference `model`.
    Assume that `sentence` is true of `model`, and use it to update learner
    weights.

    Args:
      sentence: List of token strings
      model: `Model` instance

    Returns:
      weighted_results: List of weighted parse results for the example.
    """
    return self._update_with_example(
        sentence, model,
        augment_lexicon_fn=augment_lexicon_cross_situational,
        update_perceptron_fn=update_perceptron_cross_situational)

  def update_with_2afc(self, sentence, model1, model2):
    """
    Observe a new `sentence` in the context of two possible scene references
    `model1` and `model2`, where `sentence` is true of at least one of the
    scenes. Update learner weights.

    Args:
      sentence: List of token strings
      model1: `Model` instance
      model2: `Model` instance

    Returns:
      weighted_results: List of weighted results for the example, where each
        result is a pair `(model, parse_result)`. `parse_result` is the CCG
        syntax/semantics parse result, and `model` identifies the scene for
        which the semantic parse is true.
    """
    return self._update_with_example(
        sentence, (model1, model2),
        augment_lexicon_fn=augment_lexicon_2afc,
        update_perceptron_fn=update_perceptron_2afc)
