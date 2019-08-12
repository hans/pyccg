"""
Tools for updating and expanding lexicons, dealing with logical forms, etc.
"""

from collections import defaultdict, Counter
from copy import copy, deepcopy
from functools import reduce
import itertools
import logging
import queue
import random
import re
import sys

from nltk import Tree
from nltk.ccg import lexicon as ccg_lexicon
from nltk.ccg.api import PrimitiveCategory, FunctionalCategory, AbstractCCGCategory, Direction
import numpy as np
from tqdm import tqdm, trange

from pyccg import chart, Token
from pyccg.combinator import category_search_replace
from pyccg import logic as l
from pyccg.util import ConditionalDistribution, Distribution, UniquePriorityQueue, \
    NoParsesError, tuple_unordered


L = logging.getLogger(__name__)


#------------
# Regular expressions used for parsing components of the lexicon
#------------

# Parses a primitive category and subscripts
PRIM_RE = re.compile(r'''([A-Za-z]+)(\[[A-Za-z,]+\])?''')

# Separates the next primitive category from the remainder of the
# string
NEXTPRIM_RE = re.compile(r'''([A-Za-z]+(?:\[[A-Za-z,]+\])?)(.*)''')

# Separates the next application operator from the remainder
APP_RE = re.compile(r'''([\\/])([.,]?)([.,]?)(.*)''')

# Parses the definition of the right-hand side (rhs) of either a word or a family
LEX_RE = re.compile(r'''([\S_]+)\s*(::|[-=]+>)\s*(.+)''', re.UNICODE)

# Parses the right hand side that contains category and maybe semantic predicate
RHS_RE = re.compile(r'''([^{}<>]*[^ {}<>])\s*(\{[^}]+\})?\s*(<-?\d*(?:\.\d+)?>)?''', re.UNICODE)

# Parses the semantic predicate
SEMANTICS_RE = re.compile(r'''\{([^}]+)\}''', re.UNICODE)

# Strips comments from a line
COMMENTS_RE = re.compile('''([^#]*)(?:#.*)?''')


class Lexicon(ccg_lexicon.CCGLexicon):

  def __init__(self, starts, primitives, families, entries, ontology=None):
    """
    Create a new Lexicon.

    Args:
      start: Start symbol(s). All valid parses must have a root node of this
        category. Either a string (single start) or a sequence (multiple
        allowed starts).
      primitives:
      families:
      entries: Lexical entries. Dict mapping from word strings to lists of
        `Token`s.
    """
    starts = [starts] if isinstance(starts, str) else starts
    self._starts = [ccg_lexicon.PrimitiveCategory(start) for start in starts]
    self._primitives = primitives
    self._families = families
    self._entries = entries

    self.ontology = ontology

    self._derived_categories = {}
    self._derived_categories_by_base = defaultdict(set)
    self._derived_categories_by_source = {}

  @classmethod
  def fromstring(cls, lex_str, ontology=None, include_semantics=False,
                 default_weight=0.001):
    """
    Convert string representation into a lexicon for CCGs.
    """
    ccg_lexicon.CCGVar.reset_id()
    primitives, starts = [], []
    families = {}
    entries = defaultdict(list)
    for line in lex_str.splitlines():
      # Strip comments and leading/trailing whitespace.
      line = COMMENTS_RE.match(line).groups()[0].strip()
      if line == "":
        continue

      if line.startswith(':-'):
        # A line of primitive categories.
        # The first one is the target category
        # ie, :- S, N, NP, VP
        primitives = primitives + [prim.strip() for prim in line[2:].strip().split(',')]

        # But allow multiple target categories separated by a colon in the first element:
        # ie, :- S:N,NP,VP
        starts = primitives[0].split(":")
        primitives = starts + primitives[1:]
      else:
        # Either a family definition, or a word definition
        (ident, sep, rhs) = LEX_RE.match(line).groups()
        (catstr, semantics_str, weight) = RHS_RE.match(rhs).groups()
        (cat, var) = ccg_lexicon.augParseCategory(catstr, primitives, families)

        if sep == '::':
          # Family definition
          # ie, Det :: NP/N
          families[ident] = (cat, var)
          # TODO weight?
        else:
          semantics = None
          if include_semantics is True:
            if semantics_str is None:
              raise AssertionError(line + " must contain semantics because include_semantics is set to True")
            else:
              semantics = l.Expression.fromstring(ccg_lexicon.SEMANTICS_RE.match(semantics_str).groups()[0])

              # Assign types.
              if ontology is not None:
                ontology.typecheck(semantics)

          weight = float(weight[1:-1]) if weight is not None else default_weight

          # Word definition
          # ie, which => (N\N)/(S/NP)
          entries[ident].append(Token(ident, cat, semantics, weight=weight))
    return cls(starts, primitives, families, entries,
               ontology=ontology)

  def get_entries(self, word):
    return self._entries.get(word, [])

  def set_entries(self, word, entries):
    """
    Set the list of entries for a wordform `word`.

    Arguments:
      word: String wordform
      entries: List of `(category, semantics, weight)` tuples
    """
    self._entries[word] = []
    for category, semantics, weight in entries:
      self.add_entry(word, category, semantics=semantics, weight=weight)

  def add_entry(self, word, category, semantics=None, weight=None):
    """
    Add a `Token` entry for the wordform `word`.

    Arguments:
      word: String wordform
      category: Syntactic category
      semantics: Meaning representation
      weight: Float weight
    """
    # Typecheck and assign types in semantic representation.
    if semantics is not None and self.ontology is not None:
      self.ontology.typecheck(semantics)

    token = Token(word, category, semantics=semantics, weight=weight)
    self._entries[word].append(token)

  def get_entries_with_category(self, category):
    """
    Returns a generator yielding `Token` instances with the given syntactic
    category.
    """
    for entries in self._entries.values():
      for entry in entries:
        if entry.categ() == category:
          yield entry

  def get_entries_with_semantics(self, semantics):
    """
    Returns a generator yielding `Token` instances with the given semantic
    expression.
    """
    for entries in self._entries.values():
      for entry in entries:
        if entry.semantics() == semantics:
          yield entry

  def __eq__(self, other):
    return isinstance(other, Lexicon) and self._starts == other._starts \
        and self._primitives == other._primitives and self._families == other._families \
        and self._entries == other._entries \
        and self._derived_categories == other._derived_categories

  def clone(self, retain_semantics=True):
    """
    Return a clone of the current lexicon instance.
    """
    ret = deepcopy(self)

    if not retain_semantics:
      for entry_tokens in ret._entries.values():
        for token in entry_tokens:
          token._semantics = None

    return ret

  def prune(self, max_entries=3):
    """
    Prune low-weight entries from the lexicon in-place.

    Args:
      min_weight: Minimal weight for entries which should be retained.

    Returns:
      prune_count: Number of lexical entries which were pruned.
    """
    prune_count = 0
    for token in self._entries:
      entries_t = [token for token in self._entries[token] if token.weight() > 0]
      entries_t = sorted(entries_t, key=lambda t: t.weight())[:max_entries]
      prune_count += len(self._entries[token]) - len(entries_t)
      self._entries[token] = entries_t

    return prune_count

  def debug_print(self, stream=sys.stdout):
    for token, entries in self._entries.items():
      for entry in entries:
        stream.write("%.3f %s\n" % (entry.weight(), entry))

  def parse_category(self, cat_str):
    return ccg_lexicon.augParseCategory(cat_str, self._primitives, self._families)[0]

  @property
  def primitive_categories(self):
    return set([self.parse_category(prim) for prim in self._primitives])

  @property
  def observed_categories(self):
    """
    Find categories (both primitive and functional) attested in the lexicon.
    """
    return set([token.categ()
                for token_list in self._entries.values()
                for token in token_list])

  def total_category_masses(self, exclude_tokens=frozenset(),
                            soft_propagate_roots=False):
    """
    Return the total weight mass assigned to each syntactic category. Shifts
    masses such that the minimum mass is zero.

    Args:
      exclude_tokens: Exclude entries with this token from the count.
      soft_propagate_roots: Soft-propagate derived root categories. If there is
        a derived root category `D0{S}` and some lexical entry `S/N`, even if
        no entry has the category `D0{S}`, we will add a key to the returned
        counter with category `D0{S}/N` (and zero weight).

    Returns:
      masses: `Distribution` mapping from category types to masses. The minimum
        mass value is zero and the maximum is unbounded.
    """
    ret = Distribution()
    # Track categories with root yield.
    rooted_cats = set()

    for token, entries in self._entries.items():
      if token in exclude_tokens:
        continue
      for entry in entries:
        c_yield = get_yield(entry.categ())
        if c_yield in self._starts:
          rooted_cats.add((c_yield, entry.categ()))

        if entry.weight() > 0.0:
          ret[entry.categ()] += entry.weight()

    if soft_propagate_roots:
      for c_yield, rooted_cat in rooted_cats:
        for derived_root_cat in self._derived_categories_by_base[c_yield]:
          soft_prop_cat = set_yield(rooted_cat, derived_root_cat)
          # Ensure key exists.
          ret.setdefault(soft_prop_cat, 0.0)

    return ret

  def observed_category_distribution(self, exclude_tokens=frozenset(),
                                     soft_propagate_roots=False):
    """
    Return a distribution over categories calculated using the lexicon weights.
    """
    ret = self.total_category_masses(exclude_tokens=exclude_tokens,
                                     soft_propagate_roots=soft_propagate_roots)
    return ret.normalize()

  @property
  def start_categories(self):
    """
    Return primitive categories which are valid root nodes.
    """
    return self._starts + \
        list(itertools.chain.from_iterable(self._derived_categories_by_base[start]
                                           for start in self._starts))

  def start(self):
    raise NotImplementedError("use #start_categories instead.")

  def category_semantic_arities(self, soft_propagate_roots=False):
    """
    Get the arities of semantic expressions associated with each observed
    syntactic category.
    """
    # If possible, lean on the type system to help determine expression arity.
    get_arity = (self.ontology and self.ontology.get_expr_arity) or l.get_arity

    entries_by_categ = {
      category: set(entry for entry in itertools.chain.from_iterable(self._entries.values())
                    if entry.categ() == category)
      for category in self.observed_categories
    }

    ret = {}
    rooted_cats = set()
    for category, entries in entries_by_categ.items():
      if get_yield(category) in self._starts:
        rooted_cats.add((get_yield(category), category))
      ret[category] = set(get_arity(entry.semantics()) for entry in entries)

    if soft_propagate_roots:
      for c_yield, rooted_cat in rooted_cats:
        for derived_root_cat in self._derived_categories_by_base[c_yield]:
          new_syn = set_yield(rooted_cat, derived_root_cat)
          ret.setdefault(new_syn, ret[rooted_cat])

    return ret

  def add_derived_category(self, involved_tokens, source_name=None):
    name = "D%i" % len(self._derived_categories)

    # The derived category will have as its base the yield of the source
    # category. For example, tokens of type `PP/NP` will lead to a derived
    # category of type `PP`.
    categ = DerivedCategory(name, get_yield(involved_tokens[0].categ()),
                            source_name=source_name)
    self._primitives.append(categ)
    self._derived_categories[name] = (categ, set(involved_tokens))
    self._derived_categories_by_base[categ.base].add(categ)

    if source_name is not None:
      self._derived_categories_by_source[source_name] = categ

    return name

  def propagate_derived_category(self, name):
    categ, involved_entries = self._derived_categories[name]
    originating_category = next(iter(involved_entries)).categ()
    new_entries = defaultdict(list)

    # Replace all lexical entries directly involved with the derived category.
    for word, entry_list in self._entries.items():
      for entry in entry_list:
        if entry in involved_entries:
          # Replace the yield of the syntactic category with our new derived
          # category. (For primitive categories, setting the yield is
          # equivalent to just changing the category.)
          new_entry = entry.clone()
          new_entry._categ = set_yield(entry.categ(), categ)
          new_entries[word].append(new_entry)

    # Create duplicates of all entries with functional categories involving the
    # base of the derived category.
    #
    # For example, if we have an entry of syntactic category `S/N/PP` and we
    # have just created a derived category `D0` based on `N`, we need to make
    # sure there is now a corresponding candidate entry of type `S/D0/PP`.

    replacements = {}
    # HACK: don't propagate S
    if categ.base.categ() != "S":
      for word, entries in self._entries.items():
        for entry in entries:
          if not isinstance(entry.categ(), FunctionalCategory):
            # This step only applies to functional categories.
            continue
          elif entry.categ() == originating_category:
            # We've found an entry which has a category with the same category
            # as that of the tokens involved in this derived category's
            # creation. Don't propagate -- this is exactly what allows us to
            # separate the involved tokens from other members of the same
            # category.
            #
            # e.g. if we've just derived a category from some PP/NP entries,
            # don't propagate the PP yield onto other PP/NP entries which were
            # not involved in the derived category creation.
            continue

          try:
            categ_replacements = replacements[entry.categ()]
          except KeyError:
            replacements[entry.categ()] = category_search_replace(
                entry.categ(), categ.base, categ)

            categ_replacements = replacements[entry.categ()]

          for replacement_category in categ_replacements:
            # We already know a replacement is necessary -- go ahead.
            new_entry = entry.clone()
            new_entry._categ = replacement_category
            new_entries[word].append(new_entry)

    for word, w_entries in new_entries.items():
      self._entries[word].extend(w_entries)


  def lf_ngrams(self, order=1, conditioning_fn=None, smooth=None):
    """
    Calculate n-gram statistics about the predicates present in the semantic
    forms in the lexicon.

    Args:
      order: n-gram order
      conditioning_fn: If non-`None`, returns conditional distributions mapping
        from the range of `conditioning_fn` to distributions over semantic
        predicates. This can be used to e.g. build distributions over
        predicates conditioned on syntactic category.
      smooth: If not `None`, add-k smooth the returned distributions using the
        provided float.
    """
    if order > 1:
      raise NotImplementedError()

    ret = ConditionalDistribution()
    for entry_list in self._entries.values():
      for entry in entry_list:
        keys = conditioning_fn(entry) if conditioning_fn is not None else [None]
        for key in keys:
          # Initialize the distribution, whether or not we will find any
          # predicates to count.
          ret.ensure_cond_support(key)

          for predicate in entry.semantics().predicates():
            ret[key][predicate.name] += entry.weight()

    if smooth is not None:
      support = ret.support
      for key in ret:
        for predicate in support:
          ret[key][predicate] += smooth
        ret[key][None] += smooth

    ret.normalize_all()

    if conditioning_fn is None:
      return ret[None]
    return ret

  def lf_ngrams_given_syntax(self, **kwargs):
    conditioning_fn = lambda entry: [entry.categ()]
    kwargs["conditioning_fn"] = conditioning_fn
    return self.lf_ngrams(**kwargs)

  def lf_ngrams_mixed(self, alpha=0.25, **kwargs):
    """
    Return conditional distributions over logical form n-grams conditioned on
    syntactic category, calculated by mixing two distribution classes: a
    distribution conditioned on the full syntactic category and a distribution
    conditioned on the yield of the category.
    """
    lf_syntax_ngrams = self.lf_ngrams_given_syntax(**kwargs)
    lf_support = lf_syntax_ngrams.support

    # Soft-propagate derived root categories.
    for syntax in list(lf_syntax_ngrams.dists.keys()):
      syn_yield = get_yield(syntax)
      if syn_yield in self._starts:
        for derived_root_cat in self._derived_categories_by_base[syn_yield]:
          new_yield = set_yield(syntax, derived_root_cat)
          if new_yield not in lf_syntax_ngrams:
            lf_syntax_ngrams[new_yield] = Distribution.uniform(lf_support)

    # Second distribution: P(pred | root)
    lf_yield_ngrams = self.lf_ngrams(
        conditioning_fn=lambda entry: [get_yield(entry.categ())], **kwargs)
    # Mix full-category and primitive-category predictions.
    lf_mixed_ngrams = ConditionalDistribution()
    for syntax in lf_syntax_ngrams:
      # # Mix distributions conditioned on the constituent primitives.
      # primitives = get_category_primitives(syntax)
      # prim_alpha = 1 / len(primitives)

      # Mix root-conditioned distribution and the full syntax-conditioned
      # distribution.
      yield_dist = lf_yield_ngrams[get_yield(syntax)]
      lf_mixed_ngrams[syntax] = lf_syntax_ngrams[syntax].mix(yield_dist, alpha)

    return lf_mixed_ngrams

  def sample_sentence(self, arguments, relation=None, return_dist=False):
    """
    Forward-sample a sentence given a set of argument LFs, according to the
    generative model.
    """
    # TODO this sort of code probably belongs in a separate "model" somewhere.
    # For now we'll conflate the lexicon and the probabilistic model ...

    ret = Distribution()
    ret_trees = []

    # necessary to impose arbitrary reference order on the `arguments`
    # collection -- we'll iterate over all orderings eventually
    arguments = tuple(arguments)

    if relation is None:
      # First sample a relation, assuming that relations are distributed
      # independently of their arguments.
      relations = Counter()
      for entries in self._entries.values():
        for entry in entries:
          if self.ontology.get_expr_arity(entry.semantics()) == len(arguments):
            # This is a candidate relation. Store without binding if present.
            to_store = entry.semantics()
            if isinstance(to_store, l.VariableBinderExpression):
              _, to_store = to_store.decompose()

            relations[to_store] += entry.weight()

      relations = Distribution(relations).normalize()

    # Lexicalize arguments.
    arg_entries = []
    for arg_i in arguments:
      # TODO weighted sampling
      arg_i_entries = list(self.get_entries_with_semantics(arg_i))
      if not arg_i_entries:
        raise ValueError("no entries found with semantics %r" % arg_i)

      arg_i_entries = Distribution.uniform(arg_i_entries)
      arg_entries.append(arg_i_entries)

    for relation in relations:
      to_bind = [var for var in relation.free()
                if var.name not in self.ontology.functions_dict
                and var.name not in self.ontology.constants_dict]
      # keep a mapping of integer index to actual variable, for later
      idx_to_arg = {int(arg.name[1:]): arg for arg in to_bind}

      # Get the semantic depths of each bound variable in the relation.
      semantic_depths = {arg: l.get_depths(relation, arg) for arg in to_bind}
      # Get max semantic depth for each argument.
      semantic_depths = {arg: max(depths.keys())
                        for arg, depths in semantic_depths.items()}
      # Reindex by variable number, assuming the expression is normalized (??)
      semantic_depths = {int(arg.name[1:]): depth
                        for arg, depth in semantic_depths.items()}

      # Retain syntactic depth mappings which satisfy prominence preservation.
      prior_arg_orders = list(itertools.permutations(list(range(1, len(arguments) + 1))))
      arg_orders = []
      for arg_order in prior_arg_orders:
        ok = True
        for arg1, arg2 in zip(arg_order, arg_order[1:]):
          # arg1 > arg2 in syntactic depth, so require that arg1 >= arg2 in
          # semantic depth
          if not semantic_depths[arg1] >= semantic_depths[arg2]:
            ok = False

        if ok:
          arg_orders.append(arg_order)

      # TODO weights here .. ?
      arg_orders = Distribution.uniform(arg_orders)
      for arg_order in arg_orders:
        # Create a binding expression for this argument order.
        expr = relation
        for arg_idx in arg_order[::-1]:
          expr = l.LambdaExpression(idx_to_arg[arg_idx], expr)

        for arg_entry_comb in itertools.product(*arg_entries):
          # Search for possible syntactic categories of the lexicalized relation
          # given the arguments.
          #
          # TODO this should be computed programmatically from arg_order +
          # arg_entry_comb
          if len(arguments) == 0:
            search_category = "S"
          elif len(arguments) == 1:
            search_category = r"S\N"
          elif len(arguments) == 2:
            search_category = r"S\N/N"
          else:
            raise ValueError("cannot handle more than 2-ary arguments at the moment.")
          search_category = self.parse_category(search_category)

          custom_entry = Token("XXX", search_category, expr)

          # Now compose a tree bottom-up, ordering the arguments according to
          # `arg_order`.
          tree = self._build_sentence_tree(
              custom_entry,
              [arg_entry_comb[idx - 1] for idx in arg_order])

          # collect likelihood elements
          component_ps = [relations[relation],
                          arg_orders[arg_order]]
          component_ps += [arg_i_entries[arg_entry]
                            for arg_i_entries, arg_entry
                            in zip(arg_entries, arg_entry_comb)]
          logp = sum(np.log(p) for p in component_ps)

          ret[len(ret_trees)] = logp
          ret_trees.append(tree)

    ret = ret.normalize()
    if return_dist:
      return ret, ret_trees

    sample = ret.sample()
    return ret_trees[sample]

  def _build_sentence_tree(self, verb_entry, argument_entries):
    """
    Compose lexical entries for a verb and its arguments into a full sentence.

    Args:
      verb_entry: A lexical entry for the root verb.
      argument_entries: lexical entries for each of the verb's arguments, in
        order of application with the verb entry

    Returns:
      tree: a CCG parse tree
    """
    def _make_leaf_node(entry):
      return Tree((entry, "Leaf"), [Tree(entry, [entry._token])])
    # compute leave reprs, sorted by desired processing order
    leaves = list(map(_make_leaf_node, [verb_entry] + list(argument_entries)))

    tree = leaves[0]
    for arg_leaf in leaves[1:]:
      tree_node = tree.label()[0]
      arg_node = arg_leaf.label()[0]

      tree_tokens = tree_node._token
      if isinstance(tree_tokens, str):
        tree_tokens = [tree_tokens]

      # prepare constituent metadata
      if tree_node.categ().dir().is_forward():
        op = ">"
        children = [tree, arg_leaf]
        tokens = tree_tokens + [arg_node._token]
      else:
        op = "<"
        children = [arg_leaf, tree]
        tokens = [arg_node._token] + tree_tokens

      node_syntax = tree_node.categ().res()

      # Node semantics: simple function application.
      node_semantics = l.ApplicationExpression(
          tree_node.semantics(), arg_node.semantics()).simplify()

      lhs = (Token(tokens, node_syntax, node_semantics), op)
      tree = Tree(lhs, children)

    return tree


class DerivedCategory(PrimitiveCategory):

  def __init__(self, name, base, source_name=None):
    self.name = name
    self.base = base
    self.source_name = source_name
    self._comparison_key = (name, base)

  def is_primitive(self):
    return True

  def is_function(self):
    return False

  def is_var(self):
    return False

  def categ(self):
    return self.name

  def substitute(self, subs):
    return self

  def can_unify(self, other):
    # The unification logic is critical here -- this determines how derived
    # categories are treated relative to their base categories.
    if other == self:
      return []

  def __str__(self):
    return "%s{%s}" % (self.name, self.base)

  def __repr__(self):
    return "%s{%s}{%s}" % (self.name, self.base, self.source_name)


def get_semantic_arity(category, arity_overrides=None):
  """
  Get the expected arity of a semantic form corresponding to some syntactic
  category.
  """
  arity_overrides = arity_overrides or {}
  if category in arity_overrides:
    return arity_overrides[category]

  if isinstance(category, PrimitiveCategory):
    return 0
  elif isinstance(category, FunctionalCategory):
    return 1 + get_semantic_arity(category.arg(), arity_overrides) \
      + get_semantic_arity(category.res(), arity_overrides)
  else:
    raise ValueError("unknown category type %r" % category)


def get_category_primitives(category):
  """
  Get the primitives involved in the given syntactic category.
  """
  if isinstance(category, PrimitiveCategory):
    return [category]
  elif isinstance(category, FunctionalCategory):
    return get_category_primitives(category.arg()) + \
        get_category_primitives(category.res())
  else:
    raise ValueError("unknown category type %r" % category)


def get_category_arity(category):
  if isinstance(category, PrimitiveCategory):
    return 0
  elif isinstance(category, FunctionalCategory):
    return 1 + get_category_arity(category.res())
  else:
    raise ValueError("unknown category type %r" % category)


def get_yield(category):
  """
  Get the primitive yield node of a syntactic category.
  """
  if isinstance(category, PrimitiveCategory):
    return category
  elif isinstance(category, FunctionalCategory):
    return get_yield(category.res())
  else:
    raise ValueError("unknown category type with instance %r" % category)


def set_yield(category, new_yield):
  if isinstance(category, PrimitiveCategory):
    return new_yield
  elif isinstance(category, FunctionalCategory):
    return FunctionalCategory(set_yield(category.res(), new_yield),
                              category.arg(), category.dir())
  else:
    raise ValueError("unknown category type of instance %r" % category)


def get_syntactic_parses(lex, tokens, sentence, smooth=1e-3):
  """
  Find the weighted syntactic parses for `sentence`, inducing novel
  syntactic entries for each of `tokens`.
  """
  assert set(tokens).issubset(set(sentence))

  # Make a minimal copy of `lex` which does not track semantics.
  lex = lex.clone(retain_semantics=False)

  # Remove entries for the queried tokens.
  for token in tokens:
    lex.set_entries(token, [])

  category_prior = lex.observed_category_distribution(
      exclude_tokens=set(tokens), soft_propagate_roots=True)
  if smooth is not None:
    for key in category_prior.keys():
      category_prior[key] += smooth
    category_prior = category_prior.normalize()
  L.debug("Smoothed category prior with soft root propagation: %s", category_prior)

  def get_parses(cat_assignment):
    for token, category in zip(tokens, cat_assignment):
      lex.set_entries(token, [(category, None, category_prior[category])])

    return chart.WeightedCCGChartParser(lex, chart.DefaultRuleSet) \
        .parse(sentence, return_aux=True)

  for cat_assignment in itertools.product(category_prior.keys(), repeat=len(tokens)):
    for parse, weight, _ in get_parses(cat_assignment):
      yield (weight, cat_assignment, parse)


def get_candidate_categories(lex, tokens, sentence, smooth=1e-3):
  """
  Find candidate categories for the given tokens which appear in `sentence` such
  that `sentence` yields a parse.

  Args:
    lex:
    tokens:
    sentence:
    smooth: If not `None`, add-k smooth the prior distribution over syntactic
      categories (where the float value of `smooth` specifies `k`).

  Returns:
    cat_dists: Dictionary mapping each token to a `Distribution` over
      categories.
  """
  assert set(tokens).issubset(set(sentence))

  # Make a minimal copy of `lex` which does not track semantics.
  lex = lex.clone(retain_semantics=False)

  # Remove entries for the queried tokens.
  for token in tokens:
    lex.set_entries(token, [])

  category_prior = lex.observed_category_distribution(
      exclude_tokens=set(tokens), soft_propagate_roots=True)
  if smooth is not None:
    for key in category_prior.keys():
      category_prior[key] += smooth
    category_prior = category_prior.normalize()
  L.debug("Smoothed category prior with soft root propagation: %s", category_prior)

  def score_cat_assignment(cat_assignment):
    """
    Assign a log-probability to a joint assignment of categories to tokens.
    """

    for token, category in zip(tokens, cat_assignment):
      lex.set_entries(token, [(category, None, 0.001)])

    # Attempt a parse.
    results = chart.WeightedCCGChartParser(lex, chart.DefaultRuleSet) \
        .parse(sentence, return_aux=True)
    if results:
      # Prior weight for category comes from lexicon.
      #
      # Might also downweight categories which require type-lifting parses by
      # default?
      score = 0
      for token, category in zip(tokens, cat_assignment):
        score += np.log(category_prior[category])

      # Likelihood weight comes from parse score
      score += np.log(sum(np.exp(weight)
                          for _, weight, _ in results))

      return score

    return -np.inf

  # NB does not cover the case where a single token needs multiple syntactic
  # interpretations for the sentence to parse
  cat_assignment_weights = {
    cat_assignment: score_cat_assignment(cat_assignment)
    for cat_assignment in itertools.product(category_prior.keys(), repeat=len(tokens))
  }

  cat_dists = defaultdict(Distribution)
  for cat_assignment, logp in cat_assignment_weights.items():
    for token, token_cat_assignment in zip(tokens, cat_assignment):
      cat_dists[token][token_cat_assignment] += np.exp(logp)

  # Normalize.
  cat_dists = {token: dist.normalize() for token, dist in cat_dists.items()}
  return cat_dists


def attempt_candidate_parse(lexicon, tokens, candidate_categories,
                            sentence, dummy_vars):
  """
  Attempt to parse a sentence, mapping `tokens` to new candidate
  lexical entries.

  Arguments:
    lexicon: Current lexicon. Will modify in place -- send a copy.
    tokens: List of string token(s) to be attempted.
    candidate_categories: List of candidate categories for each token (one per
      token).
    sentence: Sentence which we are attempting to parse.

  Returns:
    results: A list of tuples describing candidate parses. The first element of
      each tuple is a full parse tree of the sentence, including sentence
      semantics, where dummy function variables are assigned for each of the
      queried `tokens`. The second element of each tuple maps each token of
      `tokens` to a list of apparent type usages within the parse. These type
      observations can be used to later restrict the search for possible
      semantic expressions for each of the tokens.
  """

  # Restrict semantic arities based on the observed arities available for each
  # category. Pre-calculate the necessary associations.
  category_sem_arities = lexicon.category_semantic_arities()

  lexicon = lexicon.clone()

  # Build lexicon entries with dummy semantic expressions for each of the
  # query tokens.
  for token, syntax in zip(tokens, candidate_categories):
    var = copy(dummy_vars[token])
    expr = l.IndividualVariableExpression(var)

    if lexicon.ontology is not None:
      # assign semantic arity based on syntactic arity
      try:
        arities = category_sem_arities[syntax]
      except KeyError:
        arities = [get_semantic_arity(syntax)]

      exprs = []
      for arity in arities:
        var = copy(dummy_vars[token])
        var.type = lexicon.ontology.types[("*",) * arity + ("?",)]
        expr = l.IndividualVariableExpression(var)
        exprs.append(expr)

      lexicon.set_entries(token, [(syntax, expr, 1.0) for expr in exprs])
    else:
      lexicon.set_entries(token, [(syntax, expr, 1.0)])

  parse_results = []

  # First attempt a parse with only function application rules.
  results = chart.WeightedCCGChartParser(lexicon, ruleset=chart.ApplicationRuleSet) \
      .parse(sentence, return_aux=True)

  for result, weight, _ in results:
    apparent_types = None

    if lexicon.ontology is not None:
      # Retrieve apparent types of each token's dummy var.
      sem = result.label()[0].semantics()

      # TODO order of type inference really matters here -- there can be
      # dependencies.
      #
      # e.g. for an expression `F000(F001)` where F001 is of type `?`, we need
      # to infer `F000 :: ? -> e` before `F001`, otherwise we'll overwrite
      # `F001` type
      apparent_types = {}
      for token, var in dummy_vars.items():
        # Build up type signature from previously inferred types.
        extra_types = {dummy_vars[token].name: apparent_type
                       for token, apparent_type in apparent_types.items()}
        apparent_types[token] = lexicon.ontology.infer_type(sem, var.name,
                                                            extra_types=extra_types)

    yield weight, result, apparent_types

  # # Attempt to parse, allowing for function composition. In order to support
  # # this we need to pass a dummy expression which is a lambda.
  # arities = {expr: get_arity(expr) for expr in candidate_expressions}
  # max_arity = max(arities.values())

  # results, sub_expr_original = [], sub_expr
  # for arity in range(1, max(arities.values()) + 1):
  #   sub_expr = sub_expr_original

  #   variables = [l.Variable("z%i" % (i + 1)) for i in range(arity)]
  #   # Build curried application expression.
  #   term = sub_expr
  #   for variable in variables:
  #     term = l.ApplicationExpression(term, l.IndividualVariableExpression(variable))

  #   # Build surrounding lambda expression.
  #   sub_expr = term
  #   for variable in variables[::-1]:
  #     sub_expr = l.LambdaExpression(variable, sub_expr)

  #   lexicon._entries[token] = [Token(token, candidate_category, sub_expr)]
  #   results.extend(
  #       chart.WeightedCCGChartParser(lexicon, ruleset=chart.DefaultRuleSet).parse(sentence))

  # lexicon._entries[token] = []
  # return results, sub_target


def augment_lexicon_unification(lex, sentence, ontology, lf):
  """
  Given a supervised `sentence -> lf` mapping, update the lexicon via a variant
  of unification-GENLEX.
  """
  # compute possible syntactic parses, allowing all tokens in the sentence to
  # belong to any category.
  parses = get_syntactic_parses(lex, sentence, sentence)
  _, _, best_parse = max(parses, key=lambda parse: parse[0])

  token_categs = {}
  for leaf_str, token in best_parse.pos():
    # TODO doesn't handle edge case where `token` appears multiple times with
    # different syntactic categories in the same sentence
    token_categs[leaf_str] = token.categ()

  # now attempt a (partial) semantic parse, allowing existing lexical entries
  # to participate in the scoring process.
  #
  # we'll begin by just inducing semantic representations for *novel* words,
  # and consider novel representations for existing tokens only if necessary.
  novel_tokens = set(tok for tok in sentence if not lex.get_entries(tok))
  known_tokens = set(sentence) - novel_tokens

  induce_tokens_chain = itertools.chain(
      [[]],
      itertools.chain.from_iterable(itertools.combinations(known_tokens, n)
                                    for n in range(1, len(known_tokens))))
  ready_for_unification = False
  for induction_set in induce_tokens_chain:
    # we are going to run unification, attempting to infer all novel token
    # meanings along with the meanings of the words in `induction_set`.
    to_induce = list(novel_tokens) + list(induction_set)
    candidate_categories = [token_categs[token] for token in to_induce]
    dummy_vars = {token: l.Variable("F%03i" % i) for i, token in enumerate(to_induce)}

    # fetch full candidate semantic parses, using dummy variables for all
    # induction tokens
    candidate_parses = attempt_candidate_parse(lex, to_induce, candidate_categories,
                                               sentence, dummy_vars)
    candidate_parses = list(candidate_parses)

    if not candidate_parses:
      # no valid parses for this induction set -- try the next one.
      continue
    else:
      ready_for_unification = True
      break

  if not ready_for_unification:
    # this may be because we greedily chose the best syntactic parse?
    L.warning("no successful parse possible. quitting.")
    return lex

  L.debug("Running unification induction with tokens: %s", to_induce)
  lex = lex.clone()

  # Take each candidate parse in turn as a "guiding parse." The function of
  # the guiding parse is to 1) constrain the ways in which LFs are split, and
  # 2) constrain the desired types of the lexical entries at the leaves of the
  # parse.
  #
  # TODO this could be extremely computationally expensive for large candidate
  # sets.
  new_entries = set()
  for _, guiding_parse, apparent_types in candidate_parses:
    # BFS through parse-guided semantic splits.
    queue = [(guiding_parse, lf)]
    while queue:
      node, expr = queue.pop()
      token, parse_op = node.label()
      if parse_op == "Leaf":
        # Is this an induction token? If so, make sure its type matches what's
        # expected under the guiding parse.
        token_str = token._token
        if token_str not in to_induce:
          # Not an induction token. Skip.
          continue

        # print(token_str, expr.type, apparent_types[token_str])
        if token_str in to_induce and not expr.type.resolve(apparent_types[token_str]):
          continue

        new_entries.add((token._token, token.categ(), expr))
        continue

      # get left and right children
      left, right = list(node)

      for left_split, right_split, cand_direction in ontology.iter_application_splits(expr):
        if parse_op == ">" and cand_direction != "/" \
            or parse_op == "<" and cand_direction != "\\":
          continue

        queue.append((left, left_split))
        queue.append((right, right_split))

  for token, categ, semantics in new_entries:
    lex.add_entry(token, categ, semantics=semantics, weight=0.001)

  return lex


def build_bootstrap_likelihood(lex, sentence, ontology,
                               alpha=0.25, meaning_prior_smooth=1e-3):
  """
  Prepare a likelihood function `p(meaning, syntax | sentence)` based on
  syntactic bootstrapping.

  Args:
    lex:
    sentence:
    ontology:
    alpha: Mixing parameter for bootstrapping distributions. See `alpha`
      parameter of `Lexicon.lf_ngrams_mixed`.

  Returns:
    likelihood_fn: A likelihood function to be used with `predict_zero_shot`.
  """
  # Prepare for syntactic bootstrap: pre-calculate distributions over semantic
  # form elements conditioned on syntactic category.
  lf_ngrams = lex.lf_ngrams_mixed(alpha=alpha, order=1,
                                  smooth=meaning_prior_smooth)
  for category in lf_ngrams:
    # Redistribute UNK probability uniformly across predicates not observed for
    # this category.
    unk_lf_prob = lf_ngrams[category].pop(None)
    unobserved_preds = set(f.name for f in ontology.functions) - set(lf_ngrams[category].keys())
    lf_ngrams[category].update({pred: unk_lf_prob / len(unobserved_preds)
                                for pred in unobserved_preds})

    L.info("% 20s %s", category,
           ", ".join("%.03f %s" % (prob, pred) for pred, prob
                     in sorted(lf_ngrams[category].items(), key=lambda x: x[1], reverse=True)))

  def likelihood_fn(tokens, categories, exprs, sentence_parse,
                    sentence_semantics, model):
    likelihood = 0.0
    for token, category, expr in zip(tokens, categories, exprs):
      # Retrieve relevant bootstrap distribution p(meaning | syntax).
      cat_lf_ngrams = lf_ngrams[category]
      for predicate in expr.predicates():
        if predicate.name in cat_lf_ngrams:
          likelihood += np.log(cat_lf_ngrams[predicate.name])

    return likelihood

  return likelihood_fn


def likelihood_scene(tokens, categories, exprs, sentence_parse,
                     sentence_semantics, model):
  """
  0-1 likelihood function, 1 when a sentence is true of the model and false
  otherwise.
  """
  try:
    return 0. if model.evaluate(sentence_semantics) == True else -np.inf
  except:
    return -np.inf


def build_distant_likelihood(answer):
  """
  Prepare a likelihood function `p(meaning, syntax | sentence)` based on
  distant supervision.

  Args:
    answer: ground-truth answer

  Returns:
    likelihood_fn: A likelihood function to be used with `predict_zero_shot`.
  """
  def likelihood_fn(tokens, categories, exprs, sentence_parse,
                    sentence_semantics, model):
    try:
      success = model.evaluate(sentence_semantics) == answer
    except:
      success = None

    return 0.0 if success == True else -np.inf

  return likelihood_fn


def likelihood_2afc(tokens, categories, exprs, sentence_parse,
                    sentence_semantics, models):
  """
  0-1 likelihood function for the 2AFC paradigm, where an uttered
  sentence is known to be true of at least one of two scenes.

  Args:
    models:

  Returns:
    log_likelihood:
  """
  model1, model2 = models
  try:
    model1_success = model1.evaluate(sentence_semantics) == True
  except:
    model1_success = None
  try:
    model2_success = model2.evaluate(sentence_semantics) == True
  except:
    model2_success = None

  return 0. if model1_success or model2_success else -np.inf


def likelihood_prominence(tokens, categories, exprs, sentence_parse,
                          sentence_semantics, model):
  """
  Likelihood function which enforces prominence preservation in the
  syntax--semantics mapping.

  (For any two arguments a, b, if a >= b in the derivation, then it must be
  that a >= b in the meaning representation.)
  """
  for expr in exprs:
    if not isinstance(expr, l.LambdaExpression):
      continue

    # Get semantic tree depths for each argument.
    args, body = expr.decompose()
    semantic_depths = {arg: l.get_depths(expr, arg) for arg in args}
    # Get max semantic depth for each argument.
    semantic_depths = {arg: max(depths.keys())
                       for arg, depths in semantic_depths.items()}

    # NB `args` already encodes syntactic depth -- earlier arguments must be
    # syntactically lower than later arguments.
    # TODO re-evaluate this definition.

    # Enforce prominence principle. If a > b in derivation depth, then a >= b
    # in semantic depth.
    for arg1, arg2 in zip(args, args[1:]):
      if not semantic_depths[arg1] >= semantic_depths[arg2]:
        return -np.inf

  return 0.


def predict_zero_shot(lex, tokens, candidate_syntaxes, sentence, ontology,
                      model, likelihood_fns, queue_limit=5, max_expr_depth=6):
  """
  Make zero-shot predictions of the posterior `p(meaning, syntax | sentence)`
  for each of `tokens`.

  Args:
    lex:
    tokens:
    candidate_syntaxes: `dict` mapping each token to a `Distribution` over
      syntactic types
    sentence:
    ontology:
    model:
    likelihood_fns: Collection of likelihood functions
      `p(meaning, syntax | sentence, model)` used to score candidate
      meaning--syntax settings for a subset of `tokens`.  Each function should
      accept arguments `(tokens, candidate_categories, candidate_meanings,
      candidate_syntactic_parse, candidate_semantic_parse, model)`, where
      `tokens` are assigned specific categories given in `candidate_categories`
      and specific meanings given in `candidate_meanings`, yielding a single
      syntactic analysis `candidate_syntactic_parse` and semantic analysis
      `candidate_semantic_parse` of the sentence. The function should return a
      log-likelihood `p(candidate_meanings | candidate_syntaxes, sentence,
      model)`.
    queue_limit: For any token, the maximum number of top-probability candidate
      lexical entries to return
    max_expr_depth: When enumerating semantic expressions, limit expression
      tree depth to this number.

  Returns:
    queues: A dictionary mapping each query token to a ranked sequence of
      candidates of the form
      `(logprob, (tokens, candidate_categories, candidate_semantics))`,
      describing a nonzero-probability novel mapping of a subset `tokens` to
      syntactic categories `candidate_categories` and meanings
      `candidate_semantics`. The log-probability value given is
      `p(meaning, syntax | sentence, model)`, under the relevant provided
      meaning likelihoods and the lexicon's distribution over syntactic forms.
    dummy_vars: TODO
  """

  # Shared dummy variables which is included in candidate semantic forms, to be
  # replaced by all candidate lexical expressions and evaluated.
  dummy_vars = {token: l.Variable("F%03i" % i) for i, token in enumerate(tokens)}

  category_parse_results = {}
  candidate_queue = None
  for depth in trange(1, len(tokens) + 1, desc="Depths"):
    candidate_queue = UniquePriorityQueue(maxsize=queue_limit)

    token_combs = list(itertools.combinations(tokens, depth))
    for token_comb in tqdm(token_combs, desc="Token combinations"):
      token_syntaxes = [list(candidate_syntaxes[token].support) for token in token_comb]
      for syntax_comb in tqdm(itertools.product(*token_syntaxes),
                              total=np.prod(list(map(len, token_syntaxes))),
                              desc="Syntax combinations"):
        syntax_weights = [candidate_syntaxes[token][cat] for token, cat in zip(token_comb, syntax_comb)]
        if any(weight == 0 for weight in syntax_weights):
          continue

        # Attempt to parse with this joint syntactic assignment, and return the
        # resulting syntactic parses + sentence-level semantic forms, with
        # dummy variables in place of where the candidate expressions will go.
        results = attempt_candidate_parse(lex, token_comb,
                                          syntax_comb,
                                          sentence,
                                          dummy_vars)
        results = list(results)
        category_parse_results[syntax_comb] = results

        for _, result, apparent_types in results:
          L.debug("Searching for expressions with types: %s",
                  ";".join("%s: %s" % (token, apparent_types[token])
                           for token in token_comb))
          candidate_exprs = [list(ontology.iter_expressions(
                              max_depth=max_expr_depth,
                              type_request=apparent_types[token]))
                             for token in token_comb]

          n_expr_combs = np.prod(list(map(len, candidate_exprs)))
          for expr_comb in tqdm(itertools.product(*candidate_exprs),
                                total=n_expr_combs,
                                desc="Expressions"):
            # Compute likelihood of this joint syntax--semantics assignment.
            likelihood = 0.0

            # Swap in semantic values for each token.
            sentence_semantics = result.label()[0].semantics()
            for token, token_expr in zip(token_comb, expr_comb):
              dummy_var = dummy_vars[token]
              sentence_semantics = sentence_semantics.replace(dummy_var, token_expr)
            sentence_semantics = sentence_semantics.simplify()

            # TODO integrate p(meaning)

            # Compute p(meaning, syntax | sentence, parse)
            logp = sum(likelihood_fn(token_comb, syntax_comb, expr_comb,
                                     result, sentence_semantics, model)
                      for likelihood_fn in likelihood_fns)
            joint_score = logp

            if logp == -np.inf:
              # Zero probability. Skip.
              continue

            data = tuple_unordered([token_comb, syntax_comb, expr_comb])
            new_item = (joint_score, data)
            try:
              candidate_queue.put_nowait(new_item)
            except queue.Full:
              # See if this candidate is better than the worst item.
              worst = candidate_queue.get()
              if worst[0] < joint_score:
                replacement = new_item
              else:
                replacement = worst

              candidate_queue.put_nowait(replacement)

    if candidate_queue.qsize() > 0:
      # We have a result. Quit and don't search at higher depth.
      return candidate_queue, dummy_vars

  return candidate_queue, dummy_vars


def augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                    sentence, ontology, model, likelihood_fns,
                    beta=3.0,
                    **predict_zero_shot_args):
  """
  Augment a lexicon with candidate meanings for a given word using an abstract
  likelihood measure. (The induced meanings for the queried words must yield
  parses that have nonzero posterior probability, given the lexicon and
  `model`.)

  Candidate entries will be assigned relative weights according to a posterior
  distribution $P(word -> syntax, meaning | sentence, success_fn, lexicon)$. This
  distribution incorporates multiple prior and likelihood terms:

  1. A prior over syntactic categories (derived internally, by inspection of
     the current lexicon)
  2. A likelihood over meanings (specified by `likelihood_fns`)

  Arguments:
    old_lex: Existing lexicon which needs to be augmented. Does not write
      in-place.
    query_words: Set of tokens for which we need to search for novel lexical
      entries.
    query_word_syntaxes: Possible syntactic categories for each of the query
      words, as returned by `get_candidate_categories`.
    sentence: Token list sentence.
    ontology: Available logical ontology -- used to enumerate possible logical
      forms.
    model: Scene model which evaluates logical forms to answers.
    likelihood_fns: Collection of functions describing zero-shot likelihoods
      `p(meaning, syntax | sentence, model)`. See `predict_zero_shot` for
      more information.
    beta: Total mass to assign to novel candidate lexical entries per each
      wordform. (Mass will be divided according to the posterior distribution
      given above.)
  """

  # Target lexicon to be returned.
  lex = old_lex.clone()

  ranked_candidates, dummy_vars = \
      predict_zero_shot(lex, query_tokens, query_token_syntaxes, sentence,
                        ontology, model, likelihood_fns, **predict_zero_shot_args)

  candidates = sorted(ranked_candidates.queue, key=lambda item: -item[0])
  new_entries = {token: Counter() for token in query_tokens}
  # Calculate marginal p(syntax, meaning | sentence) for each token.
  for logp, (tokens, syntaxes, meanings) in candidates:
    for token, syntax, meaning in zip(tokens, syntaxes, meanings):
      new_entries[token][syntax, meaning] += np.exp(logp)

  if all(len(candidates) == 0 for candidates in new_entries.values()):
    raise NoParsesError("Failed to derive any meanings for tokens %s."
                        % query_tokens, sentence)

  # Construct a new lexicon.
  for token, candidates in new_entries.items():
    # HACK: post-hoc normalize by meaning. This should come out naturally once
    # we have the right generative model defined earlier on.
    meaning_masses = Counter()
    decomposed_meanings = []
    for (_, meaning), weight in candidates.items():
      if isinstance(meaning, l.VariableBinderExpression):
        # ignore variable binding; condition only on body
        _, meaning = meaning.decompose()
      decomposed_meanings.append(meaning)
      meaning_masses[meaning] += weight

    candidates = {(syntax, meaning): weight / meaning_masses[decomposed_meaning]
                  for ((syntax, meaning), weight), decomposed_meaning
                  in zip(candidates.items(), decomposed_meanings)}

    total_mass = sum(candidates.values())
    if len(candidates) > 0:
      lex.set_entries(token,
                      [(syntax, meaning, weight / total_mass * beta)
                       for (syntax, meaning), weight in candidates.items()])

      L.info("Inferred %i novel entries for token %s:", len(candidates), token)
      for entry, weight in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
        L.info("%.4f %s %s", weight / total_mass * beta, entry[0], entry[1])

  return lex


def augment_lexicon_distant(old_lex, query_tokens, query_token_syntaxes,
                            sentence, ontology, model, likelihood_fns, answer,
                            **augment_kwargs):
  """
  Augment a lexicon with candidate meanings for a given word using distant
  supervision. (Find word meanings such that the whole sentence meaning yields
  an expected `answer`.)

  For argument documentation, see `augment_lexicon`.
  """
  likelihood_fns = (build_distant_likelihood(answer),) + \
      tuple(likelihood_fns)

  return augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, model, likelihood_fns,
                         **augment_kwargs)


def augment_lexicon_cross_situational(old_lex, query_tokens, query_token_syntaxes,
                                      sentence, ontology, model, likelihood_fns,
                                      **augment_kwargs):
  likelihood_fns = (likelihood_scene,) + tuple(likelihood_fns)
  return augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, model, likelihood_fns,
                         **augment_kwargs)


def augment_lexicon_2afc(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, models, likelihood_fns,
                         **augment_kwargs):
  """
  Augment a lexicon with candidate meanings for a given word using 2AFC
  supervision. (We assume that the uttered sentence is true of at least one of
  the 2 scenes given in the tuple `models`.)

  For argument documentation, see `augment_lexicon`.
  """
  likelihood_fns = (likelihood_2afc,) + tuple(likelihood_fns)
  return augment_lexicon(old_lex, query_tokens, query_token_syntaxes,
                         sentence, ontology, models, likelihood_fns,
                         **augment_kwargs)


def filter_lexicon_entry(lexicon, entry, sentence, lf):
  """
  Filter possible syntactic/semantic mappings for a given lexicon entry s.t.
  the given sentence renders the given LF, holding everything else
  constant.

  This process is of course not fail-safe -- the rest of the lexicon must
  provide the necessary definitions to guarantee that any valid parse can
  result.

  Args:
    lexicon: CCGLexicon
    entry: string word
    sentence: list of word tokens, must contain `entry`
    lf: logical form string
  """
  if entry not in sentence:
    raise ValueError("Sentence does not contain given entry")

  entry_idxs = [i for i, val in enumerate(sentence) if val == entry]
  parse_results = chart.WeightedCCGChartParser(lexicon).parse(sentence, True)

  valid_cands = [set() for _ in entry_idxs]
  for _, _, edge_cands in parse_results:
    for entry_idx, valid_cands_set in zip(entry_idxs, valid_cands):
      valid_cands_set.add(edge_cands[entry_idx])

  # Find valid interpretations across all uses of the word in the
  # sentence.
  valid_cands = list(reduce(lambda x, y: x & y, valid_cands))
  if not valid_cands:
    raise ValueError("no consistent interpretations of word found.")

  new_lex = lexicon.clone()
  new_lex.set_entries(entry, [(cand.token().categ(), cand.token().semantics(), cand.token().weight())
                              for cand in valid_cands])

  return new_lex
