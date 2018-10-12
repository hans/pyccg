from nose.tools import *

from pyccg.lexicon import *

from nltk.ccg.lexicon import FunctionalCategory, PrimitiveCategory, Direction
from nltk.sem.logic import Expression


def test_filter_lexicon_entry():
  lex = Lexicon.fromstring(r"""
    :- NN, DET, ADJ

    DET :: NN/NN
    ADJ :: NN/NN

    the => DET {\x.unique(x)}
    sphere => NN {filter_shape(scene,sphere)}
    sphere => NN {filter_shape(scene,cube)}
    """, include_semantics=True)

  lex_filtered = filter_lexicon_entry(lex, "sphere", "the sphere".split(), "unique(filter_shape(scene,sphere))")

  entries = lex_filtered.categories("sphere")
  assert len(entries) == 1

  eq_(str(entries[0].semantics()), "filter_shape(scene,sphere)")


def test_get_semantic_arity():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- NN, DET, ADJ

    DET :: NN/NN
    ADJ :: NN/NN

    the => DET {\x.unique(x)}
    sphere => NN {filter_shape(scene,sphere)}
    sphere => NN {filter_shape(scene,cube)}
    """, include_semantics=True)

  cases = [
      (r"NN", 0),
      (r"NN/NN", 1),
      (r"NN\NN", 1),
      (r"(NN\NN)/NN", 2),
  ]

  def test_case(cat, expected):
    eq_(get_semantic_arity(augParseCategory(cat, lex._primitives, lex._families)[0]),
        expected, msg=str(cat))

  for cat, expected in cases:
    yield test_case, cat, expected


def test_get_lf_unigrams():
  lex = Lexicon.fromstring(r"""
    :- NN

    the => NN/NN {\x.unique(x)}
    sphere => NN {\x.and_(object(x),sphere(x))}
    cube => NN {\x.and_(object(x),cube(x))}
    """, include_semantics=True)

  expected = {
    "NN": Counter({"and_": 2 / 6, "object": 2 / 6, "sphere": 1 / 6, "cube": 1 / 6, None: 0.0, "unique": 0.0}),
    "(NN/NN)": Counter({"unique": 1, "cube": 0.0, "object": 0.0, "sphere": 0.0, "and_": 0.0, None: 0.0})
  }

  ngrams = lex.lf_ngrams_given_syntax(order=1, smooth=False)
  for categ, dist in ngrams.dists.items():
    eq_(dist, expected[str(categ)])


def test_get_yield():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- S, NN, PP

    on => PP/NN
    the => S/NN
    the => NN/NN
    sphere => NN
    sphere => NN
    """)

  cases = [
      ("S", "S"),
      ("S/NN", "S"),
      (r"NN\PP/NN", "NN"),
      (r"(S\NN)/NN", "S"),
  ]

  def test_case(cat, cat_yield):
    eq_(get_yield(lex.parse_category(cat)), lex.parse_category(cat_yield))

  for cat, cat_yield in cases:
    yield test_case, cat, cat_yield


def test_set_yield():
  from nltk.ccg.lexicon import augParseCategory
  lex = Lexicon.fromstring(r"""
    :- S, NN, PP

    on => PP/NN
    the => S/NN
    the => NN/NN
    sphere => NN
    sphere => NN
    """)

  cases = [
      ("S", "NN", "NN"),
      ("S/NN", "NN", "NN/NN"),
      (r"NN\PP/NN", "S", r"S\PP/NN"),
      (r"(S\NN)/NN", "NN", r"(NN\NN)/NN"),
  ]

  def test_case(cat, update, expected):
    source = lex.parse_category(cat)
    updated = set_yield(source, update)

    eq_(str(updated), str(lex.parse_category(expected)))

  for cat, update, expected in cases:
    yield test_case, cat, update, expected


def test_attempt_candidate_parse():
  """
  Find parse candidates even when the parse requires composition.
  """
  lex = Lexicon.fromstring(r"""
  :- S, N

  gives => S\N/N/N {\o x y.give(x, y, o)}
  John => N {\x.John(x)}
  Mark => N {\x.Mark(x)}
  it => N {\x.T}
  """, include_semantics=True)
  # TODO this doesn't actually require composition .. get one which does

  cand_category = lex.parse_category(r"S\N/N/N")
  cand_expressions = [Expression.fromstring(r"\o x y.give(x,y,o)")]
  results = attempt_candidate_parse(lex, "sends", cand_category,
                                    cand_expressions, "John sends Mark it".split())

  ok_(len(results) > 0)
