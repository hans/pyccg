from nose.tools import *

from pyccg.chart import *
from pyccg.lexicon import Lexicon

def _make_lexicon_with_derived_category():
  lex = Lexicon.fromstring(r"""
  :- S, NP

  the => S/NP {\x.unique(x)}

  foo => NP {\x.foo(x)}
  bar => NP {\x.bar(x)}
  baz => NP {\x.baz(x)}
  """, include_semantics=True)
  old_lex = lex.clone()

  # Induce a derived category involving `foo` and `bar`.
  involved_tokens = [lex._entries["foo"][0], lex._entries["bar"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)

  return old_lex, lex, involved_tokens, derived_categ


def test_parse_with_derived_category():
  """
  Ensure that we can still parse with derived categories.
  """

  old_lex, lex, involved_tokens, categ_name = _make_lexicon_with_derived_category()
  lex.propagate_derived_category(categ_name)

  old_results = WeightedCCGChartParser(old_lex).parse("the foo".split())
  results = WeightedCCGChartParser(lex).parse("the foo".split())

  eq_(len(results), len(results))
  eq_(results[0].label()[0].semantics(), old_results[0].label()[0].semantics())


def test_parse_with_derived_root_category():
  """
  Ensure that we can parse with a derived category whose base is the root
  category.
  """
  lex = Lexicon.fromstring(r"""
      :- S, N
      the => S/N {\x.unique(x)}
      foo => N {\x.foo(x)}
      """, include_semantics=True)

  involved_tokens = [lex._entries["the"][0]]
  derived_categ = lex.add_derived_category(involved_tokens)
  lex.propagate_derived_category(derived_categ)
  derived_categ_obj, _ = lex._derived_categories[derived_categ]

  results = WeightedCCGChartParser(lex).parse("the foo".split())
  eq_(set(str(result.label()[0].categ()) for result in results),
      {"S", str(derived_categ_obj)})


def test_parse_oblique():
  """
  Test parsing a verb with an oblique PP -- this shouldn't require type raising?
  """

  lex = Lexicon.fromstring(r"""
  :- S, NP, PP

  place => S/NP/PP
  it => NP
  on => PP/NP
  the_table => NP
  """)

  parser = WeightedCCGChartParser(lex, ApplicationRuleSet)
  printCCGDerivation(parser.parse("place it on the_table".split())[0])


def test_parse_oblique_raised():
  lex = Lexicon.fromstring(r"""
  :- S, NP, PP

  place => S/NP/(PP/NP)/NP
  it => NP
  on => PP/NP
  the_table => NP
  """)

  parser = WeightedCCGChartParser(lex, DefaultRuleSet)
  printCCGDerivation(parser.parse("place it on the_table".split())[0])


def test_get_derivation_tree():
  lex = Lexicon.fromstring(r"""
  :- S, N

  John => N
  saw => S\N/N
  Mary => N
  """)

  parser = WeightedCCGChartParser(lex, ruleset=DefaultRuleSet)
  top_parse = parser.parse("Mary saw John".split())[0]

  from io import StringIO
  stream = StringIO()
  get_clean_parse_tree(top_parse).pretty_print(stream=stream)

  eq_([line.strip() for line in stream.getvalue().strip().split("\n")],
      [line.strip() for line in r"""
         S
  _______|_______
 |             (S\N)
 |        _______|____
 N   ((S\N)/N)        N
 |       |            |
Mary    saw          John""".strip().split("\n")])
