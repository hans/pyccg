from nose.tools import *

from pyccg.chart import *
from pyccg.lexicon import Lexicon


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
