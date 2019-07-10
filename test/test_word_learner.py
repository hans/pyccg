import sys

from nose.tools import *

from pyccg import chart
from pyccg import lexicon as lex
from pyccg import logic as log
from pyccg.model import Model
from pyccg.word_learner import *


def _make_mock_learner(**kwargs):
  ## Specify ontology.
  types = log.TypeSystem(["object", "location", "boolean", "action"])
  functions = [
    types.new_function("go", ("location", "action"), lambda x: ("go", x)),
  ]
  constants = [
    types.new_constant("there", "location"),
    types.new_constant("here", "location"),
    types.new_constant("away", "location"),
  ]
  ont = log.Ontology(types, functions, constants)

  ## Specify initial lexicon.
  lexicon = lex.Lexicon.fromstring(r"""
  :- S, N

  goto => S/N {\x.go(x)}
  go => S/N {\x.go(x)}
  there => N {there}
  away => N {away}
  """, ontology=ont, include_semantics=True)

  return WordLearner(lexicon, **kwargs)


def _make_mock_model(learner):
  """
  Mock learner does not have any grounding -- just build a spurious Model
  instance.
  """
  return Model({"objects": []}, learner.ontology)


def test_update_distant_existing_words():
  """
  update_distant with no novel words
  """
  sentence = "goto there".split()
  answer = ("go", "there")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  old_lex.debug_print()
  print("====\n")
  learner.lexicon.debug_print()
  eq_(old_lex, learner.lexicon)


def test_update_distant_one_novel_word():
  """
  update_distant with one novel word
  """
  sentence = "goto here".split()
  answer = ("go", "here")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  old_lex.debug_print()
  print("====\n")
  learner.lexicon.debug_print()

  # other words should not have changed.
  eq_(old_lex._entries["there"], learner.lexicon._entries["there"])
  eq_(old_lex._entries["goto"], learner.lexicon._entries["goto"])

  eq_(len(learner.lexicon._entries["here"]), 1, "One valid new word entry")
  entry = learner.lexicon._entries["here"][0]
  eq_(str(entry.categ()), "N")
  eq_(str(entry.semantics()), "here")


def test_update_distant_one_novel_sense():
  """
  update_distant with one novel sense for an existing wordform
  """
  sentence = "goto there".split()
  answer = ("go", "here")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for valid sentence")

  old_lex.debug_print()
  print("====\n")
  learner.lexicon.debug_print()

  # other words should not have changed.
  eq_(old_lex._entries["goto"], learner.lexicon._entries["goto"])

  eq_(len(learner.lexicon._entries["there"]), 1, "New entry for 'there'")
  entry = learner.lexicon._entries["there"][0]
  eq_(str(entry.categ()), "N")
  eq_(str(entry.semantics()), "here")

  # expected_entries = [
  #   ("N", "here"),
  #   ("N", "there")
  # ]
  # entries = [(str(entry.categ()), str(entry.semantics())) for entry in learner.lexicon._entries["there"]]
  # eq_(set(expected_entries), entries)


def test_update_distant_two_novel_words():
  """
  update_distant with two novel words
  """
  sentence = "allez y".split()
  answer = ("go", "there")
  learner = _make_mock_learner()
  old_lex = learner.lexicon.clone()

  model = _make_mock_model(learner)
  results = learner.update_with_distant(sentence, model, answer)
  ok_(len(results) > 0, "Parser has >0 parses for sentence")

  old_lex.debug_print()
  print("======\n")
  learner.lexicon.debug_print()

  expected_entries = {
    "allez": set([("(S/N)", r"go")]),
    "y": set([("N", r"there")])
  }
  for token, expected in expected_entries.items():
    entries = learner.lexicon._entries[token]
    eq_(set([(str(e.categ()), str(e.semantics())) for e in entries]),
        expected)


def _make_complex_mock_ontology():
  def fn_unique(xs):
    true_xs = [x for x, matches in xs.items() if matches]
    assert len(true_xs) == 1
    return true_xs[0]

  types = log.TypeSystem(["obj", "num", "ax", "boolean"])

  functions = [
    types.new_function("cmp_pos", ("ax", "obj", "obj", "num"),
                       lambda ax, a, b: a["3d_coords"][ax()] - b["3d_coords"][ax()]),
    types.new_function("ltzero", ("num", "boolean"), lambda x: x < 0),

    types.new_function("ax_x", ("ax",), lambda: 0),
    types.new_function("ax_y", ("ax",), lambda: 1),
    types.new_function("ax_z", ("ax",), lambda: 2),

    types.new_function("unique", (("obj", "boolean"), "obj"), fn_unique),

    types.new_function("cube", ("obj", "boolean"), lambda x: x["shape"] == "cube"),
    types.new_function("sphere", ("obj", "boolean"), lambda x: x["shape"] == "sphere"),

    types.new_function("and_", ("boolean", "boolean", "boolean"), lambda x, y: x and y),
  ]

  constants = [types.new_constant("one", "num"), types.new_constant("two", "num")]

  ontology = log.Ontology(types, functions, constants, variable_weight=0.1)

  return ontology


def test_update_with_supervision():
  """
  supervised learning update
  """
  ontology = _make_complex_mock_ontology()
  lexicon = lex.Lexicon.fromstring(r"""
  :- N

  or => N\N/N {\F G x.and_(F(x),G(x))}
  blork => N {cube}
  spork => N {cube}
  """, ontology=ontology, include_semantics=True)

  learner = WordLearner(lexicon)

  sentence = "cubic and spherical".split()
  model = _make_mock_model(learner)
  answer = log.Expression.fromstring(r"\x.and_(cube(x),sphere(x))")

  learner.update_with_supervision(sentence, learner, answer)
