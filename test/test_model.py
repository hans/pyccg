from nose.tools import *

from frozendict import frozendict

from pyccg.logic import Expression, Ontology, TypeSystem, Function
from pyccg.model import *


def test_model_constants():
  """
  Test evaluating with constant values.
  """
  types = TypeSystem(["num"])

  functions = [
    types.new_function("add", ("num", "num", "num"), lambda a, b: str(int(a) + int(b)))
  ]
  constants = [types.new_constant("1", "num"), types.new_constant("2", "num")]

  ontology = Ontology(types, functions, constants)
  model = Model(scene={"objects": []}, ontology=ontology)

  cases = [
    ("Test basic constant evaluation", r"1", "1"),
    ("Test constants as arguments to functions", r"add(1,1)", "2"),
  ]

  def test(msg, expr, expected):
    print("ret", model.evaluate(Expression.fromstring(expr)))
    eq_(model.evaluate(Expression.fromstring(expr)), expected, msg=msg)

  for msg, expr, expected in cases:
    yield test, msg, expr, expected


def test_model_induced_functions():
  """
  Test evaluating a model with an ontology which has induced functions.
  """

  fake_scene = {
    "objects": ["foo", "bar"],
  }

  types = TypeSystem(["a"])
  functions = [
      types.new_function("test", ("a", "a"), lambda x: True),
      types.new_function("test2", ("a", "a"), Expression.fromstring(r"\x.test(test(x))")),
  ]
  ontology = Ontology(types, functions, [])

  model = Model(scene=fake_scene, ontology=ontology)

  cases = [
    ("Test basic call of an abstract function", r"\a.test2(a)", {"foo": True, "bar": True}),
    ("Test embedded call of an abstract function", r"\a.test(test2(a))", {"foo": True, "bar": True}),
  ]

  def test(msg, expr, expected):
    eq_(model.evaluate(Expression.fromstring(expr)), expected)

  for msg, expr, expected in cases:
    yield test, msg, expr, expected


def test_model_partial_application():
  types = TypeSystem(["obj"])
  functions = [
    types.new_function("lotsofargs", ("obj", "obj", "obj"), lambda a, b: b),
  ]
  constants = [
      types.new_constant("obj1", "obj"),
      types.new_constant("obj2", "obj"),
  ]
  ontology = Ontology(types, functions, constants)

  scene = {"objects": []}
  model = Model(scene, ontology)

  eq_(model.evaluate(Expression.fromstring(r"(lotsofargs(obj1))(obj2)")), "obj2")


def test_model_stored_partial_application():
  types = TypeSystem(["obj"])
  functions = [
    types.new_function("lotsofargs", ("obj", "obj", "obj"), lambda a, b: b),
  ]
  constants = [
      types.new_constant("obj1", "obj"),
      types.new_constant("obj2", "obj"),
  ]
  ontology = Ontology(types, functions, constants)
  ontology.add_functions([types.new_function("partial", ("obj", "obj"), Expression.fromstring(r"lotsofargs(obj2)"))])

  scene = {"objects": []}
  model = Model(scene, ontology)

  eq_(model.evaluate(Expression.fromstring(r"partial(obj1)")), "obj1")
