from nose.tools import *

from nltk.sem.logic import Expression, Variable, \
    FunctionVariableExpression, AndExpression, NegatedExpression

from pyccg.logic import *


def _make_mock_ontology():
  def fn_unique(xs):
    true_xs = [x for x, matches in xs.items() if matches]
    assert len(true_xs) == 1
    return true_xs[0]

  types = TypeSystem(["obj", "num", "ax", "boolean"])

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

  ontology = Ontology(types, functions, constants, variable_weight=0.1)

  return ontology


def _make_simple_mock_ontology():
  types = TypeSystem(["boolean", "obj"])
  functions = [
      types.new_function("and_", ("boolean", "boolean", "boolean"), lambda x, y: x and y),
      types.new_function("foo", ("obj", "boolean"), lambda x: True),
      types.new_function("bar", ("obj", "boolean"), lambda x: True),
      types.new_function("not_", ("boolean", "boolean"), lambda x: not x),

      types.new_function("invented_1", (("obj", "boolean"), "obj", "boolean"), lambda f, x: x is not None and f(x)),

      types.new_function("threeplace", ("obj", "obj", "boolean", "boolean"), lambda x, y, o: True),
  ]
  constants = [types.new_constant("baz", "boolean"), types.new_constant("qux", "obj")]

  ontology = Ontology(types, functions, constants, variable_weight=0.1)
  return ontology


def test_type_match():
  ont = _make_simple_mock_ontology()
  ok_(ont.types["boolean"].matches(ont.types["e"]),
      "subtypes are recognized in type.matches")
  ok_(not ont.types["e"].matches(ont.types["boolean"]),
      "type.matches enforces asymmetric-ness of subtype relation")
  ok_(not ont.constants_dict["baz"].type.matches(ont.constants_dict["qux"].type),
      "'boolean' and 'obj' types should not match")

  ok_(ont.types["?"].matches(ont.types["boolean"]),
       "wildcard basic type matches in both directions")
  ok_(ont.types["boolean"].matches(ont.types["?"]),
       "wildcard basic type matches in both directions")
  ok_(not ont.types["?"].matches(ont.types["boolean", "boolean"]),
       "wildcard basic type does not match with complex types")
  ok_(not ont.types["?"].matches(ont.types["?", "?"]),
       "wildcard basic type does not match with complex types")

  ok_(ont.types["*"].matches(ont.types["?"]),
       "wildcard type matches with wildcard basic type")
  ok_(ont.types["?"].matches(ont.types["*"]),
       "wildcard type matches with wildcard basic type")
  ok_(ont.types["*"].matches(ont.types["boolean"]),
       "wildcard type matches with basic types")
  ok_(ont.types["*"].matches(ont.types["boolean", "boolean"]),
       "wildcard type matches with complex types")


def test_get_expr_arity():
  ont = _make_simple_mock_ontology()

  cases = [
      (r"\x.x", 1),
      (r"x", 0),
  ]

  def do_case(expr, expected):
    expr = Expression.fromstring(expr)
    eq_(ont.get_expr_arity(expr), expected)

  for expr, expected in cases:
    yield do_case, expr, expected


def test_extract_lambda():
  """
  `extract_lambda` should support all possible orderings of the variables it
  encounters.
  """
  expr = Expression.fromstring(r"foo(\a.a,\a.a)")
  extracted = extract_lambda(expr)
  eq_(len(extracted), 2)


def test_iter_expressions():
  ontology = _make_simple_mock_ontology()
  from pprint import pprint

  cases = [
    (3, "Reuse of bound variable",
      ((("boolean", "boolean"), r"\z1.and_(z1,z1)",),),
      ()),
    (3, "Support passing functions as arguments to higher-order functions",
     ((("obj", "boolean"), r"\z1.invented_1(foo,z1)",),),
     ()),
    (3, "Consider both argument orders",
     ((("boolean", "boolean", "boolean"), r"\z2 z1.and_(z1,z2)"),
      (("boolean", "boolean", "boolean"), r"and_")),
     ()),
    (3, "Consider both argument orders for three-place function",
     ((("obj", "obj", "boolean"), r"\z2 z1.threeplace(z1,z2,baz)"),
      (("obj", "obj", "boolean"), r"\z2 z1.threeplace(z2,z1,baz)")),
     ()),
    (3, "Enforce type constraints on higher-order functions",
     (),
     ((("obj", "boolean"), r"\z1.invented_1(not_,z1)",),)),
    (3, "Enforce type constraints on constants",
     (),
     ((("boolean", "boolean"), r"\z1.and_(z1,qux)",),)),
    (3, "Enforce type constraints on lambda expressions as arguments",
     (),
     ((("boolean"), r"and_(\z1.z1,\z1.z1)",),)),
    (5, "Support passing lambdas as function arguments",
     ((("boolean"), r"invented_1(\z1.not_(foo(z1)),qux)",),),
     ()),
    (3, "Support abstract type requests",
      ((("boolean", "boolean", "boolean"), r"and_"),
       (("boolean", "boolean"), r"\z1.and_(z1,z1)"),
       (("e", "e", "e"), r"and_"),
       (("e", "e"), r"\z1.and_(z1,z1)"),),
      ()),
    (3, r"Don't enumerate syntactically equivalent `\x.f(x)` and `f`",
      ((("e", "e"), r"not_"),),
      ((("e", "e"), r"\z1.not_(z1)"),)),
  ]

  def do_case(max_depth, msg, assert_in, assert_not_in):
    def get_exprs(max_depth, type_request):
      type_request = ontology.types[type_request]
      expressions = set(ontology.iter_expressions(max_depth=max_depth,
                                                  type_request=type_request))
      expression_strs = sorted(map(str, expressions))
      return expression_strs

    for type_request, expr in assert_in:
      ok_(expr in get_exprs(max_depth, type_request),
          "%s: for type request %s, should contain %s" % (msg, type_request, expr))
    for type_request, expr in assert_not_in:
      ok_(expr not in get_exprs(max_depth, type_request),
          "%s: for type request %s, should not contain %s" % (msg, type_request, expr))

  for max_depth, msg, assert_in, assert_not_in in cases:
    yield do_case, max_depth, msg, assert_in, assert_not_in


def test_iter_expressions_with_used_constants():
  ontology = _make_simple_mock_ontology()

  ontology.register_expressions([Expression.fromstring(r"\z1.and_(foo(z1),baz)")])
  expressions = set(ontology.iter_expressions(max_depth=3, use_unused_constants=True))
  expression_strs = list(map(str, expressions))

  ok_(r"foo(qux)" in expression_strs, "Use of new constant variable")
  ok_(r"baz" not in expression_strs, "Cannot use used constant variable")


def test_iter_expressions_after_update():
  """
  Ensure that ontology correctly returns expression options after adding a new
  function.
  """
  ontology = _make_simple_mock_ontology()
  ontology.add_functions([ontology.types.new_function("newfunction", ("obj", "boolean"), lambda a: True)])

  expressions = set(ontology.iter_expressions(max_depth=3, type_request=ontology.types["obj", "boolean"]))
  expression_strs = sorted(map(str, expressions))
  assert r"newfunction" in expression_strs


def test_as_ec_sexpr():
  ont = _make_mock_ontology()
  expr = Expression.fromstring(r"\x y z.foo(bar(x,y),baz(y,z),blah)")
  eq_(ont.as_ec_sexpr(expr), "(lambda (lambda (lambda (foo (bar $2 $1) (baz $1 $0) blah))))")


def test_as_ec_sexpr_function():
  ont = _make_mock_ontology()
  expr = FunctionVariableExpression(Variable("and_", ont.types["boolean", "boolean", "boolean"]))
  eq_(ont.as_ec_sexpr(expr), "(lambda (lambda (and_ $1 $0)))")


def test_as_ec_sexpr_event():
  types = TypeSystem(["obj"])
  functions = [
    types.new_function("e", ("v",), lambda: ()),
    types.new_function("result", ("v", "obj"), lambda e: e),
  ]
  constants = []

  ontology = Ontology(types, functions, constants)

  cases = [
    (r"result(e)", "(result e)"),
    (r"\x.foo(x,e)", "(lambda (foo $0 e))"),
    (r"\x.foo(e,x)", "(lambda (foo e $0))"),
    (r"\x.foo(x,e,x)", "(lambda (foo $0 e $0))"),
    (r"\a.constraint(ltzero(cmp_pos(ax_z,pos,e,a)))", "(lambda (constraint (ltzero (cmp_pos ax_z pos e $0))))"),
  ]

  def do_case(expr, expected):
    expr = Expression.fromstring(expr)
    eq_(ontology.as_ec_sexpr(expr), expected)

  for expr, expected in cases:
    yield do_case, expr, expected


def test_read_ec_sexpr():
  ontology = _make_simple_mock_ontology()
  expr, bound_vars = ontology.read_ec_sexpr(
      "(lambda (lambda (lambda (and_ (threeplace $0 qux $1) (and_ (foo $2) baz)))))")
  eq_(expr, Expression.fromstring(r"\a b c.and_(threeplace(c,qux,b),and_(foo(a),baz))"))
  eq_(len(bound_vars), 3)


def test_read_ec_sexpr_de_bruijn():
  """
  properly handle de Bruijn indexing in EC lambda expressions.
  """
  ontology = _make_simple_mock_ontology()
  expr, bound_vars = ontology.read_ec_sexpr("(lambda ((lambda ($0 (lambda $0))) (lambda ($1 $0))))")
  print(expr)
  eq_(expr, Expression.fromstring(r"\A.((\B.B(\C.C))(\C.A(C)))"))
  eq_(len(bound_vars), 4)


def test_read_ec_sexpr_nested():
  """
  read_ec_sexpr should support reading in applications where the function
  itself is an expression (i.e. there is some not-yet-reduced beta reduction
  candidate).
  """
  ontology = _make_simple_mock_ontology()
  expr, bound_vars = ontology.read_ec_sexpr("(lambda ((lambda (abc $0)) $0))")
  eq_(expr, Expression.fromstring(r"\a.((\b.abc(b))(a))"))
  eq_(len(bound_vars), 2)


def test_read_ec_sexpr_higher_order_param():
  ontology = _make_simple_mock_ontology()
  expr, bound_vars = ontology.read_ec_sexpr("(lambda (lambda ($0 $1)))")
  eq_(expr, Expression.fromstring(r"\a P.P(a)"))
  eq_(len(bound_vars), 2)


def test_read_ec_sexpr_typecheck():
  ontology = _make_simple_mock_ontology()

  expr, bound_vars = ontology.read_ec_sexpr("(lambda (foo $0))")
  eq_(expr.type, ontology.types["obj", "boolean"],
      "read_ec_sexpr runs type inference")
  eq_(expr.term.type, ontology.types["boolean"])
  bound_var, = bound_vars
  eq_(bound_var.type, ontology.types["obj"],
      "read_ec_sexpr infers types of bound variables")


def test_read_ec_sexpr_typecheck_complex():
  ontology = _make_mock_ontology()
  expr, bound_vars = ontology.read_ec_sexpr(
      "(lambda (unique (lambda (and_ (cube $0) (ltzero (cmp_pos ax_x $0 $1))))))")
  eq_(expr.type, ontology.types["obj", "obj"])
  eq_(len(bound_vars), 2)
  eq_([var.type for var in bound_vars], [ontology.types["obj"]] * len(bound_vars))


def test_valid_lambda_expr():
  """
  Regression test: valid_lambda_expr was rejecting this good sub-expression at c720b4
  """
  ontology = _make_mock_ontology()
  eq_(ontology._valid_lambda_expr(Expression.fromstring(r"\b.ltzero(cmp_pos(ax_x,a,b))"), ctx_bound_vars=()), False)
  eq_(ontology._valid_lambda_expr(Expression.fromstring(r"\b.ltzero(cmp_pos(ax_x,a,b))"), ctx_bound_vars=(Variable('a'),)), True)


def test_typecheck():
  ontology = _make_mock_ontology()

  def do_test(expr, extra_signature, expected):
    expr = Expression.fromstring(expr)

    if expected == None:
      assert_raises(l.TypeException, ontology.typecheck, expr, extra_signature)
    else:
      ontology.typecheck(expr, extra_signature)
      eq_(expr.type, expected)

  exprs = [
      (r"ltzero(cmp_pos(ax_x,unique(\x.sphere(x)),unique(\y.cube(y))))",
       {"x": ontology.types["obj"], "y": ontology.types["obj"]},
       ontology.types["boolean"]),

      (r"\a b.ltzero(cmp_pos(ax_x,a,b))",
       {"a": ontology.types["obj"], "b": ontology.types["obj"]},
       ontology.types["obj", "obj", "boolean"]),

      (r"\A b.and_(ltzero(b),A(b))",
       {"A": ontology.types[ontology.types.ANY_TYPE, "boolean"], "b": ontology.types["num"]},
       ontology.types[(ontology.types.ANY_TYPE, "boolean"), "num", "boolean"]),

      (r"\a.ltzero(cmp_pos,a,a,a)",
       {"a": ontology.types["obj"]},
       None),

      (r"and_(\x.ltzero(x),ltzero(one))",
       {},
       None),

      (r"and_(\x.ltzero(x),\y.ltzero(y))",
       {},
       None),

  ]

  for expr, extra_signature, expected in exprs:
    yield do_test, expr, extra_signature, expected


def test_infer_type():
  ontology = _make_mock_ontology()

  def do_test(expr, query_variable, expected_type):
    eq_(ontology.infer_type(Expression.fromstring(expr), query_variable), expected_type)

  cases = [
    (r"\a.sphere(a)", "a", ontology.types["obj"]),
    (r"\a.ltzero(cmp_pos(ax_x,a,a))", "a", ontology.types["obj"]),
    (r"\a b.ltzero(cmp_pos(ax_x,a,b))", "a", ontology.types["obj"]),
    (r"\a b.ltzero(cmp_pos(ax_x,a,b))", "b", ontology.types["obj"]),
    (r"\A b.and_(ltzero(b),A(b))", "A", ontology.types[ontology.types.ANY_TYPE, "boolean"]),
    (r"F00(one)", "F00", ontology.types["num", "*"]),
    (r"\F.unique(F(and_))", "F", ontology.types[("boolean", "boolean", "boolean"), ("obj", "boolean")]),
  ]

  for expr, query_variable, expected_type in cases:
    yield do_test, expr, query_variable, expected_type


def test_resolve_types():
  ontology = _make_mock_ontology()
  def do_test(type_set, expected_type):
    result_type = ontology.types.resolve_types(type_set)
    eq_(result_type, expected_type)

  cases = [
    ({ontology.types["?"], ontology.types["boolean"]}, ontology.types["boolean"]),
    ({ontology.types["?", "?"], ontology.types["?", "boolean"]}, ontology.types["?", "boolean"]),
    ({ontology.types["boolean", "boolean"], ontology.types["?", "?"]}, ontology.types["boolean", "boolean"]),
  ]

  for type_set, expected_type in cases:
    yield do_test, type_set, expected_type


def test_expression_bound():
  eq_(set(x.name for x in Expression.fromstring(r"\x.foo(x)").bound()),
      {"x"})
  eq_(set(x.name for x in Expression.fromstring(r"\x y.foo(x,y)").bound()),
      {"x", "y"})


def test_set_argument():
  e = Expression.fromstring(r"foo(a,b,c)")
  e.set_argument(2, Expression.fromstring("d"))
  eq_(str(e), r"foo(a,b,d)")


def test_set_argument0():
  e = Expression.fromstring(r"foo(a,b,c)")
  e.set_argument(0, Expression.fromstring("d"))
  eq_(str(e), r"foo(d,b,c)")


def test_unwrap_function():
  ontology = _make_mock_ontology()

  eq_(str(ontology.unwrap_function("sphere")), r"\z1.sphere(z1)")


def test_unwrap_base_functions():
  ontology = _make_mock_ontology()

  eq_(str(ontology.unwrap_base_functions(Expression.fromstring(r"unique(sphere)"))),
      r"unique(\z1.sphere(z1))")
  eq_(str(ontology.unwrap_base_functions(Expression.fromstring(r"cmp_pos(ax_x,unique(sphere),unique(cube))"))),
      r"cmp_pos(ax_x,unique(\z1.sphere(z1)),unique(\z1.cube(z1)))")


def test_iter_application_splits():
  # TODO test that, for every split, application on split parts yields a
  # subexpression
  ontology = _make_mock_ontology()
  cases = [
    (r"unique(\a.and_(cube(a),sphere(a)))",
      {(r"\z1.unique(\a.z1(cube,sphere,a))", r"\z1 z2 z3.and_(z1(z3),z2(z3))", "/"),},
      {
        # should not yield exprs which don't use their bound variables
        (r"\z1.unique(\a.and_(cube(a),sphere(a)))", None, None),
      }),

    (r"\z1.unique(z1(and_))",
      {},
      {
        # should not shadow existing bound variables
        (r"\z1 z1.z1(z1)", None, None),
      }),
  ]

  def do_test(expression, expected_members, expected_non_members):
    expr = Expression.fromstring(expression)
    splits = list(ontology.iter_application_splits(expr))
    split_tuples = [(str(part1), str(part2), dir) for part1, part2, dir in splits]
    from pprint import pprint
    pprint(split_tuples)

    all_parts = set(part1 for part1, _, _ in split_tuples) | set(part2 for _, part2, _ in split_tuples)

    for el in expected_members:
      if el[1] is None:
        # just want to assert that a logical expr appears *somewhere*
        ok_(el[0] in all_parts, el[0])
      else:
        # full split specified
        ok_(el in split_tuples, el)
    for el in expected_non_members:
      if el[1] is None:
        # just want to assert that a logical expr appears *nowhere*
        ok_(el[0] not in all_parts, el[0])
      else:
        # full split specified
        ok_(el not in split_tuples, el)

  for expr, expected, expected_not in cases:
    yield do_test, expr, expected, expected_not


def test_iter_application_splits_sound():
  """
  Every proposed split should, after application, yield a subexpression of the
  original expression.
  """

  ontology = _make_mock_ontology()
  cases = [
    r"unique(\a.and_(cube(a),sphere(a)))",
  ]

  def do_test(expression):
    expr = Expression.fromstring(expression)
    subexprs = [str(x) for x, _, _ in get_subexpressions(expr)]

    for part1, part2, dir in ontology.iter_application_splits(expr):
      arg1, arg2 = (part1, part2) if dir == "/" else (part2, part1)
      reapplied = str(ApplicationExpression(arg1, arg2).simplify())
      ok_(reapplied in subexprs, "%s %s %s --> %s" % (part1, dir, part2, reapplied))

  for expr in cases:
    yield do_test, expr
