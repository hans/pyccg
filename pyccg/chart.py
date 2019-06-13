from collections import OrderedDict
from copy import deepcopy
from functools import total_ordering
import itertools

from nltk.ccg import chart as nchart
from nltk.tree import Tree
import numpy as np

from pyccg import Token
from pyccg.combinator import *
from pyccg.logic import *


printCCGDerivation = nchart.printCCGDerivation


@total_ordering
class EdgeI(object):
  """
  A hypothesis about the structure of part of a sentence.
  Each edge records the fact that a structure is (partially)
  consistent with the sentence.  An edge contains:
  - A span, indicating what part of the sentence is
   consistent with the hypothesized structure.
  - A left-hand side, specifying what kind of structure is
   hypothesized.
  - A right-hand side, specifying the contents of the
   hypothesized structure.
  - A dot position, indicating how much of the hypothesized
   structure is consistent with the sentence.
  Every edge is either complete or incomplete:
  - An edge is complete if its structure is fully consistent
   with the sentence.
  - An edge is incomplete if its structure is partially
   consistent with the sentence.  For every incomplete edge, the
   span specifies a possible prefix for the edge's structure.
  There are two kinds of edge:
  - A ``TreeEdge`` records which trees have been found to
   be (partially) consistent with the text.
  - A ``LeafEdge`` records the tokens occurring in the text.
  The ``EdgeI`` interface provides a common interface to both types
  of edge, allowing chart parsers to treat them in a uniform manner.
  """

  def __init__(self):
    if self.__class__ == EdgeI:
      raise TypeError('Edge is an abstract interface')

  # ////////////////////////////////////////////////////////////
  # Span
  # ////////////////////////////////////////////////////////////

  def span(self):
    """
    Return a tuple ``(s, e)``, where ``tokens[s:e]`` is the
    portion of the sentence that is consistent with this
    edge's structure.
    :rtype: tuple(int, int)
    """
    raise NotImplementedError()

  def start(self):
    """
    Return the start index of this edge's span.
    :rtype: int
    """
    raise NotImplementedError()

  def end(self):
    """
    Return the end index of this edge's span.
    :rtype: int
    """
    raise NotImplementedError()

  def length(self):
    """
    Return the length of this edge's span.
    :rtype: int
    """
    raise NotImplementedError()

  # ////////////////////////////////////////////////////////////
  # Left Hand Side
  # ////////////////////////////////////////////////////////////

  def lhs(self):
    """
    Return this edge's left-hand side, which specifies what kind
    of structure is hypothesized by this edge.
    :see: ``TreeEdge`` and ``LeafEdge`` for a description of
      the left-hand side values for each edge type.
    """
    raise NotImplementedError()

  # ////////////////////////////////////////////////////////////
  # Right Hand Side
  # ////////////////////////////////////////////////////////////

  def rhs(self):
    """
    Return this edge's right-hand side, which specifies
    the content of the structure hypothesized by this edge.
    :see: ``TreeEdge`` and ``LeafEdge`` for a description of
      the right-hand side values for each edge type.
    """
    raise NotImplementedError()

  def dot(self):
    """
    Return this edge's dot position, which indicates how much of
    the hypothesized structure is consistent with the
    sentence.  In particular, ``self.rhs[:dot]`` is consistent
    with ``tokens[self.start():self.end()]``.
    :rtype: int
    """
    raise NotImplementedError()

  def nextsym(self):
    """
    Return the element of this edge's right-hand side that
    immediately follows its dot.
    :rtype: Nonterminal or terminal or None
    """
    raise NotImplementedError()

  def is_complete(self):
    """
    Return True if this edge's structure is fully consistent
    with the text.
    :rtype: bool
    """
    raise NotImplementedError()

  def is_incomplete(self):
    """
    Return True if this edge's structure is partially consistent
    with the text.
    :rtype: bool
    """
    raise NotImplementedError()

  # ////////////////////////////////////////////////////////////
  # Comparisons & hashing
  # ////////////////////////////////////////////////////////////

  def __eq__(self, other):
    return (
      self.__class__ is other.__class__
      and self._comparison_key == other._comparison_key
    )

  def __ne__(self, other):
    return not self == other

  def __lt__(self, other):
    if not isinstance(other, EdgeI):
      raise_unorderable_types("<", self, other)
    if self.__class__ is other.__class__:
      return self._comparison_key < other._comparison_key
    else:
      return self.__class__.__name__ < other.__class__.__name__

  def __hash__(self):
    try:
      return self._hash
    except AttributeError:
      self._hash = hash(self._comparison_key)
      return self._hash


# Based on the EdgeI class from NLTK.
# A number of the properties of the EdgeI interface don't
# transfer well to CCGs, however.
class CCGEdge(EdgeI):
  def __init__(self, span, categ, rule, semantics=None):
    self._span = span
    self._categ = categ
    self._rule = rule
    self._semantics = semantics
    self._comparison_key = (span, categ, rule, semantics)

  # Accessors
  def lhs(self): return self._categ
  def span(self): return self._span
  def start(self): return self._span[0]
  def end(self): return self._span[1]
  def length(self): return self._span[1] - self.span[0]
  def rhs(self): return ()
  def dot(self): return 0
  def is_complete(self): return True
  def is_incomplete(self): return False
  def nextsym(self): return None

  def categ(self): return self._categ
  def rule(self): return self._rule
  def semantics(self): return self._semantics

class CCGLeafEdge(EdgeI):
  '''
  Class representing leaf edges in a CCG derivation.
  '''
  def __init__(self, pos, token, leaf):
    self._pos = pos
    self._token = token
    self._leaf = leaf
    self._comparison_key = (pos, token.categ(), leaf)

  # Accessors
  def lhs(self): return self._token.categ()
  def span(self): return (self._pos, self._pos+1)
  def start(self): return self._pos
  def end(self): return self._pos + 1
  def length(self): return 1
  def rhs(self): return self._leaf
  def dot(self): return 0
  def is_complete(self): return True
  def is_incomplete(self): return False
  def nextsym(self): return None

  def token(self): return self._token
  def categ(self): return self._token.categ()
  def semantics(self): return self._token.semantics()
  def leaf(self): return self._leaf


########################################################################
##  Chart Rules
########################################################################


class ChartRuleI(object):
  """
  A rule that specifies what new edges are licensed by any given set
  of existing edges.  Each chart rule expects a fixed number of
  edges, as indicated by the class variable ``NUM_EDGES``.  In
  particular:
  - A chart rule with ``NUM_EDGES=0`` specifies what new edges are
   licensed, regardless of existing edges.
  - A chart rule with ``NUM_EDGES=1`` specifies what new edges are
   licensed by a single existing edge.
  - A chart rule with ``NUM_EDGES=2`` specifies what new edges are
   licensed by a pair of existing edges.
  :type NUM_EDGES: int
  :cvar NUM_EDGES: The number of existing edges that this rule uses
    to license new edges.  Typically, this number ranges from zero
    to two.
  """

  def apply(self, chart, grammar, *edges):
    """
    Return a generator that will add edges licensed by this rule
    and the given edges to the chart, one at a time.  Each
    time the generator is resumed, it will either add a new
    edge and yield that edge; or return.
    :type edges: list(EdgeI)
    :param edges: A set of existing edges.  The number of edges
      that should be passed to ``apply()`` is specified by the
      ``NUM_EDGES`` class variable.
    :rtype: iter(EdgeI)
    """
    raise NotImplementedError()

  def apply_everywhere(self, chart, grammar):
    """
    Return a generator that will add all edges licensed by
    this rule, given the edges that are currently in the
    chart, one at a time.  Each time the generator is resumed,
    it will either add a new edge and yield that edge; or return.
    :rtype: iter(EdgeI)
    """
    raise NotImplementedError()


class AbstractChartRule(ChartRuleI):
  """
  An abstract base class for chart rules.  ``AbstractChartRule``
  provides:
  - A default implementation for ``apply``.
  - A default implementation for ``apply_everywhere``,
   (Currently, this implementation assumes that ``NUM_EDGES``<=3.)
  - A default implementation for ``__str__``, which returns a
   name based on the rule's class name.
  """

  # Subclasses must define apply.
  def apply(self, chart, grammar, *edges):
    raise NotImplementedError()

  # Default: loop through the given number of edges, and call
  # self.apply() for each set of edges.
  def apply_everywhere(self, chart, grammar):
    if self.NUM_EDGES == 0:
      for new_edge in self.apply(chart, grammar):
        yield new_edge

    elif self.NUM_EDGES == 1:
      for e1 in chart:
        for new_edge in self.apply(chart, grammar, e1):
          yield new_edge

    elif self.NUM_EDGES == 2:
      for e1 in chart:
        for e2 in chart:
          for new_edge in self.apply(chart, grammar, e1, e2):
            yield new_edge

    elif self.NUM_EDGES == 3:
      for e1 in chart:
        for e2 in chart:
          for e3 in chart:
            for new_edge in self.apply(chart, grammar, e1, e2, e3):
              yield new_edge

    else:
      raise AssertionError('NUM_EDGES>3 is not currently supported')

  # Default: return a name based on the class name.
  def __str__(self):
    # Add spaces between InitialCapsWords.
    return re.sub('([a-z])([A-Z])', r'\1 \2', self.__class__.__name__)


class CCGChartRule(AbstractChartRule):

  def set_ontology(self, ontology):
    if hasattr(self, "_combinator"):
      self._combinator.set_ontology(ontology)


class BinaryCombinatorRule(CCGChartRule):
  '''
  Class implementing application of a binary combinator to a chart.
  Takes the directed combinator to apply.
  '''
  NUMEDGES = 2
  def __init__(self,combinator):
    self._combinator = combinator

  # Apply a combinator
  def apply(self, chart, grammar, left_edge, right_edge):
    # The left & right edges must be touching.
    if not (left_edge.end() == right_edge.start()):
      return

    # Check if the two edges are permitted to combine.
    # If so, generate the corresponding edge.
    can_combine = self._combinator.can_combine(left_edge, right_edge)
    if can_combine:#self._combinator.can_combine(left_edge, right_edge):
      for categ, semantics in self._combinator.combine(left_edge, right_edge):
        new_edge = CCGEdge(span=(left_edge.start(), right_edge.end()),
                 categ=categ, semantics=semantics,
                 rule=self._combinator)
        if chart.insert(new_edge,(left_edge,right_edge)):
          yield new_edge

  # The representation of the combinator (for printing derivations)
  def __str__(self):
    return "%s" % self._combinator

# Type-raising must be handled slightly differently to the other rules, as the
# resulting rules only span a single edge, rather than both edges.
class ForwardTypeRaiseRule(CCGChartRule):
  '''
  Class for applying forward type raising
  '''
  NUMEDGES = 2

  def __init__(self):
   self._combinator = ForwardT
  def apply(self, chart, grammar, left_edge, right_edge):
    if not (left_edge.end() == right_edge.start()):
      return

    for categ, semantics in self._combinator.combine(left_edge, right_edge):
      new_edge = CCGEdge(span=left_edge.span(), categ=categ, semantics=semantics,
               rule=self._combinator)
      if chart.insert(new_edge,(left_edge,)):
        yield new_edge

  def __str__(self):
    return "%s" % self._combinator

class BackwardTypeRaiseRule(CCGChartRule):
  '''
  Class for applying backward type raising.
  '''
  NUMEDGES = 2

  def __init__(self):
    self._combinator = BackwardT

  def apply(self, chart, grammar, left_edge, right_edge):
    if not (left_edge.end() == right_edge.start()):
      return

    for categ, semantics in self._combinator.combine(left_edge, right_edge):
      new_edge = CCGEdge(span=right_edge.span(), categ=categ, semantics=semantics,
               rule=self._combinator)
      if chart.insert(new_edge,(right_edge,)):
        yield new_edge

  def __str__(self):
    return "%s" % self._combinator


# Common sets of combinators used for English derivations.
ApplicationRuleSet = [BinaryCombinatorRule(ForwardApplication),
                      BinaryCombinatorRule(BackwardApplication)]
CompositionRuleSet = [BinaryCombinatorRule(ForwardComposition),
                      BinaryCombinatorRule(BackwardComposition),
                      BinaryCombinatorRule(BackwardBx)]
SubstitutionRuleSet = [BinaryCombinatorRule(ForwardSubstitution),
                       BinaryCombinatorRule(BackwardSx)]
TypeRaiseRuleSet = [ForwardTypeRaiseRule(), BackwardTypeRaiseRule()]

# The standard English rule set.
DefaultRuleSet = ApplicationRuleSet + CompositionRuleSet + \
    SubstitutionRuleSet + TypeRaiseRuleSet


########################################################################
##  Chart
########################################################################


class Chart(object):
  """
  A blackboard for hypotheses about the syntactic constituents of a
  sentence.  A chart contains a set of edges, and each edge encodes
  a single hypothesis about the structure of some portion of the
  sentence.
  The ``select`` method can be used to select a specific collection
  of edges.  For example ``chart.select(is_complete=True, start=0)``
  yields all complete edges whose start indices are 0.  To ensure
  the efficiency of these selection operations, ``Chart`` dynamically
  creates and maintains an index for each set of attributes that
  have been selected on.
  In order to reconstruct the trees that are represented by an edge,
  the chart associates each edge with a set of child pointer lists.
  A child pointer list is a list of the edges that license an
  edge's right-hand side.
  :ivar _tokens: The sentence that the chart covers.
  :ivar _num_leaves: The number of tokens.
  :ivar _edges: A list of the edges in the chart
  :ivar _edge_to_cpls: A dictionary mapping each edge to a set
    of child pointer lists that are associated with that edge.
  :ivar _indexes: A dictionary mapping tuples of edge attributes
    to indices, where each index maps the corresponding edge
    attribute values to lists of edges.
  """

  def __init__(self, tokens):
    """
    Construct a new chart. The chart is initialized with the
    leaf edges corresponding to the terminal leaves.
    :type tokens: list
    :param tokens: The sentence that this chart will be used to parse.
    """
    # Record the sentence token and the sentence length.
    self._tokens = tuple(tokens)
    self._num_leaves = len(self._tokens)

    # Initialise the chart.
    self.initialize()

  def initialize(self):
    """
    Clear the chart.
    """
    # A list of edges contained in this chart.
    self._edges = []

    # The set of child pointer lists associated with each edge.
    self._edge_to_cpls = {}

    # Indexes mapping attribute values to lists of edges
    # (used by select()).
    self._indexes = {}

  # ////////////////////////////////////////////////////////////
  # Sentence Access
  # ////////////////////////////////////////////////////////////

  def num_leaves(self):
    """
    Return the number of words in this chart's sentence.
    :rtype: int
    """
    return self._num_leaves

  def leaf(self, index):
    """
    Return the leaf value of the word at the given index.
    :rtype: str
    """
    return self._tokens[index]

  def leaves(self):
    """
    Return a list of the leaf values of each word in the
    chart's sentence.
    :rtype: list(str)
    """
    return self._tokens

  # ////////////////////////////////////////////////////////////
  # Edge access
  # ////////////////////////////////////////////////////////////

  def edges(self):
    """
    Return a list of all edges in this chart.  New edges
    that are added to the chart after the call to edges()
    will *not* be contained in this list.
    :rtype: list(EdgeI)
    :see: ``iteredges``, ``select``
    """
    return self._edges[:]

  def iteredges(self):
    """
    Return an iterator over the edges in this chart.  It is
    not guaranteed that new edges which are added to the
    chart before the iterator is exhausted will also be generated.
    :rtype: iter(EdgeI)
    :see: ``edges``, ``select``
    """
    return iter(self._edges)

  # Iterating over the chart yields its edges.
  __iter__ = iteredges

  def num_edges(self):
    """
    Return the number of edges contained in this chart.
    :rtype: int
    """
    return len(self._edge_to_cpls)

  def select(self, **restrictions):
    """
    Return an iterator over the edges in this chart.  Any
    new edges that are added to the chart before the iterator
    is exahusted will also be generated.  ``restrictions``
    can be used to restrict the set of edges that will be
    generated.
    :param span: Only generate edges ``e`` where ``e.span()==span``
    :param start: Only generate edges ``e`` where ``e.start()==start``
    :param end: Only generate edges ``e`` where ``e.end()==end``
    :param length: Only generate edges ``e`` where ``e.length()==length``
    :param lhs: Only generate edges ``e`` where ``e.lhs()==lhs``
    :param rhs: Only generate edges ``e`` where ``e.rhs()==rhs``
    :param nextsym: Only generate edges ``e`` where
      ``e.nextsym()==nextsym``
    :param dot: Only generate edges ``e`` where ``e.dot()==dot``
    :param is_complete: Only generate edges ``e`` where
      ``e.is_complete()==is_complete``
    :param is_incomplete: Only generate edges ``e`` where
      ``e.is_incomplete()==is_incomplete``
    :rtype: iter(EdgeI)
    """
    # If there are no restrictions, then return all edges.
    if restrictions == {}:
      return iter(self._edges)

    # Find the index corresponding to the given restrictions.
    restr_keys = sorted(restrictions.keys())
    restr_keys = tuple(restr_keys)

    # If it doesn't exist, then create it.
    if restr_keys not in self._indexes:
      self._add_index(restr_keys)

    vals = tuple(restrictions[key] for key in restr_keys)
    return iter(self._indexes[restr_keys].get(vals, []))

  def _add_index(self, restr_keys):
    """
    A helper function for ``select``, which creates a new index for
    a given set of attributes (aka restriction keys).
    """
    # Make sure it's a valid index.
    for key in restr_keys:
      if not hasattr(EdgeI, key):
        raise ValueError('Bad restriction: %s' % key)

    # Create the index.
    index = self._indexes[restr_keys] = {}

    # Add all existing edges to the index.
    for edge in self._edges:
      vals = tuple(getattr(edge, key)() for key in restr_keys)
      index.setdefault(vals, []).append(edge)

  def _register_with_indexes(self, edge):
    """
    A helper function for ``insert``, which registers the new
    edge with all existing indexes.
    """
    for (restr_keys, index) in self._indexes.items():
      vals = tuple(getattr(edge, key)() for key in restr_keys)
      index.setdefault(vals, []).append(edge)

  # ////////////////////////////////////////////////////////////
  # Edge Insertion
  # ////////////////////////////////////////////////////////////

  def insert_with_backpointer(self, new_edge, previous_edge, child_edge):
    """
    Add a new edge to the chart, using a pointer to the previous edge.
    """
    cpls = self.child_pointer_lists(previous_edge)
    new_cpls = [cpl + (child_edge,) for cpl in cpls]
    return self.insert(new_edge, *new_cpls)

  def insert(self, edge, *child_pointer_lists):
    """
    Add a new edge to the chart, and return True if this operation
    modified the chart.  In particular, return true iff the chart
    did not already contain ``edge``, or if it did not already associate
    ``child_pointer_lists`` with ``edge``.
    :type edge: EdgeI
    :param edge: The new edge
    :type child_pointer_lists: sequence of tuple(EdgeI)
    :param child_pointer_lists: A sequence of lists of the edges that
      were used to form this edge.  This list is used to reconstruct
      the trees (or partial trees) that are associated with ``edge``.
    :rtype: bool
    """
    # Is it a new edge?
    if edge not in self._edge_to_cpls:
      # Add it to the list of edges.
      self._append_edge(edge)
      # Register with indexes.
      self._register_with_indexes(edge)

    # Get the set of child pointer lists for this edge.
    cpls = self._edge_to_cpls.setdefault(edge, OrderedDict())
    chart_was_modified = False
    for child_pointer_list in child_pointer_lists:
      child_pointer_list = tuple(child_pointer_list)
      if child_pointer_list not in cpls:
        # It's a new CPL; register it, and return true.
        cpls[child_pointer_list] = True
        chart_was_modified = True
    return chart_was_modified

  def _append_edge(self, edge):
    self._edges.append(edge)

  # ////////////////////////////////////////////////////////////
  # Tree extraction & child pointer lists
  # ////////////////////////////////////////////////////////////

  def parses(self, root, tree_class=Tree):
    """
    Return an iterator of the complete tree structures that span
    the entire chart, and whose root node is ``root``.
    """
    for edge in self.select(start=0, end=self._num_leaves, lhs=root):
      for tree in self.trees(edge, tree_class=tree_class, complete=True):
        yield tree

  def trees(self, edge, tree_class=Tree, complete=False):
    """
    Return an iterator of the tree structures that are associated
    with ``edge``.
    If ``edge`` is incomplete, then the unexpanded children will be
    encoded as childless subtrees, whose node value is the
    corresponding terminal or nonterminal.
    :rtype: list(Tree)
    :note: If two trees share a common subtree, then the same
      Tree may be used to encode that subtree in
      both trees.  If you need to eliminate this subtree
      sharing, then create a deep copy of each tree.
    """
    return iter(self._trees(edge, complete, memo={}, tree_class=tree_class))

  def _trees(self, edge, complete, memo, tree_class):
    """
    A helper function for ``trees``.
    :param memo: A dictionary used to record the trees that we've
      generated for each edge, so that when we see an edge more
      than once, we can reuse the same trees.
    """
    # If we've seen this edge before, then reuse our old answer.
    if edge in memo:
      return memo[edge]

    # when we're reading trees off the chart, don't use incomplete edges
    if complete and edge.is_incomplete():
      return []

    # Leaf edges.
    if isinstance(edge, LeafEdge):
      leaf = self._tokens[edge.start()]
      memo[edge] = [leaf]
      return [leaf]

    # Until we're done computing the trees for edge, set
    # memo[edge] to be empty.  This has the effect of filtering
    # out any cyclic trees (i.e., trees that contain themselves as
    # descendants), because if we reach this edge via a cycle,
    # then it will appear that the edge doesn't generate any trees.
    memo[edge] = []
    trees = []
    lhs = edge.lhs().symbol()

    # Each child pointer list can be used to form trees.
    for cpl in self.child_pointer_lists(edge):
      # Get the set of child choices for each child pointer.
      # child_choices[i] is the set of choices for the tree's
      # ith child.
      child_choices = [self._trees(cp, complete, memo, tree_class) for cp in cpl]

      # For each combination of children, add a tree.
      for children in itertools.product(*child_choices):
        trees.append(tree_class(lhs, children))

  def child_pointer_lists(self, edge):
    """
    Return the set of child pointer lists for the given edge.
    Each child pointer list is a list of edges that have
    been used to form this edge.
    :rtype: list(list(EdgeI))
    """
    # Make a copy, in case they modify it.
    return self._edge_to_cpls.get(edge, {}).keys()

  # ////////////////////////////////////////////////////////////
  # Display
  # ////////////////////////////////////////////////////////////
  def pretty_format_edge(self, edge, width=None):
    """
    Return a pretty-printed string representation of a given edge
    in this chart.
    :rtype: str
    :param width: The number of characters allotted to each
      index in the sentence.
    """
    if width is None:
      width = 50 // (self.num_leaves() + 1)
    (start, end) = (edge.start(), edge.end())

    str = '|' + ('.' + ' ' * (width - 1)) * start

    # Zero-width edges are "#" if complete, ">" if incomplete
    if start == end:
      if edge.is_complete():
        str += '#'
      else:
        str += '>'

    # Spanning complete edges are "[===]"; Other edges are
    # "[---]" if complete, "[--->" if incomplete
    elif edge.is_complete() and edge.span() == (0, self._num_leaves):
      str += '[' + ('=' * width) * (end - start - 1) + '=' * (width - 1) + ']'
    elif edge.is_complete():
      str += '[' + ('-' * width) * (end - start - 1) + '-' * (width - 1) + ']'
    else:
      str += '[' + ('-' * width) * (end - start - 1) + '-' * (width - 1) + '>'

    str += (' ' * (width - 1) + '.') * (self._num_leaves - end)
    return str + '| %s' % edge

  def pretty_format_leaves(self, width=None):
    """
    Return a pretty-printed string representation of this
    chart's leaves.  This string can be used as a header
    for calls to ``pretty_format_edge``.
    """
    if width is None:
      width = 50 // (self.num_leaves() + 1)

    if self._tokens is not None and width > 1:
      header = '|.'
      for tok in self._tokens:
        header += tok[: width - 1].center(width - 1) + '.'
      header += '|'
    else:
      header = ''

    return header

  def pretty_format(self, width=None):
    """
    Return a pretty-printed string representation of this chart.
    :param width: The number of characters allotted to each
      index in the sentence.
    :rtype: str
    """
    if width is None:
      width = 50 // (self.num_leaves() + 1)
    # sort edges: primary key=length, secondary key=start index.
    # (and filter out the token edges)
    edges = sorted([(e.length(), e.start(), e) for e in self])
    edges = [e for (_, _, e) in edges]

    return (
      self.pretty_format_leaves(width)
      + '\n'
      + '\n'.join(self.pretty_format_edge(edge, width) for edge in edges)
    )

  # ////////////////////////////////////////////////////////////
  # Display: Dot (AT&T Graphviz)
  # ////////////////////////////////////////////////////////////

  def dot_digraph(self):
    # Header
    s = 'digraph nltk_chart {\n'
    # s += '  size="5,5";\n'
    s += '  rankdir=LR;\n'
    s += '  node [height=0.1,width=0.1];\n'
    s += '  node [style=filled, color="lightgray"];\n'

    # Set up the nodes
    for y in range(self.num_edges(), -1, -1):
      if y == 0:
        s += '  node [style=filled, color="black"];\n'
      for x in range(self.num_leaves() + 1):
        if y == 0 or (
          x <= self._edges[y - 1].start() or x >= self._edges[y - 1].end()
        ):
          s += '  %04d.%04d [label=""];\n' % (x, y)

    # Add a spacer
    s += '  x [style=invis]; x->0000.0000 [style=invis];\n'

    # Declare ranks.
    for x in range(self.num_leaves() + 1):
      s += '  {rank=same;'
      for y in range(self.num_edges() + 1):
        if y == 0 or (
          x <= self._edges[y - 1].start() or x >= self._edges[y - 1].end()
        ):
          s += ' %04d.%04d' % (x, y)
      s += '}\n'

    # Add the leaves
    s += '  edge [style=invis, weight=100];\n'
    s += '  node [shape=plaintext]\n'
    s += '  0000.0000'
    for x in range(self.num_leaves()):
      s += '->%s->%04d.0000' % (self.leaf(x), x + 1)
    s += ';\n\n'

    # Add the edges
    s += '  edge [style=solid, weight=1];\n'
    for y, edge in enumerate(self):
      for x in range(edge.start()):
        s += '  %04d.%04d -> %04d.%04d [style="invis"];\n' % (
          x,
          y + 1,
          x + 1,
          y + 1,
        )
      s += '  %04d.%04d -> %04d.%04d [label="%s"];\n' % (
        edge.start(),
        y + 1,
        edge.end(),
        y + 1,
        edge,
      )
      for x in range(edge.end(), self.num_leaves()):
        s += '  %04d.%04d -> %04d.%04d [style="invis"];\n' % (
          x,
          y + 1,
          x + 1,
          y + 1,
        )
    s += '}\n'
    return s

class CCGChart(Chart):
  def __init__(self, tokens):
    Chart.__init__(self, tokens)

  # Constructs the trees for a given parse. Unfortnunately, the parse trees need to be
  # constructed slightly differently to those in the default Chart class, so it has to
  # be reimplemented
  def _trees(self, edge, complete, memo, tree_class):
    assert complete, "CCGChart cannot build incomplete trees"

    if edge in memo:
      return memo[edge]

    if isinstance(edge,CCGLeafEdge):
      word = tree_class(edge.token(), [self._tokens[edge.start()]])
      leaf = tree_class((edge.token(), "Leaf"), [word])
      memo[edge] = [leaf]
      return [leaf]

    memo[edge] = []
    trees = []

    for cpl in self.child_pointer_lists(edge):
      child_choices = [self._trees(cp, complete, memo, tree_class)
                       for cp in cpl]
      for children in itertools.product(*child_choices):
        lhs = (Token(self._tokens[edge.start():edge.end()], edge.lhs(), edge.semantics()), str(edge.rule()))
        trees.append(tree_class(lhs, children))

    memo[edge] = trees
    return trees


class WeightedCCGChartParser(nchart.CCGChartParser):
  """
  CCG chart parser building off of the basic NLTK parser.

  Current extensions:

  1. Weighted inference (with weights on lexicon)
  2. Exhaustive search in cases where lexicon entries have ambiguous
  semantics. By default, NLTK ignores entries which have different
  semantics but share syntactic categories.
  """

  def __init__(self, lexicon, ruleset=None, *args, **kwargs):
    if ruleset is None:
      ruleset = ApplicationRuleSet

    if lexicon.ontology is not None:
      ruleset = deepcopy(ruleset)
      for rule in ruleset:
        rule.set_ontology(lexicon.ontology)

    super().__init__(lexicon, ruleset, *args, **kwargs)

  def _parse_inner(self, chart):
    """
    Run a chart parse on a chart with the edges already filled in.
    """

    # Select a span for the new edges
    for span in range(2,chart.num_leaves()+1):
      for start in range(0,chart.num_leaves()-span+1):
        # Try all possible pairs of edges that could generate
        # an edge for that span
        for part in range(1,span):
          lstart = start
          mid = start + part
          rend = start + span

          for left in chart.select(span=(lstart,mid)):
            for right in chart.select(span=(mid,rend)):
              # Generate all possible combinations of the two edges
              for rule in self._rules:
                edges_added_by_rule = 0
                for newedge in rule.apply(chart,self._lexicon,left,right):
                  edges_added_by_rule += 1

    # Attempt parses with the lexicon's start category as the root, or any
    # derived category which has the start category as base.
    parses = []
    for start_cat in self._lexicon.start_categories:
      parses.extend(chart.parses(start_cat))
    return parses

  def parse(self, tokens, return_aux=False):
    """
    Args:
      tokens: list of string tokens
      return_aux: return auxiliary information (`weights`, `valid_edges`)

    Returns:
      parses: list of CCG derivation results
      if return_aux, the list is actually a tuple with `parses` as its first
      element and the other following elements:
        weight: float parse weight
        edges: `tokens`-length list of the edge tokens used to generate this
          parse
    """
    tokens = list(tokens)
    lex = self._lexicon

    # Collect potential leaf edges for each index. May be multiple per
    # token.
    edge_cands = [[CCGLeafEdge(i, l_token, token) for l_token in lex.categories(token)]
                   for i, token in enumerate(tokens)]

    # Run a parse for each of the product of possible leaf nodes,
    # and merge results.
    results = []
    used_edges = []
    for edge_sequence in itertools.product(*edge_cands):
      chart = CCGChart(list(tokens))
      for leaf_edge in edge_sequence:
        chart.insert(leaf_edge, ())

      partial_results = list(self._parse_inner(chart))
      results.extend(partial_results)

      if return_aux:
        # Track which edge values were used to generate these parses.
        used_edges.extend([edge_sequence] * len(partial_results))

    # Score using Bayes' rule, calculated with lexicon weights.
    cat_priors = self._lexicon.observed_category_distribution()
    total_cat_masses = self._lexicon.total_category_masses()
    def score_parse(parse):
      score = 0.0
      for _, token in parse.pos():
        if total_cat_masses[token.categ()] == 0:
          return -np.inf
        # TODO not the same scoring logic as in novel word induction .. an
        # ideal Bayesian model would have these aligned !! (No smoothing here)
        likelihood = max(token.weight(), 1e-6) / total_cat_masses[token.categ()]
        logp = 0.5 * np.log(cat_priors[token.categ()])
        logp += np.log(likelihood)

        score += logp
      return score

    results = sorted(results, key=score_parse, reverse=True)
    if not return_aux:
      return results
    return [(parse, score_parse(parse), used_edges_i)
            for parse, used_edges_i in zip(results, used_edges)]


def get_clean_parse_tree(ccg_chart_result):
  """
  Get a clean parse tree representation of a CCG derivation, as returned by
  `CCGChartParser.parse`.
  """
  def traverse(node):
    if not isinstance(node, Tree):
      return

    label = node.label()
    if isinstance(label, tuple):
      token, op = label
      node.set_label(str(token.categ()))

    for i, child in enumerate(node):
      if len(child) == 1:
        new_preterminal = child[0]
        new_preterminal.set_label(str(new_preterminal.label().categ()))
        node[i] = new_preterminal
      else:
        traverse(child)

  ret = ccg_chart_result.copy(deep=True)
  traverse(ret)

  return ret
