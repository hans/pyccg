from frozendict import frozendict
import operator


class Event(object):

  def __init__(self):
    pass

  def __eq__(self, other):
    if isinstance(other, Event):
      return True
    return False

  def __hash__(self):
    return hash(Event)

  def __getitem__(self, attr):
    return EventOp(self, getattr, attr)

  def __getattr__(self, attr):
    if attr.startswith("__"):
      # Avoid returning EventOps when client is trying to access a dunder
      # method!
      raise AttributeError
    return EventOp(self, getattr, attr)

  def __call__(self):
    # Dummy method which allows us to use an instance of this class as a
    # function in the ontology.
    return None

  def __str__(self):
    return "<event>"

  __repr__ = __str__


class EventOp(object):
  """
  Lazy-evaluated operation on an event object.
  """

  def __init__(self, base, op, *args):
    self.base = base
    self.op = op
    self.args = tuple(args)

  def __hash__(self):
    return hash((self.base, self.op, self.args))

  def __eq__(self, other):
    """
    Compares two `EventOp` instances. To do lazy equality checks, use
    `EventOp.equals`.
    """
    return hash(self) == hash(other)

  def equals(self, other):
    """
    Builds a lazy equality check op. To compare `EventOp` instances, use `==`.
    """
    return EventOp(self, operator.eq, other)

  def __getitem__(self, attr):
    return EventOp(self, getattr, attr)

  def __getattr__(self, attr):
    if attr.startswith("__"):
      # Avoid returning EventOps when client is trying to access a dunder
      # method!
      raise AttributeError
    return EventOp(self, getattr, attr)

  def __add__(self, other):
    return EventOp(self, operator.add, other)

  def __sub__(self, other):
    return EventOp(self, operator.sub, other)

  def __mul__(self, other):
    return EventOp(self, operator.mul, other)

  def __rmul__(self, other):
    return EventOp(self, operator.mul, other)

  def __lt__(self, other):
    return EventOp(self, operator.lt, other)

  def __gt__(self, other):
    return EventOp(self, operator.gt, other)

  def __contains__(self, el):
    return EventOp(self, operator.contains, el)

  def __call__(self, *args, **kwargs):
    return EventOp(self, operator.methodcaller, (*args, frozendict(kwargs)))

  def __str__(self, verbose=False):
    if verbose:
      op_str = repr(self.op)
    else:
      if hasattr(self.op, "__name__"):
        op_str = self.op.__name__
      elif hasattr(self.op, "__call__"):
        op_str = self.op.__class__.__name__
      else:
        op_str = str(self.op)
    return "EventOp<%s>(%s, %s)" % \
        (op_str, self.base, ", ".join(str(arg) for arg in self.args))

  def __repr__(self):
    return self.__str__(verbose=True)


class Object(object):

  def __init__(self, name=None, **attrs):
    self.attrs = frozendict(attrs)
    self.name = name or self.attrs.get("type")

  def __hash__(self):
    return hash((self.name, self.attrs))

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __str__(self):
    return self.name

  def __repr__(self):
    return "O(%s: %s)" % (self.name, self.attrs)

  def __getattr__(self, attr):
    if attr.startswith("__"):
      # Don't muck with dunder methods
      raise AttributeError
    return self[attr]

  def __getitem__(self, attr):
    return self.attrs[attr]


class Collection(object):

  def __init__(self, characteristic):
    self.characteristic = characteristic

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __hash__(self):
    # TODO not sure about the semantics!
    return hash(self.characteristic)


def fn_unique(xs):
  true_xs = [x for x, matches in xs.items() if matches]
  assert len(true_xs) == 1
  return true_xs[0]

def fn_cmp_pos(ax, manner, a, b):
  sign = 1 if manner == "pos" else -1
  return sign * (a["3d_coords"][ax()] - b["3d_coords"][ax()])

def fn_ltzero(x): return x < 0
def fn_and(a, b): return a and b
def fn_eq(a, b):
  if hasattr(a, "equals"):
    return a.equals(b)
  elif hasattr(b, "equals"):
    return b.equals(a)
  else:
    return a == b

## Ops on collections
def fn_set(a): return isinstance(a, Collection)
def fn_characteristic(a): return a.characteristic

def fn_ax_x(): return 0
def fn_ax_y(): return 1
def fn_ax_z(): return 2

## Ops on objects
def fn_cube(x): return x.shape == "cube"
def fn_sphere(x): return x.shape == "sphere"
def fn_donut(x): return x.shape == "donut"
def fn_pyramid(x): return x.shape == "pyramid"
def fn_hose(x): return x.shape == "hose"
def fn_cylinder(x): return x.shape == "cylinder"
def fn_apple(x): return x.type == "apple"
def fn_cookie(x): return x.type == "cookie"
def fn_book(x): return x.type == "book"
def fn_water(x): return x.type == "water"

def fn_object(x): return isinstance(x, (frozendict, dict))
def fn_vertical(x): return x.orientation == "vertical"
def fn_horizontal(x): return x.orientation == "horizontal"
def fn_liquid(x): return x.state.equals("liquid")
def fn_full(x): return x.full

# Two-place ops on objects
def fn_contain(x, y):
  if isinstance(x, (Event, EventOp)) or isinstance(y, (Event, EventOp)):
    return x.contain(y)
  return x in y
def fn_contact(x, y):
  if isinstance(x, (Event, EventOp)):
    return x.contact(y)
  elif isinstance(y, (Event, EventOp)):
    return y.contact(x)
  # TODO implement the actual op rather than the lazy comp representation :)
  return True

## Ops on events

class Action(object):
  def __add__(self, other):
    return ComposedAction(self, other)

  def __eq__(self, other):
    return isinstance(other, self.__class__) and hash(self) == hash(other)


class Constraint(object):
  # TODO semantics not right -- subclasses don't take multiple constraints. We
  # should have a separate `ComposedConstraint` class
  def __init__(self, *constraints):
    constraints_flat = []
    for constraint in constraints:
      if constraint.__class__ == Constraint:
        # This is a composite constraint instance -- merge its containing
        # constraints.
        constraints_flat.extend(constraint.constraints)
      else:
        constraints_flat.append(constraint)
    self.constraints = frozenset(constraints_flat)

  def __add__(self, other):
    return Constraint(self.constraints | other.constraints)

  def __hash__(self):
    return hash(self.constraints)

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __str__(self):
    return "Constraint(%s)" % (", ".join(map(str, self.constraints)))

  __repr__ = __str__

class Contain(Constraint):
  def __init__(self, container, obj):
    self.container = container
    self.obj = obj

  def __hash__(self):
    return hash((self.container, self.obj))

  def __str__(self):
    return "%s(%s in %s)" % (self.__class__.__name__, self.obj, self.container)

class Contact(Constraint):
  def __init__(self, *objects):
    self.objects = frozenset(objects)

  def __hash__(self):
    return hash((self.objects))

  def __str__(self):
    return "%s(%s)" % (self.__class__.__name__, ",".join(map(str, self.objects)))

class ComposedAction(Action):
  def __init__(self, *actions):
    self.actions = actions

  def __hash__(self):
    return hash(tuple(self.actions))

  def __str__(self):
    return "+(%s)" % (",".join(str(action) for action in self.actions))

  __repr__ = __str__

class Move(Action):
  def __init__(self, obj, dest, manner):
    self.obj = obj
    self.dest = dest
    self.manner = manner

  def __hash__(self):
    return hash((self.obj, self.dest, self.manner))

  def __eq__(self, other):
    return hash(self) == hash(other)

  def __str__(self):
    return "%s(%s -> %s, %s)" % (self.__class__.__name__, self.obj, self.dest, self.manner)

  __repr__ = __str__

class Transfer(Move):
  pass


class Put(Action):
  def __init__(self, event, obj, manner):
    self.event = event
    self.obj = obj
    self.manner = manner

  def __hash__(self):
    return hash((self.event, self.obj, self.manner))

  def __str__(self):
    return "%s(%s,%s,%s)" % (self.__class__.__name__, self.event, self.obj, self.manner)

  __repr__ = __str__


class Eat(Action):
  def __init__(self, event, food):
    self.event = event
    self.food = food

  def __hash__(self):
    return hash((self.event, self.food))

  def __str__(self):
    return "%s(%s)" % (self.__class__.__name__, self.food)


class ActAndEntail(Action):
  """
  Joins an action with entailments about the event.
  """
  def __init__(self, action, entail):
    self.action = action
    self.entail = entail

  def __hash__(self):
    return hash((self.action, self.entail))


class StateChange(Action): pass
class CausePossession(StateChange):
  def __init__(self, agent, obj):
    self.agent = agent
    self.obj = obj

  def __hash__(self):
    return hash((self.agent, self.obj))

  def __str__(self):
    return "%s(%s <- %s)" % (self.__class__.__name__, self.agent, self.obj)

  __repr__ = __str__
