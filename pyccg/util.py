from collections import Counter, defaultdict
from copy import copy
import heapq
import itertools
from queue import PriorityQueue

import numpy as np


class Distribution(Counter):
  """
  Weight distribution with discrete support.
  """

  @classmethod
  def uniform(cls, support):
    ret = cls()
    for key in support:
      ret[key] = 1 / len(support)
    return ret

  def __mul__(self, scale):
    assert isinstance(scale, (int, float))

    ret = Distribution()
    for key in self:
      ret[key] = self[key] * scale
    return ret

  def __add__(self, other):
    ret = copy(self)

    if isinstance(other, dict):
      for key, val in other.items():
        ret[key] += val
    else:
      for key in self:
        ret[key] += other

    return ret

  def __iadd__(self, other):
    return self + other

  def normalize(self):
    Z = sum(self.values())
    if Z > 0:
      return self * (1 / Z)
    return self

  def mix(self, other, alpha=0.5):
    assert alpha >= 0 and alpha <= 1
    return self * alpha + other * (1 - alpha)

  def argmax(self):
    return max(self, key=lambda k: self[k])


class ConditionalDistribution(object):

  def __init__(self):
    self.dists = defaultdict(Distribution)

  def __getitem__(self, key):
    return self.dists[key]

  def __setitem__(self, key, val):
    self.dists[key] = val

  def __iter__(self):
    return iter(self.dists)

  def __str__(self):
    return "{" + ", ".join("%s: %s" % (key, dist) for key, dist in self.dists.items()) + "}"

  __repr__ = __str__

  @property
  def support(self):
    return set(itertools.chain.from_iterable(
      dist.keys() for dist in self.dists.values()))

  @property
  def cond_support(self):
    return set(self.dists.keys())

  def ensure_cond_support(self, key):
    """
    Ensure that `key` is in the support of the conditioning factor.
    """
    return self.dists[key]

  def mix(self, other, alpha=0.5):
    # TODO assert that distributions are normalized
    assert 0 <= alpha and 1 >= alpha
    support = self.support
    assert support == other.support

    mixed = ConditionalDistribution()
    cond_support = self.cond_support
    other_cond_support = other.cond_support
    for key in cond_support | other_cond_support:
      if key in cond_support:
        self_dist = self[key]
      else:
        self_dist = Distribution.uniform(support)

      if key in other_cond_support:
        other_dist = other[key]
      else:
        other_dist = Distribution.uniform(support)

      mixed[key] = self_dist.mix(other_dist, alpha)

    return mixed

  def normalize_all(self):
    for key in self.dists.keys():
      self.dists[key] = self.dists[key].normalize()


class UniquePriorityQueue(PriorityQueue):
  def _init(self, maxsize):
    PriorityQueue._init(self, maxsize)
    self.values = set()

  def _put(self, item, heappush=heapq.heappush):
    if item[1] not in self.values:
      self.values.add(item[1])
      heappush(self.queue, item)

  def _get(self, heappop=heapq.heappop):
    item = heappop(self.queue)
    self.values.remove(item[1])
    return item

  def as_distribution(self):
    # NB critical region
    # NB assumes priorities are log-probabilities
    ret = Distribution()
    for priority, item in self.queue:
      val = np.exp(priority)
      ret[item] = val

    ret = ret.normalize()
    return ret
