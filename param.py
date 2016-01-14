from __future__ import division, print_function

class Param:
    """Model parameter."""

    def __init__(self, val, minval=-1e99, maxval=1e99, frozen=False):
        val = float(val)
        self.val = val
        self.defval = val
        self.minval = minval
        self.maxval = maxval
        self.frozen = frozen

    def __repr__(self):
        return '<Param: val=%.2g, minval=%.2g, maxval=%.2g, frozen=%s>' % (
            self.val, self.minval, self.maxval, self.frozen)
