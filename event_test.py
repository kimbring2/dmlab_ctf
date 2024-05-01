# Copyright 2016-2018 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""Basic test for DeepMind Lab Python wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import absltest
import numpy as np
import six

import deepmind_lab

env = deepmind_lab.Lab('tests/event_test', [], config={'height': '32', 'width': '32'})
env.reset(episode=1, seed=7)
events = env.events()
print("events 1 : ", events)
#self.assertLen(events, 4)

name, obs = events[0]
name, obs = events[1]
name, obs = events[2]
name, obs = events[3]

action = np.zeros((7,), dtype=np.intc)
env.step(action, num_steps=1)
print("events 2: ", env.events())

env.step(action, num_steps=58)
print("events 3: ", env.events())

env.step(action, num_steps=1)
print("events 4: ", env.events())

print("env.is_running(): ", env.is_running())
#self.assertFalse(env.is_running())

events = env.events()
print("events 5: ", events)

env.reset(episode=2, seed=8)
events = env.events()
print("events 6 : ", events)