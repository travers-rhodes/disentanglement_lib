# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for mig.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.evaluation.metrics import mig
import numpy as np
import gin.tf


def _identity_discretizer(target, num_bins):
  del num_bins
  return target


class MIGTest(absltest.TestCase):

  def test_metric(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = lambda x: x
    random_state = np.random.RandomState(0)
    scores = mig.compute_mig(
        ground_truth_data, representation_function, random_state, None, 3000)
    self.assertBetween(scores["discrete_mig"], 0.9, 1.0)

  def test_bad_metric(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    representation_function = np.zeros_like
    random_state = np.random.RandomState(0)
    scores = mig.compute_mig(
        ground_truth_data, representation_function, random_state, None, 3000)
    self.assertBetween(scores["discrete_mig"], 0.0, 0.2)

  def test_duplicated_latent_space(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    ground_truth_data = dummy_data.IdentityObservationsData()
    def representation_function(x):
      x = np.array(x, dtype=np.float64)
      return np.hstack([x, x])
    random_state = np.random.RandomState(0)
    scores = mig.compute_mig(
        ground_truth_data, representation_function, random_state, None, 3000)
    self.assertBetween(scores["discrete_mig"], 0.0, 0.1)

  # the representation here folds the dataset in on itself.
  # it looks like MIG counts this as a relatively good representation 
  def test_locally_good_metric(self):
    gin.bind_parameter("discretizer.discretizer_fn", _identity_discretizer)
    gin.bind_parameter("discretizer.num_bins", 10)
    # 3 factors each with 40 possible values (0-39)
    num_factors = 3
    num_possible_values = 40
    ground_truth_data = dummy_data.IdentityObservationsCustomSize(
        [num_possible_values] * num_factors)
    # fold the dataset on itself, so it's locally a good representation, but
    # globally not
    def representation_function(x):
      x = np.abs(x - num_possible_values/2, dtype=np.float64)
      return x 
    random_state = np.random.RandomState(0)
    scores = mig.compute_mig(
        ground_truth_data, representation_function, random_state, None, 3000)
    self.assertBetween(scores["discrete_mig"], 0.7, 1.0)

if __name__ == "__main__":
  absltest.main()
