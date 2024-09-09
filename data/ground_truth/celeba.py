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

"""Shapes3D data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from data.ground_truth import ground_truth_data
from data.ground_truth import util
import numpy as np
from torchvision.datasets.celeba import CelebA
from six.moves import range
# import tensorflow as tf
import h5py

# wouldn't clash with the images unpacked via prepare_celeba.py
CELEBA_DIR = os.path.join("datasets", 'celeba') 

class MyCelebA(ground_truth_data.GroundTruthData):
    """CelebA dataset following GroundTruthData abstract class.
    """

    def __init__(self):
        # since this is only called for evaluation, can use only test split 
        # for faster runtime
        celeba = CelebA(CELEBA_DIR, 
                        split='test', 
                        target_type='attr',
                        download=True)

        images = np.stack([img.numpy() for img, _ in celeba], axis=0)
        features = np.stack([attr.numpy() for _, attr in celeba], axis=0)
        n_samples = 19_962
        self.images = (images.reshape([n_samples, 224, 224, 3]).astype(np.float32) / 255.)
        assert np.shape(features) == [n_samples, 40]
        self.factor_sizes = [2 for _ in range(40)]
        self.latent_factor_indices = list(range(40))
        self.num_total_factors = features.shape[1]
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(
            self.factor_sizes)

    @property
    def num_factors(self):
        return self.state_space.num_latent_factors

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return [224, 224, 3]

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        all_factors = self.state_space.sample_all_factors(factors, random_state)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices]
