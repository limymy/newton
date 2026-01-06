# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np
import warp as wp

from newton import JointType
from newton._src.solvers.mujoco.kernels import convert_mj_acc_to_warp_kernel, convert_mj_body_cacc_to_warp_kernel


class TestMuJoCoAccelerationKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._cache_dir = tempfile.mkdtemp(prefix="warp_cache_", dir=os.path.dirname(__file__))
        wp.config.kernel_cache_dir = cls._cache_dir
        wp.init()
        cls.device = wp.get_device("cpu")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._cache_dir, ignore_errors=True)

    def setUp(self):
        self.device = self.__class__.device

    def test_cacc_to_body_qdd_mapping_and_swap(self):
        # mjc_body_to_newton maps [world, mjc_body] -> newton_body
        mjc_body_to_newton = wp.array([[-1, 0, 2]], dtype=wp.int32, device=self.device)

        # MuJoCo cacc convention: (rot, lin)
        cacc_np = np.zeros((1, 3, 6), dtype=np.float32)
        cacc_np[0, 1, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        cacc_np[0, 2, :] = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], dtype=np.float32)
        cacc = wp.array(cacc_np, dtype=wp.spatial_vector, device=self.device)

        body_qdd = wp.zeros(3, dtype=wp.spatial_vector, device=self.device)

        wp.launch(
            convert_mj_body_cacc_to_warp_kernel,
            dim=(1, 3),
            inputs=[mjc_body_to_newton, cacc],
            outputs=[body_qdd],
            device=self.device,
        )

        out = body_qdd.numpy()
        np.testing.assert_allclose(out[0], np.array([4.0, 5.0, 6.0, 1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(out[1], np.zeros(6, dtype=np.float32))
        np.testing.assert_allclose(out[2], np.array([10.0, 11.0, 12.0, 7.0, 8.0, 9.0], dtype=np.float32))

    def test_qacc_to_joint_qdd_writes_expected_values(self):
        joints_per_world = 1

        qpos = wp.array([[0.0]], dtype=wp.float32, device=self.device)
        qacc = wp.array([[3.25]], dtype=wp.float32, device=self.device)

        joint_type = wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=self.device)
        joint_q_start = wp.array([0], dtype=wp.int32, device=self.device)
        joint_qd_start = wp.array([0], dtype=wp.int32, device=self.device)
        joint_dof_dim = wp.array([[0, 1]], dtype=wp.int32, device=self.device)

        joint_qdd = wp.zeros(1, dtype=wp.float32, device=self.device)

        wp.launch(
            convert_mj_acc_to_warp_kernel,
            dim=(1, joints_per_world),
            inputs=[
                qpos,
                qacc,
                joints_per_world,
                0,
                joint_type,
                joint_q_start,
                joint_qd_start,
                joint_dof_dim,
            ],
            outputs=[joint_qdd],
            device=self.device,
        )

        np.testing.assert_allclose(joint_qdd.numpy(), np.array([3.25], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
