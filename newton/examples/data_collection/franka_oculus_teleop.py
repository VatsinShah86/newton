###########################################################################
# Franka Oculus Teleop
#
# Franka Panda robot on a table, driven by Oculus Quest controller
# teleoperation via QuestStream. End-effector position and orientation
# deltas are mapped to joint velocities via the pseudoinverse Jacobian.
# Setting all deltas to zero holds the robot in position.
#
# Command: python newton/examples/data_collection/franka_oculus_teleop.py
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import ModelBuilder, State, eval_fk
from newton.math import transform_twist
from newton.solvers import SolverFeatherstone
from newton.examples.data_collection.oculus_teleop import QuestStream

GRIPPER_SPEED = 0.05   # m/s — full stroke (0 → 0.04 m) in ~0.8 s
GRIPPER_MAX   = 0.04   # m, fully open (Franka finger travel limit)
GRIPPER_MIN   = 0.0    # m, fully closed


class Example:
    def __init__(self, viewer, args=None, oculus_ip: str | None = None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.scene = ModelBuilder(gravity=-9.81)
        self.viewer = viewer

        # create robot
        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_world(franka)

        self.scene.add_ground_plane()
        self.model = self.scene.finalize(requires_grad=False)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.set_up_control()

        # gravity arrays for swapping during simulate()
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3)

        self.oculus = QuestStream(ip=oculus_ip) if oculus_ip is not None else None

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")
        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            scale=1.0,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # rest pose: arm up, slightly bent
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        self.endeffector_id = builder.body_count - 3
        # tool-center-point offset along the gripper z-axis [m]
        self.endeffector_offset = wp.transform(wp.vec3(0.0, 0.0, 0.22), wp.quat_identity())

    def set_up_control(self):
        n_dofs = self.model.joint_dof_count

        # target joint velocities; zero = hold position
        self.target_joint_qd = wp.zeros(n_dofs, dtype=float)

        # finger DOFs are the last two prismatic joints
        self._finger_dof_0 = n_dofs - 2
        self._finger_dof_1 = n_dofs - 1
        self.gripper_pos = GRIPPER_MAX  # start fully open

        # Jacobian buffers: 6 rows (linear + angular EE velocity) x n_dofs columns
        self._n_dofs = n_dofs
        self._J_flat = wp.empty(6 * n_dofs, dtype=float)
        self._body_out = wp.empty(6, dtype=float, requires_grad=True)
        self._temp_state = self.model.state(requires_grad=True)

        # one-hot gradient seeds for each of the 6 EE velocity components
        self._one_hots = [
            wp.array([1.0 if j == i else 0.0 for j in range(6)], dtype=float)
            for i in range(6)
        ]

        # kernel closed over the (fixed) end-effector body index and TCP offset
        ee_id = self.endeffector_id
        ee_offset = self.endeffector_offset

        @wp.kernel
        def compute_ee_body_out(
            body_qd: wp.array(dtype=wp.spatial_vector),
            body_out: wp.array(dtype=float),
        ):
            mv = transform_twist(wp.static(ee_offset), body_qd[wp.static(ee_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self._compute_ee_body_out = compute_ee_body_out

    def _compute_jacobian(self, state: State) -> np.ndarray:
        """Compute the 6 x n_dofs end-effector Jacobian via autodiff.

        Differentiates EE velocity (linear + angular) w.r.t. joint velocities.

        Args:
            state: Current simulation state.

        Returns:
            J: ndarray of shape (6, n_dofs).
        """
        joint_q = state.joint_q
        joint_qd = state.joint_qd
        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        tape = wp.Tape()
        with tape:
            eval_fk(self.model, joint_q, joint_qd, self._temp_state)
            wp.launch(
                self._compute_ee_body_out,
                dim=1,
                inputs=[self._temp_state.body_qd],
                outputs=[self._body_out],
            )

        for i in range(6):
            tape.backward(grads={self._body_out: self._one_hots[i]})
            wp.copy(self._J_flat[i * self._n_dofs : (i + 1) * self._n_dofs], joint_qd.grad)
            tape.zero()

        return self._J_flat.numpy().reshape(6, self._n_dofs)

    def apply_ee_delta(self, delta_pos: np.ndarray, delta_rot: np.ndarray):
        """Map an end-effector delta command to joint velocities.

        Uses the pseudoinverse Jacobian to convert desired EE displacement into
        joint velocities. Call with zero arrays to hold the current position.

        Args:
            delta_pos: Desired EE position displacement this frame [m], shape (3,).
            delta_rot: Desired EE rotation displacement this frame [rad], shape (3,).
        """
        ee_delta = np.concatenate([delta_pos, delta_rot]).astype(np.float32)

        if np.allclose(ee_delta, 0.0):
            self.target_joint_qd.zero_()
            return

        J = self._compute_jacobian(self.state_0)
        # divide by frame_dt so Featherstone's q += qd * dt integration
        # produces exactly the desired EE displacement
        joint_qd = (np.linalg.pinv(J) @ ee_delta / self.frame_dt).astype(np.float32)
        self.target_joint_qd.assign(joint_qd)

    def apply_gripper(self, gripper_cmd: float):
        """Move fingers toward open or closed at a fixed speed.

        Args:
            gripper_cmd: 1.0 = close, 0.0 = open.
        """
        if gripper_cmd > 0.5:
            target = max(GRIPPER_MIN, self.gripper_pos - GRIPPER_SPEED * self.frame_dt)
        else:
            target = min(GRIPPER_MAX, self.gripper_pos + GRIPPER_SPEED * self.frame_dt)

        finger_vel = float((target - self.gripper_pos) / self.frame_dt)
        self.gripper_pos = target

        # Override the finger DOFs — these must come after apply_ee_delta so the
        # pseudoinverse result for those columns is replaced.
        qd = self.target_joint_qd.numpy()
        qd[self._finger_dof_0] = finger_vel
        qd[self._finger_dof_1] = finger_vel
        self.target_joint_qd.assign(qd)

    def step(self):
        if self.oculus is not None:
            action = self.oculus.get_action()
            delta_pos, delta_rot, gripper_cmd = action[:3], action[3:6], action[6]
        else:
            delta_pos, delta_rot, gripper_cmd = np.zeros(3), np.zeros(3), 0.0
        self.apply_ee_delta(delta_pos, delta_rot)
        self.apply_gripper(gripper_cmd)
        self.simulate()
        self.sim_time += self.frame_dt

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.state_0.joint_qd.assign(self.target_joint_qd)

            # Disable gravity and contacts so Featherstone acts as a pure
            # kinematic integrator (integrates q from qd, no dynamics).
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.model.gravity.assign(self.gravity_earth)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    parser.add_argument("--oculus-ip", type=str, default=None, help="Quest IP address for teleoperation (omit to run without Oculus)")
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args, oculus_ip=args.oculus_ip)
    newton.examples.run(example, args)