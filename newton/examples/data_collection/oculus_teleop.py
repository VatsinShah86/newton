import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third-party" / "oculus_reader"))

import numpy as np
from scipy.spatial.transform import Rotation
from oculus_reader.reader import OculusReader


def _eprint(*args):
    sys.stderr.write("\033[1;31m")
    print(*args, file=sys.stderr)
    sys.stderr.write("\033[0;0m")


class PatchedOculusReader(OculusReader):
    """OculusReader with upstream bug fix: get_network_device() was incorrectly
    recursing into get_device() (which takes no arguments) instead of itself."""

    def get_network_device(self, client, retry=0):
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError:
            os.system('adb devices')
            client.remote_connect(self.ip_address, self.port)
        device = client.device(self.ip_address + ':' + str(self.port))

        if device is None:
            if retry == 1:
                os.system('adb tcpip ' + str(self.port))
            if retry == 2:
                _eprint('Make sure that device is running and is available at the IP address specified as the OculusReader argument `ip_address`.')
                _eprint('Currently provided IP address:', self.ip_address)
                _eprint('Run `adb shell ip route` to verify the IP address.')
                exit(1)
            else:
                self.get_network_device(client=client, retry=retry + 1)
        return device


class QuestStream:
    def __init__(self, ip: str, pose_scaler=(1.0, 0.85), channel_signs=(1,1,1, -1,-1,-1)):
        self.reader = PatchedOculusReader(ip_address=ip)
        self.pos_scale = pose_scaler[0]
        self.rot_scale = pose_scaler[1]
        self.signs = np.array(channel_signs)
        self._prev_pos = None
        self._prev_rot = None

    def get_action(self):
        transforms, buttons = self.reader.get_transformations_and_buttons()
        if transforms is None or 'r' not in transforms:
            return np.zeros(7)

        T = transforms['r']           # 4x4 right controller pose
        pos = T[:3, 3]
        rot = Rotation.from_matrix(T[:3, :3])

        enabled = buttons.get('rightGrip', (0,))[0] > 0.5
        gripper  = float(buttons.get('rightTrig', (0,))[0] > 0.5)

        if self._prev_pos is None or not enabled:
            self._prev_pos = pos.copy()
            self._prev_rot = rot
            return np.zeros(7)

        oculus_delta_pos = (pos - self._prev_pos) * self.pos_scale
        oculus_delta_rotvec = (self._prev_rot.inv() * rot).as_rotvec() * self.rot_scale

        self._prev_pos = pos.copy()
        self._prev_rot = rot

        # Oculus -> Robot coordinate frame alignment
        T_OCULUS_TO_ROBOT = np.array([
            [ 0.,  0., -1.],
            [-1.,  0.,  0.],
            [ 0.,  1.,  0.],
        ])
        delta_pos = T_OCULUS_TO_ROBOT @ oculus_delta_pos
        delta_rot = np.array([
            oculus_delta_rotvec[1],   # robot roll  = oculus rz
            oculus_delta_rotvec[0],   # robot pitch = oculus rx
            oculus_delta_rotvec[2],   # robot yaw   = oculus ry
        ])

        action = np.append(delta_pos, delta_rot)
        action = action * self.signs
        return np.append(action, gripper)

    def stop(self):
        print("Stopping")
        self.reader.stop()

if __name__ == "__main__":
    import csv
    import time
    from pathlib import Path

    CSV_PATH = Path(__file__).parents[3] / "teleop.csv"
    FPS = 10
    dt = 1.0 / FPS

    oculus = QuestStream(ip='10.155.158.226')

    # Overwrite the CSV fresh on every run
    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "time_s",
            "delta_pos_x", "delta_pos_y", "delta_pos_z",
            "delta_rot_x", "delta_rot_y", "delta_rot_z",
            "gripper",
        ])

        t_start = time.monotonic()
        print(f"Recording to {CSV_PATH}  (Ctrl-C to stop)")

        while True:
            try:
                t0 = time.monotonic()
                action = oculus.get_action()
                delta_pos, delta_rot, gripper = action[:3], action[3:6], action[6]

                t_elapsed = t0 - t_start
                writer.writerow([
                    f"{t_elapsed:.3f}",
                    *[f"{v:.7f}" for v in delta_pos],
                    *[f"{v:.7f}" for v in delta_rot],
                    f"{gripper:.1f}",
                ])
                csv_file.flush()

                elapsed = time.monotonic() - t0
                time.sleep(max(0.0, dt - elapsed))
            except KeyboardInterrupt:
                oculus.stop()
                print(f"\nSaved {CSV_PATH}")
                break