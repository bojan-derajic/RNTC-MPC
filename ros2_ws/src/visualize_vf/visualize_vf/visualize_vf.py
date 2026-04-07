"""
Value-function visualiser for the RNTC-MPC framework.

This node computes and publishes a real-time image of the learned value function
in the robot-centred local frame.  The image encodes three regions:

    Pixel = 0   → collision zone      (SDF ≤ 0)
    Pixel = 120 → safe but constrained (SDF > 0 and VF = SDF + φ < 0)
    Pixel = 255 → safe and unconstrained (SDF > 0 and VF ≥ 0)

where VF = terminal_SDF + φ(grid; θ) is the value function evaluated over the
local XY grid, and φ is the main network parametrised by the RNTC hypernetwork.

Subscriptions:
    /local_obstacles  (Float64MultiArray) — obstacle positions and velocities
    /odometry         (Odometry)          — current robot state
    /predicted_path   (Path)              — predicted terminal heading (yaw)

Publication:
    /vf  (sensor_msgs/Image, mono8, 100×100) — value-function image
"""

import os
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import Image
from launch_ros.substitutions import FindPackageShare
from std_msgs.msg import Float64MultiArray

from visualize_vf.models import MainNetwork, Hypernetwork


class VisualizeVF(Node):
    """Publishes a local-frame value-function image from the RNTC hypernetwork."""

    # SDF grid parameters (must match mpc_local_planner / rntc_mpc settings)
    SDF_SIZE: float = 8.0    # Physical extent of the local costmap [m]
    SDF_RES: float = 0.08    # Grid resolution [m/cell]

    # Physical radii (must match mpc_local_planner settings)
    ROBOT_RADIUS: float = 0.4
    OBSTACLE_RADIUS: float = 0.5
    MIN_DISTANCE: float = ROBOT_RADIUS + OBSTACLE_RADIUS

    # Visualisation horizon used for the SDF channels (steps)
    VIS_HORIZON: int = 10
    DT: float = 0.1

    def __init__(self):
        super().__init__("visualize_vf")

        sdf_num_cells = int(self.SDF_SIZE / self.SDF_RES)

        # ---- Build neural network ----
        main_net_config = {
            "input_size": 3,
            "output_size": 1,
            "num_hidden_units": [36, 36, 36, 18, 18, 18, 9, 9, 9],
            "activation_function": "selu",
        }
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"PyTorch device: {self.torch_device}")

        self.main_net = MainNetwork(main_net_config).to(self.torch_device)
        self.hypernet = Hypernetwork(
            input_size=2, output_size=self.main_net.num_params()
        ).to(self.torch_device)

        # Load pre-trained RNTC hypernetwork weights
        pkg_share = FindPackageShare("mpc_local_planner").find("mpc_local_planner")
        hypernet_path = os.path.join(pkg_share, "hypernet_weights", "hypernet_rntc.pth")
        self.hypernet.load_state_dict(
            torch.load(hypernet_path, weights_only=True, map_location=self.torch_device)
        )
        self.hypernet.eval()
        self.main_net.eval()

        # ---- Pre-compute the evaluation grid ----
        axis = np.linspace(
            -(self.SDF_SIZE - self.SDF_RES) / 2,
            (self.SDF_SIZE - self.SDF_RES) / 2,
            sdf_num_cells,
        )
        self.xy_grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)

        # Full 3-D grid [x, y, yaw]; yaw channel is updated from /predicted_path
        grid_np = np.concatenate(
            [self.xy_grid, np.zeros((sdf_num_cells, sdf_num_cells, 1))], axis=-1
        )
        self.grid = torch.tensor(
            grid_np, dtype=torch.float, device=self.torch_device
        ).reshape(-1, 3)

        # Default robot state (overwritten by odometry callback)
        self.x_init = [0.0, 0.0, 0.0]

        # ---- Subscriptions ----
        self.predicted_path_sub = self.create_subscription(
            msg_type=Path,
            topic="/predicted_path",
            callback=self.predicted_path_callback,
            qos_profile=5,
        )
        self.local_obstacles_sub = self.create_subscription(
            msg_type=Float64MultiArray,
            topic="/local_obstacles",
            callback=self.local_obstacles_callback,
            qos_profile=5,
        )
        self.odometry_sub = self.create_subscription(
            msg_type=Odometry,
            topic="/odometry",
            callback=self.odometry_callback,
            qos_profile=5,
        )

        # ---- Publisher ----
        self.vf_pub = self.create_publisher(
            msg_type=Image,
            topic="/vf",
            qos_profile=5,
        )

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #

    def odometry_callback(self, msg: Odometry) -> None:
        """Update the current robot state."""
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        yaw = float(np.arctan2(w * z, 0.5 - z ** 2))
        self.x_init = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def predicted_path_callback(self, msg: Path) -> None:
        """Update the terminal yaw estimate from the last waypoint in the path."""
        if not msg.poses:
            return
        z = msg.poses[-1].pose.orientation.z
        w = msg.poses[-1].pose.orientation.w
        yaw = float(np.arctan2(w * z, 0.5 - z ** 2))
        # Broadcast the predicted terminal yaw over the entire evaluation grid
        self.grid[:, 2] = yaw

    def local_obstacles_callback(self, msg: Float64MultiArray) -> None:
        """Compute and publish the value-function image.

        1. Decode obstacle observations.
        2. Compute two-channel SDF in the local frame.
        3. Run the hypernetwork to get main-network weights.
        4. Evaluate the main network over the grid.
        5. Compute VF = SDF + main-network output.
        6. Encode as a mono8 image and publish.
        """
        obs_shape = (msg.layout.dim[0].size, msg.layout.dim[1].size)
        observation = np.array(msg.data).reshape(obs_shape)
        positions = observation[:, :2]
        velocities = observation[:, 2:]
        robot_pos = np.array(self.x_init[:2])

        # Linearly propagate obstacles over the visualisation horizon
        predictions = np.stack(
            [
                positions + velocities * t
                for t in np.arange(0, (self.VIS_HORIZON + 1) * self.DT, self.DT)
            ],
            axis=0,
        )
        predictions[np.isnan(predictions)] = 100.0  # Place absent obstacles far away

        num_obstacles = int(np.count_nonzero(~np.isnan(positions[:, 0])))

        # Two-channel SDF: terminal step and an intermediate step
        large_val = self.SDF_SIZE
        terminal_sdf = [np.full(self.xy_grid.shape[:2], large_val)]
        stage_sdf = [np.full(self.xy_grid.shape[:2], large_val)]

        for i in range(num_obstacles):
            terminal_local = predictions[-1, i, :] - robot_pos
            stage_local = predictions[-5, i, :] - robot_pos
            terminal_sdf.append(
                np.linalg.norm(self.xy_grid - terminal_local, axis=-1) - self.MIN_DISTANCE
            )
            stage_sdf.append(
                np.linalg.norm(self.xy_grid - stage_local, axis=-1) - self.MIN_DISTANCE
            )

        terminal_sdf = np.min(np.stack(terminal_sdf, axis=-1), axis=-1)
        stage_sdf = np.min(np.stack(stage_sdf, axis=-1), axis=-1)

        # Run hypernetwork → main-network weights → value function
        sdf_seq = np.stack([terminal_sdf, stage_sdf], axis=0)
        sdf_tensor = torch.tensor(
            sdf_seq[None, ...], dtype=torch.float, device=self.torch_device
        )
        with torch.no_grad():
            main_net_params = self.hypernet(sdf_tensor)
            self.main_net.set_params(main_net_params)
            residual = self.main_net(self.grid).reshape(
                self.xy_grid.shape[0], self.xy_grid.shape[1]
            ).cpu().numpy()

        vf = terminal_sdf + residual

        # Encode into uint8: three distinct grey levels
        img = np.empty_like(vf, dtype=np.uint8)
        img[terminal_sdf <= 0.0] = 0     # Collision zone
        safe = terminal_sdf > 0.0
        img[safe & (vf < 0.0)] = 120     # Safe but constrained by the value function
        img[safe & (vf >= 0.0)] = 255    # Freely safe

        # Rotate 180° so the image is oriented with the robot facing right
        img = np.rot90(img, k=2)

        # Publish
        image_msg = Image()
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = "base_link"
        image_msg.height = self.xy_grid.shape[0]
        image_msg.width = self.xy_grid.shape[1]
        image_msg.encoding = "mono8"
        image_msg.step = self.xy_grid.shape[1]   # bytes per row = width for mono8
        image_msg.data = img.tobytes()
        self.vf_pub.publish(image_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizeVF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
