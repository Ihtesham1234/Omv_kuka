from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import SingleArticulation
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.prims import create_fixed_joint, remove_fixed_joint
from omni.isaac.core.articulations import ArticulationAction
from pxr import Gf
import time

class ExampleScenario:
    def __init__(self):
        self._robot_path = "/World/kuka_kr210"
        self._cube_paths = [
            "/World/Cubes/Pickup_A",
            "/World/Cubes/Pickup_B",
            "/World/Cubes/Pickup_C",
            "/World/Cubes/Pickup_D",
        ]
        self._target_positions = [
            Gf.Vec3d(0.5, -0.3, 0.8),
            Gf.Vec3d(0.5, -0.1, 0.8),
            Gf.Vec3d(0.5,  0.1, 0.8),
            Gf.Vec3d(0.5,  0.3, 0.8),
        ]

        self._robot = SingleArticulation(self._robot_path)
        self._robot.initialize()

        self._cubes = [XFormPrim(path) for path in self._cube_paths]
        for cube in self._cubes:
            cube.initialize()

        self._stage = get_current_stage()

    def setup_scenario(self):
        print("Scenario setup complete.")

    def teardown_scenario(self):
        print("Scenario teardown complete.")

    def update_scenario(self):
        print("Starting pick-and-place operation...")
        for cube, target_pos in zip(self._cubes, self._target_positions):
            cube_pos = cube.get_world_pose()[0]
            print(f"Moving to pick cube at {cube_pos}")
            self._move_arm_to(cube_pos)

            print(f"Picking cube: {cube.prim_path}")
            self._pick_cube(cube)

            print(f"Moving to place cube at {target_pos}")
            self._move_arm_to(target_pos)

            print(f"Placing cube at {target_pos}")
            self._place_cube(cube, target_pos)

    def _move_arm_to(self, position):
        # Simulate movement with dummy joint positions
        joint_positions = [0.5] * self._robot.num_joints
        action = ArticulationAction(joint_positions=joint_positions)
        self._robot.apply_action(action)
        time.sleep(1.0)  # Simulate time delay for movement

    def _pick_cube(self, cube):
        robot_prim = self._stage.GetPrimAtPath(self._robot_path)
        cube_prim = self._stage.GetPrimAtPath(cube.prim_path)
        create_fixed_joint(robot_prim, cube_prim)

    def _place_cube(self, cube, position):
        remove_fixed_joint(cube.prim_path)
        cube.set_world_pose(position=position)
