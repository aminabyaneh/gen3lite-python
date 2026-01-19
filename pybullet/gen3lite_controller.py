import pybullet as pb
import time
import math
from typing import Dict, List
from enum import Enum
import numpy as np


class ControlModes(Enum):
    """
    Pybullet control modes, only for the end effector. We use IK to translate
    EF commands to joint vellocities using a PID controller.
    """

    END_EFFECTOR_POSE = pb.POSITION_CONTROL
    END_EFFECTOR_TWIST = pb.VELOCITY_CONTROL


class Gen3LiteArmController:
    """
    A controller for the Kinova Gen3 Lite robotic arm in PyBullet.

    This class provides methods to control the arm's joints, move the end-effector
    to desired positions and orientations using inverse kinematics, and operate the gripper.
    """
    def __init__(self, dt=1/50.0):
        self.dt = dt

        self.LEFT_FINGER_JOINT = 7    # Example index; update if needed.
        self.RIGHT_FINGER_JOINT = 9   # Example index; update if needed.
        self.GRIPPER_OPEN_POS = 0.7   # Adjust as needed.
        self.GRIPPER_CLOSED_POS = 0.0 # Adjust as needed.

        # End-effector link index as used in your URDF.
        self.END_EFFECTOR_INDEX = 7

        # Load the Kinova Gen3 Lite URDF model.
        self.__kinova_id = pb.loadURDF("gen3lite_urdf/gen3_lite.urdf", [0, 0, 0], useFixedBase=True) # [0, 0, 0.7]
        pb.resetBasePositionAndOrientation(self.__kinova_id, [0, 0, 0.0], [0, 0, 0, 1]) # [0, 0, 0.0]

        self.__n_joints = 7 # pb.getNumJoints(self.__kinova_id) - 5, where -5 for the gripper
        print(f'Found {self.__n_joints} active joints for the robot.')

        # Joint limits and rest/home poses.
        self.__lower_limits: List = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        self.__upper_limits: List = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        self.__joint_ranges: List = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        self.__rest_poses: List = [0, 0, 0, 0, 0, 0, 0]
        self.__home_poses: List = [0, 0, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, -math.pi * 0.5, 0]

        self.joint_ids = [pb.getJointInfo(self.__kinova_id, i) for i in range(self.__n_joints)]
        self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pb.JOINT_REVOLUTE]

        # Initialize to rest position.
        for i in range(self.__n_joints):
            pb.resetJointState(self.__kinova_id, i, self.__rest_poses[i])

        self.default_ori = list(pb.getQuaternionFromEuler([0, -math.pi, 0]))

    def set_to_home(self):
        """
        Resets the arm to its predefined home position.
        """

        for i in range(self.__n_joints):
            pb.resetJointState(self.__kinova_id, i, self.__home_poses[i])

    def move_to_joint_positions(self, joints, max_steps=100):
        """
        Move to target joint positions with position control.

        Args:
            joints (list): Target joint positions.
            max_steps (int): Maximum simulation steps to reach the target.
        """
        for i in range(self.__n_joints):
            pb.setJointMotorControl2(
                bodyIndex=self.__kinova_id,
                jointIndex=i,
                controlMode=pb.POSITION_CONTROL,
                targetPosition=joints[i],
                force=500,
                positionGain=0.05,
                velocityGain=1
            )

        # Step the simulation for a short duration to allow movement.
        for _ in range(max_steps):
            pb.stepSimulation()
            time.sleep(self.dt)

    def move_to_cartesian(self, target_pos, target_ori, max_steps=240, error_threshold=0.01):
        """
        Moves the arm using inverse kinematics and closed-loop control until the end effector
        reaches the desired position and orientation within a threshold.

        Args:
            target_pos (list or np.array): Desired end-effector position [x, y, z].
            target_ori (list or np.array): Desired end-effector orientation (quaternion).
            max_steps (int): Maximum number of simulation steps to try.
            error_threshold (float): Acceptable Euclidean distance (in meters) between
                                    the current and target positions.
        """

        # Calculate the inverse kinematics solution.
        jointPoses = pb.calculateInverseKinematics(
            self.__kinova_id,
            self.END_EFFECTOR_INDEX,
            target_pos,
            target_ori,
            lowerLimits=self.__lower_limits,
            upperLimits=self.__upper_limits,
            jointRanges=self.__joint_ranges,
            restPoses=self.__rest_poses,
            maxNumIterations=100
        )

        # Slice the IK solution so that only the controlled joints are used.
        jointPoses = jointPoses[:self.__n_joints]

        for step in range(max_steps):
            # Command each joint to the desired position.
            for i in range(self.__n_joints):
                pb.setJointMotorControl2(
                    bodyIndex=self.__kinova_id,
                    jointIndex=i,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=jointPoses[i],
                    force=500,
                    positionGain=0.05,
                    velocityGain=1
                )

            # Step the simulation.
            pb.stepSimulation()
            time.sleep(self.dt)

            # Get the current end-effector state.
            ee_state = pb.getLinkState(self.__kinova_id, self.END_EFFECTOR_INDEX)
            current_pos = np.array(ee_state[0])
            current_error = np.linalg.norm(np.array(target_pos) - current_pos)

            # If within threshold, break out.
            if current_error < error_threshold:
                print("Target reached within threshold.")
                break

        # Final achieved state.
        final_state = pb.getLinkState(self.__kinova_id, self.END_EFFECTOR_INDEX)
        final_pos = final_state[0]
        final_ori = final_state[1]

        print("Target end-effector position:", target_pos)
        print("Final achieved end-effector position:", final_pos)

    def open_gripper(self):
        """
        Opens the gripper.
        """
        pb.setJointMotorControl2(self.__kinova_id, self.LEFT_FINGER_JOINT, pb.POSITION_CONTROL,
                                 targetPosition=self.GRIPPER_OPEN_POS, force=500)
        pb.setJointMotorControl2(self.__kinova_id, self.RIGHT_FINGER_JOINT, pb.POSITION_CONTROL,
                                 targetPosition=-self.GRIPPER_OPEN_POS, force=500)
        for _ in range(100):
            pb.stepSimulation()
            time.sleep(self.dt)

        print("Gripper opened.")

    def close_gripper(self):
        """
        Closes the gripper.
        """
        pb.setJointMotorControl2(self.__kinova_id, self.LEFT_FINGER_JOINT, pb.POSITION_CONTROL,
                                 targetPosition=self.GRIPPER_CLOSED_POS, force=500)
        pb.setJointMotorControl2(self.__kinova_id, self.RIGHT_FINGER_JOINT, pb.POSITION_CONTROL,
                                 targetPosition=self.GRIPPER_CLOSED_POS, force=500)
        for _ in range(100):
            pb.stepSimulation()
            time.sleep(self.dt)

        print("Gripper closed.")

def main():
    """
    Test the Gen3Lite Arm moving and gripper functionalities.
    """

    # Initialize PyBullet simulation
    pb.connect(pb.GUI, options="--opengl2") # NOTE: Make sure the GUI is set properly based on your system
    pb.setGravity(0, 0, -9.8)

    # Create the controller
    controller = Gen3LiteArmController()
    pb.setTimeStep(controller.dt)

    # Test homing functionality
    print("\nTesting Gen3Lite Arm controller homing...")
    controller.move_to_cartesian([0.4, 0, 0.4], controller.default_ori)
    controller.move_to_cartesian([-0.4, 0, 0.4], controller.default_ori)

    # Test gripper functionality
    print("\nTesting gripper open/close...")
    controller.open_gripper()
    controller.close_gripper()

    # Test the direct joint control
    print("\nTesting direct joint control...")
    home_ = [0, 0, 0, 0, 0, -math.pi * 0.5, 0]
    controller.move_to_joint_positions(home_)

    print("Homing test completed.")
    pb.disconnect()


if __name__ == "__main__":
    main()