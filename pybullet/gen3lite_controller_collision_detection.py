import pybullet as pb
import time
import math
from typing import Dict, List, Optional
from enum import Enum
import numpy as np


class ControlModes(Enum):
    """
    Pybullet control modes, only for the end effector. We use IK to translate
    EF commands to joint velocities using a PID controller.
    """

    END_EFFECTOR_POSE = pb.POSITION_CONTROL
    END_EFFECTOR_TWIST = pb.VELOCITY_CONTROL


class Gen3LiteArmController:
    """
    A controller for the Kinova Gen3 Lite robotic arm in PyBullet.

    This class provides methods to control the arm's joints, move the end-effector
    to desired positions and orientations using inverse kinematics, and operate the gripper.
    """

    def __init__(self, dt=1 / 50.0):
        self.dt = dt

        self.LEFT_FINGER_JOINT = 7  # Example index; update if needed.
        self.RIGHT_FINGER_JOINT = 9  # Example index; update if needed.
        self.GRIPPER_OPEN_POS = 0.7  # Adjust as needed.
        self.GRIPPER_CLOSED_POS = 0.0  # Adjust as needed.

        # End-effector link index as used in your URDF.
        self.END_EFFECTOR_INDEX = 7

        pb.connect(pb.GUI,)
        pb.setGravity(0, 0, -9.8)
        pb.setTimeStep(self.dt)

        # Load the Kinova Gen3 Lite URDF model.
        # Ensure the path "gen3lite_urdf/gen3_lite.urdf" exists in your directory
        self.__kinova_id = pb.loadURDF("gen3lite_urdf/gen3_lite.urdf", [0, 0, 0], useFixedBase=True)
        pb.resetBasePositionAndOrientation(self.__kinova_id, [0, 0, 0.0], [0, 0, 0, 1])

        self.__n_joints = 7  # pb.getNumJoints(self.__kinova_id) - 5, where -5 for the gripper
        print(f'Found {self.__n_joints} active joints for the robot.')

        # Joint limits and rest/home poses.
        self.__lower_limits: List = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        self.__upper_limits: List = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        self.__joint_ranges: List = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        self.__rest_poses: List = [0, 0, 0, 0, 0, 0, 0]
        self.__home_poses: List = [0, 0, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, -math.pi * 0.5, 0]

        self.joint_ids = [pb.getJointInfo(self.__kinova_id, i) for i in range(self.__n_joints)]
        self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pb.JOINT_REVOLUTE]

        # Initialize to home position.
        for i in range(self.__n_joints):
            pb.resetJointState(self.__kinova_id, i, self.__home_poses[i])

        self.default_ori = list(pb.getQuaternionFromEuler([0, -math.pi, 0]))
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

    def getRanges(self):
        return (self.__lower_limits,self.__upper_limits)
    
    def getCurrentJointAngles(self):
        angles = []
        for id in self.joint_ids:
            joint_state = pb.getJointState(self.__kinova_id,id)
            angles.append(joint_state[0])
        return angles
        
    def createBalloonMaze(self):
        for y in [-0.125, 0.125]:
            for z in [0.25, 0.5]:
                col_box_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.1)
                box_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box_id, basePosition=[0.4, y, z])
    
    def set_to_home(self):
        """
        Resets the arm to its predefined home position.
        """
        for i in range(self.__n_joints):
            pb.resetJointState(self.__kinova_id, i, self.__home_poses[i])

    def execPath(self,path):
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        for p in path:
            self.move_to_joint_positions(p)

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
                force=2000,
                positionGain=1.0,
                velocityGain=1.0
            )

        # Step the simulation for a short duration to allow movement.
        for k in range(max_steps):
            pb.stepSimulation()
            curr = self.getCurrentJointAngles()
            e = np.linalg.norm(np.array(joints) - np.array(curr))
            print("Error at iter ", k, " is ", e)
            print("Target: ", joints)
            print("Current ", curr)
            if e < 0.1:
                break

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

    def set_joint_positions(self, joint_positions):
        for joint_index, q in enumerate(joint_positions):
            pb.resetJointState(self.__kinova_id, joint_index, q)

    def collision_free(self,p1,p2):
        
        self.set_joint_positions(p1)
        if self.check_collision():
            return False
        self.set_joint_positions(p2)
        if self.check_collision():
            return False
        return True

    # ------------------------------------------------------------------
    # UPDATED COLLISION FUNCTION
    # ------------------------------------------------------------------
    def check_collision(self):
        """
        Checks for collisions between the robot and ANY other body in the environment,
        as well as self-collisions (robot links hitting each other).

        Returns:
            bool: True if any collision is detected, False otherwise.
        """
        # Ensure collision detection is up to date
        pb.performCollisionDetection()

        # Iterate over all bodies in the PyBullet simulation
        for i in range(pb.getNumBodies()):
            other_body_id = pb.getBodyUniqueId(i)

            # Case 1: Self-collision (Body vs itself)
            if other_body_id == self.__kinova_id:
                contact_points = pb.getContactPoints(bodyA=self.__kinova_id, bodyB=self.__kinova_id)

            # Case 2: Environment collision (Body vs other object)
            else:
                contact_points = pb.getContactPoints(bodyA=self.__kinova_id, bodyB=other_body_id)

            # If contacts are found, return True immediately
            if contact_points is None or len(contact_points) > 0:
                return True

        # If loop completes without returning, no collisions were found
        return False

def main():
    """
    Test the Gen3Lite Arm moving, gripper functionalities, and collision detection.
    """

    # Initialize PyBullet simulation
    pb.connect(pb.GUI)
    pb.setGravity(0, 0, -9.8)

    # Create the controller
    controller = Gen3LiteArmController()
    pb.setTimeStep(controller.dt)

    # Test homing functionality
    print("\nTesting Gen3Lite Arm controller homing...")
    controller.move_to_cartesian([0.5, 0, 0.375], controller.default_ori)
    #controller.set_joint_positions([0, 0, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, -math.pi * 0.5, 0])

    # --- COLLISION DETECTION TEST START ---
    print("\nTesting Collision Detection...")

    for y in [-0.125, 0.125]:
        for z in [0.25, 0.5]:
            col_box_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.125)
            box_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box_id, basePosition=[0.4, y, z])

    #col_box_id = pb.createCollisionShape(pb.GEOM_SPHERE, halfExtents=[0.2, 0.2, 0.2])
    #box_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box_id, basePosition=[-1.0, 0, 0.5])

    #col_box_id = pb.createCollisionShape(pb.GEOM_SPHERE, halfExtents=[0.2, 0.2, 0.2])
    #box_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box_id, basePosition=[0.0, 1.0, 0.5])

    print(controller.getCurrentJointAngles())
    for i in range (10000):
        pb.stepSimulation()
        time.sleep(1./240.)

    pb.disconnect()

    # Note: No arguments passed to check_collision
    is_collision = controller.check_collision()
    print(f"Box at [1.0, 0, 0.5]. Collision detected? {is_collision} (Expected: False)")

    return 

    # 3. Move the box to where the arm currently is (approx [0.4, 0, 0.4])
    print("Moving box to collide with arm...")
    pb.resetBasePositionAndOrientation(box_id, [0.4, 0, 0.4], [0, 0, 0, 1])
    pb.stepSimulation()

    is_collision = controller.check_collision()
    print(f"Box at [0.4, 0, 0.4]. Collision detected? {is_collision} (Expected: True)")

    # Remove the box to continue other tests cleanly
    pb.removeBody(box_id)
    # --- COLLISION DETECTION TEST END ---

    controller.move_to_cartesian([-0.4, 0, 0.4], controller.default_ori)

    # Test gripper functionality
    print("\nTesting gripper open/close...")
    controller.open_gripper()
    controller.close_gripper()

    # Test the direct joint control
    print("\nTesting direct joint control...")
    home_ = [0, 0, 0, 0, 0, -math.pi * 0.5, 0]
    controller.move_to_joint_positions(home_)

    print("All tests completed.")
    pb.disconnect()


if __name__ == "__main__":
    main()

# import pybullet as pb
# import time
# import math
# from typing import Dict, List
# from enum import Enum
# import numpy as np
#
#
# class ControlModes(Enum):
#     """
#     Pybullet control modes, only for the end effector. We use IK to translate
#     EF commands to joint velocities using a PID controller.
#     """
#
#     END_EFFECTOR_POSE = pb.POSITION_CONTROL
#     END_EFFECTOR_TWIST = pb.VELOCITY_CONTROL
#
#
# class Gen3LiteArmController:
#     """
#     A controller for the Kinova Gen3 Lite robotic arm in PyBullet.
#
#     This class provides methods to control the arm's joints, move the end-effector
#     to desired positions and orientations using inverse kinematics, and operate the gripper.
#     """
#
#     def __init__(self, dt=1 / 50.0):
#         self.dt = dt
#
#         self.LEFT_FINGER_JOINT = 7  # Example index; update if needed.
#         self.RIGHT_FINGER_JOINT = 9  # Example index; update if needed.
#         self.GRIPPER_OPEN_POS = 0.7  # Adjust as needed.
#         self.GRIPPER_CLOSED_POS = 0.0  # Adjust as needed.
#
#         # End-effector link index as used in your URDF.
#         self.END_EFFECTOR_INDEX = 7
#
#         # Load the Kinova Gen3 Lite URDF model.
#         # Ensure the path "gen3lite_urdf/gen3_lite.urdf" exists in your directory
#         self.__kinova_id = pb.loadURDF("gen3lite_urdf/gen3_lite.urdf", [0, 0, 0], useFixedBase=True)
#         pb.resetBasePositionAndOrientation(self.__kinova_id, [0, 0, 0.0], [0, 0, 0, 1])
#
#         self.__n_joints = 7  # pb.getNumJoints(self.__kinova_id) - 5, where -5 for the gripper
#         print(f'Found {self.__n_joints} active joints for the robot.')
#
#         # Joint limits and rest/home poses.
#         self.__lower_limits: List = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#         self.__upper_limits: List = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#         self.__joint_ranges: List = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#         self.__rest_poses: List = [0, 0, 0, 0, 0, 0, 0]
#         self.__home_poses: List = [0, 0, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, -math.pi * 0.5, 0]
#
#         self.joint_ids = [pb.getJointInfo(self.__kinova_id, i) for i in range(self.__n_joints)]
#         self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pb.JOINT_REVOLUTE]
#
#         # Initialize to rest position.
#         for i in range(self.__n_joints):
#             pb.resetJointState(self.__kinova_id, i, self.__rest_poses[i])
#
#         self.default_ori = list(pb.getQuaternionFromEuler([0, -math.pi, 0]))
#
#     def set_to_home(self):
#         """
#         Resets the arm to its predefined home position.
#         """
#         for i in range(self.__n_joints):
#             pb.resetJointState(self.__kinova_id, i, self.__home_poses[i])
#
#     def move_to_joint_positions(self, joints, max_steps=100):
#         """
#         Move to target joint positions with position control.
#
#         Args:
#             joints (list): Target joint positions.
#             max_steps (int): Maximum simulation steps to reach the target.
#         """
#         for i in range(self.__n_joints):
#             pb.setJointMotorControl2(
#                 bodyIndex=self.__kinova_id,
#                 jointIndex=i,
#                 controlMode=pb.POSITION_CONTROL,
#                 targetPosition=joints[i],
#                 force=500,
#                 positionGain=0.05,
#                 velocityGain=1
#             )
#
#         # Step the simulation for a short duration to allow movement.
#         for _ in range(max_steps):
#             pb.stepSimulation()
#             time.sleep(self.dt)
#
#     def move_to_cartesian(self, target_pos, target_ori, max_steps=240, error_threshold=0.01):
#         """
#         Moves the arm using inverse kinematics and closed-loop control until the end effector
#         reaches the desired position and orientation within a threshold.
#
#         Args:
#             target_pos (list or np.array): Desired end-effector position [x, y, z].
#             target_ori (list or np.array): Desired end-effector orientation (quaternion).
#             max_steps (int): Maximum number of simulation steps to try.
#             error_threshold (float): Acceptable Euclidean distance (in meters) between
#                                     the current and target positions.
#         """
#
#         # Calculate the inverse kinematics solution.
#         jointPoses = pb.calculateInverseKinematics(
#             self.__kinova_id,
#             self.END_EFFECTOR_INDEX,
#             target_pos,
#             target_ori,
#             lowerLimits=self.__lower_limits,
#             upperLimits=self.__upper_limits,
#             jointRanges=self.__joint_ranges,
#             restPoses=self.__rest_poses,
#             maxNumIterations=100
#         )
#
#         # Slice the IK solution so that only the controlled joints are used.
#         jointPoses = jointPoses[:self.__n_joints]
#
#         for step in range(max_steps):
#             # Command each joint to the desired position.
#             for i in range(self.__n_joints):
#                 pb.setJointMotorControl2(
#                     bodyIndex=self.__kinova_id,
#                     jointIndex=i,
#                     controlMode=pb.POSITION_CONTROL,
#                     targetPosition=jointPoses[i],
#                     force=500,
#                     positionGain=0.05,
#                     velocityGain=1
#                 )
#
#             # Step the simulation.
#             pb.stepSimulation()
#             time.sleep(self.dt)
#
#             # Get the current end-effector state.
#             ee_state = pb.getLinkState(self.__kinova_id, self.END_EFFECTOR_INDEX)
#             current_pos = np.array(ee_state[0])
#             current_error = np.linalg.norm(np.array(target_pos) - current_pos)
#
#             # If within threshold, break out.
#             if current_error < error_threshold:
#                 print("Target reached within threshold.")
#                 break
#
#         # Final achieved state.
#         final_state = pb.getLinkState(self.__kinova_id, self.END_EFFECTOR_INDEX)
#         final_pos = final_state[0]
#         final_ori = final_state[1]
#
#         print("Target end-effector position:", target_pos)
#         print("Final achieved end-effector position:", final_pos)
#
#     def open_gripper(self):
#         """
#         Opens the gripper.
#         """
#         pb.setJointMotorControl2(self.__kinova_id, self.LEFT_FINGER_JOINT, pb.POSITION_CONTROL,
#                                  targetPosition=self.GRIPPER_OPEN_POS, force=500)
#         pb.setJointMotorControl2(self.__kinova_id, self.RIGHT_FINGER_JOINT, pb.POSITION_CONTROL,
#                                  targetPosition=-self.GRIPPER_OPEN_POS, force=500)
#         for _ in range(100):
#             pb.stepSimulation()
#             time.sleep(self.dt)
#
#         print("Gripper opened.")
#
#     def close_gripper(self):
#         """
#         Closes the gripper.
#         """
#         pb.setJointMotorControl2(self.__kinova_id, self.LEFT_FINGER_JOINT, pb.POSITION_CONTROL,
#                                  targetPosition=self.GRIPPER_CLOSED_POS, force=500)
#         pb.setJointMotorControl2(self.__kinova_id, self.RIGHT_FINGER_JOINT, pb.POSITION_CONTROL,
#                                  targetPosition=self.GRIPPER_CLOSED_POS, force=500)
#         for _ in range(100):
#             pb.stepSimulation()
#             time.sleep(self.dt)
#
#         print("Gripper closed.")
#
#     def check_collision(self, other_object_id):
#         """
#         Determines if there is a collision between the Kinova Gen3 Lite robotic arm
#         and another object based on their current positions in the PyBullet simulation.
#
#         Args:
#             other_object_id (int): The PyBullet body unique ID of the other object.
#
#         Returns:
#             bool: True if a collision is detected, False otherwise.
#         """
#         # Ensure collision detection is up to date
#         pb.performCollisionDetection()
#
#         # Check for contact points between the arm and the specified object
#         contact_points = pb.getContactPoints(bodyA=self.__kinova_id, bodyB=other_object_id)
#
#         # Return True if any contact points exist
#         return len(contact_points) > 0
#
#
# def main():
#     """
#     Test the Gen3Lite Arm moving, gripper functionalities, and collision detection.
#     """
#
#     # Initialize PyBullet simulation
#     pb.connect(pb.GUI, options="--opengl2")
#     pb.setGravity(0, 0, -9.8)
#
#     # Create the controller
#     controller = Gen3LiteArmController()
#     pb.setTimeStep(controller.dt)
#
#     # Test homing functionality
#     print("\nTesting Gen3Lite Arm controller homing...")
#     controller.move_to_cartesian([0.4, 0, 0.4], controller.default_ori)
#
#     # --- COLLISION DETECTION TEST START ---
#     print("\nTesting Collision Detection...")
#
#     # 1. Create a dummy box object
#     col_box_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
#
#     # 2. Spawn the box far away (at x=1.0) where it shouldn't hit the arm
#     box_id = pb.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box_id, basePosition=[1.0, 0, 0.5])
#     pb.stepSimulation()
#
#     is_collision = controller.check_collision(box_id)
#     print(f"Box at [1.0, 0, 0.5]. Collision detected? {is_collision} (Expected: False)")
#
#     # 3. Move the box to where the arm currently is (approx [0.4, 0, 0.4])
#     print("Moving box to collide with arm...")
#     pb.resetBasePositionAndOrientation(box_id, [0.4, 0, 0.4], [0, 0, 0, 1])
#     pb.stepSimulation()
#
#     is_collision = controller.check_collision(box_id)
#     print(f"Box at [0.4, 0, 0.4]. Collision detected? {is_collision} (Expected: True)")
#
#     # Remove the box to continue other tests cleanly
#     pb.removeBody(box_id)
#     # --- COLLISION DETECTION TEST END ---
#
#     controller.move_to_cartesian([-0.4, 0, 0.4], controller.default_ori)
#
#     # Test gripper functionality
#     print("\nTesting gripper open/close...")
#     controller.open_gripper()
#     controller.close_gripper()
#
#     # Test the direct joint control
#     print("\nTesting direct joint control...")
#     home_ = [0, 0, 0, 0, 0, -math.pi * 0.5, 0]
#     controller.move_to_joint_positions(home_)
#
#     print("All tests completed.")
#     pb.disconnect()
#
#
# if __name__ == "__main__":
#     main()