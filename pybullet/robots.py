import pybullet as pb
import time
import math
from typing import List
import numpy as np

class Gen3LiteArmController(object):
    """
    A controller for the Kinova Gen3 Lite robotic arm in PyBullet.

    This class provides methods to control the arm's joints and interact 
    with it surrounding environment through collision checking.
    """
    def __init__(self, dt=1 / 50.0, env_name="Free"):
        
        self.dt = dt
        pb.connect(pb.GUI,)
        pb.setGravity(0, 0, -9.8)
        pb.setTimeStep(self.dt)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # Load the Kinova Gen3 Lite URDF model.
        # Ensure the path "gen3lite_urdf/gen3_lite.urdf" exists in your directory
        self.__kinova_id = pb.loadURDF("gen3lite_urdf/gen3_lite.urdf", [0, 0, 0], useFixedBase=True)
        pb.resetBasePositionAndOrientation(self.__kinova_id, [0, 0, 0.0], [0, 0, 0, 1])
        self.END_EFFECTOR_INDEX = 7

        self.__n_joints = 7 
        self.__lower_limits: List = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        self.__upper_limits: List = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        self.__home_poses: List = [math.pi, 0, 0.5 * math.pi, 0.5 * math.pi, 0.5 * math.pi, -math.pi * 0.5, 0]

        self.joint_ids = [pb.getJointInfo(self.__kinova_id, i) for i in range(self.__n_joints)]
        self.joint_ids = [j[0] for j in self.joint_ids if j[2] == pb.JOINT_REVOLUTE]

        # This function creates obstacles and sets a reachable goal
        self.createBalloonMaze(env_name)

        # Set the viewer's camera viewpoint
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=-120,
            cameraPitch=-10,
            cameraTargetPosition=self.goal_position
        )

    def getRanges(self):
        return (self.__lower_limits,self.__upper_limits)
    
    def getCurrentJointAngles(self):
        angles = []
        for id in self.joint_ids:
            joint_state = pb.getJointState(self.__kinova_id,id)
            angles.append(joint_state[0])
        return angles
    
    def setJointAngles(self, joint_angles):
        for joint_index, q in enumerate(joint_angles):
            pb.resetJointState(self.__kinova_id, joint_index, q)

    def setToHome(self):
        self.setJointAngles(self.__home_poses)

    def fkine(self,angles):
        # Record where we were (usually home, but just to be safe...)
        curr = self.getCurrentJointAngles()
        # Move the arm to the goal, specified in joint angles
        self.setJointAngles(angles)
        # Record the end effectors x,y,z position
        state = pb.getLinkState(self.__kinova_id, self.END_EFFECTOR_INDEX)
        # Move the arm back to where it began
        self.setJointAngles(curr)

        return state[0]

    def execPath(self,path):
        self.setJointAngles(self.__home_poses)
        
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        for p in path:
            self.setJointAngles(p)
            time.sleep(0.25)

    def createBalloonMaze(self,env_name):
        if env_name == "Hardest":
            self.createHardestMaze()
        elif env_name == "Easiest":
            self.createEasiestMaze()
        elif env_name == "Free":
            self.createFreeMaze()

    def createFreeMaze(self):
        
        # Initialize to home position.
        self.setToHome()

        # This is a hard-coded sensible goal to try to reach
        self.goal_angles = [0.1026325022237283, -0.2931188624740633, 1.2717083400432991, 0.048794139164578594, 0.07744723004754135, -0.8437927483158898, -0.024709326684397483]

        self.goal_position = self.fkine(self.goal_angles)

        # Make a visual marker for the goal
        goal_visual_id = pb.createVisualShape(pb.GEOM_BOX, 
                                              halfExtents=[0.05,0.05,0.05],
                                              rgbaColor=[0.0,0.0,1.0,0.5]
                                              )
        
        pb.createMultiBody(baseMass=0, basePosition=self.goal_position, baseVisualShapeIndex=goal_visual_id )
  
    def createEasiestMaze(self):
        self.createFreeMaze()
        balloon_collision_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.07)
        balloon_visual_id = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.07,rgbaColor=[1.0,0.0,0.0,0.5])

        x = 0.4
        for y in [-0.125, 0.125]:
            for z in [0.25, 0.5]:
                box_id = pb.createMultiBody(baseMass=0, basePosition=[x, y, z],baseCollisionShapeIndex=balloon_collision_id,
                baseVisualShapeIndex=balloon_visual_id  
                )

    def createHardestMaze(self):
        self.createEasiestMaze()

        balloon_collision_id = pb.createCollisionShape(pb.GEOM_SPHERE, radius=0.15)
        balloon_visual_id = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.15,rgbaColor=[1.0,0.0,0.0,0.5])

        x = 0
        for y in [-0.3, 0.3]:
            for z in [0.2, 0.6]:
                box_id = pb.createMultiBody(baseMass=0, basePosition=[x, y, z],baseCollisionShapeIndex=balloon_collision_id,
                baseVisualShapeIndex=balloon_visual_id  
                )        
    
    def visTreesAndPaths(self,trees,paths,rgbas_in):
        
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        for tree_idx in range(len(trees)):
            tree = trees[tree_idx]
            rgba_in = rgbas_in[tree_idx]
            for n in tree.node_list:
                parent = n.parent
                if not parent is None:
                    self.setJointAngles(n.point)
                    n_pos = pb.getLinkState(self.__kinova_id, self.END_EFFECTOR_INDEX)
            
                    self.setJointAngles(parent.point)
                    parent_pos = pb.getLinkState(self.__kinova_id, self.END_EFFECTOR_INDEX)

                    point_vis_id = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.005,rgbaColor=rgba_in)
                    node_sphere_id = pb.createMultiBody(baseMass=0, basePosition=n_pos[0], baseVisualShapeIndex=point_vis_id )

                    pb.addUserDebugLine(lineFromXYZ=n_pos[0],lineToXYZ=parent_pos[0],lineColorRGB=rgba_in[0:3],lineWidth=0.01,lifeTime=0)

        point_vis_id = pb.createVisualShape(pb.GEOM_SPHERE, radius=0.01,
                                            rgbaColor=[1.0, 1.0, 0.0, 0.5])

        for path in paths:
            for idx in range(len(path)):
                    
                    parent_pos = self.fkine(path[idx])

                    node_sphere_id = pb.createMultiBody(baseMass=0, basePosition=parent_pos, baseVisualShapeIndex=point_vis_id )

                    if idx+1 < len(path):
                        child_pos = self.fkine(path[idx+1])

                        pb.addUserDebugLine(lineFromXYZ=parent_pos,lineToXYZ=child_pos,lineColorRGB=[1.0, 1.0, 0.0],lineWidth=0.02,lifeTime=0)
        
        self.setToHome()
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        print("Finished plotting tree. Press q to continue.")

        while True:
            pb.stepSimulation()
            time.sleep(1./240.)

            keys = pb.getKeyboardEvents()
    
            if ord('q') in keys and keys[ord('q')] & pb.KEY_WAS_TRIGGERED:
                print("Q pressed. Continuing.")
                break

    def collision_free(self,p1,p2):
        
        self.setJointAngles(p1)
        if self.check_collision():
            return False
        self.setJointAngles(p2)
        if self.check_collision():
            return False
        return True

    def check_collision(self):
        """
        Checks for collisions between the robot and ANY other body in the environment, as well as self-collisions 
        (robot links hitting each other).

        Returns:
            bool: True if any collision is detected, False otherwise.
        """
        # Ensure collision detection is up to date
        pb.performCollisionDetection()

        # Iterate over all bodies in the PyBullet simulation
        for i in range(pb.getNumBodies()):
            other_body_id = pb.getBodyUniqueId(i)
            contact_points = pb.getContactPoints(bodyA=self.__kinova_id, bodyB=other_body_id)

            # If contacts are found, return True immediately
            if contact_points is None or len(contact_points) > 0:
                return True

        # If loop completes without returning, no collisions were found
        return False
