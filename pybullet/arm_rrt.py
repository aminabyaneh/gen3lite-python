import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import random

import gen3lite_controller_collision_detection

# -----------------------------
# RRT Node
# -----------------------------
class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None

class Tree:
    def __init__(self,node_list,kdtree):
        self.node_list = node_list
        self.kdtree = kdtree
    
    def add(self,new_point):
        self.node_list.append(new_point)
        
        if len(self.node_list) % 50 == 0:
            data = [n.point for n in self.node_list]
            self.kdtree = cKDTree(data)

# -----------------------------
# RRT Planner
# -----------------------------
class RRT:
    def __init__(
        self,
        start=None,
        goal=None,
        obstacles=None,
        rand_area=[],
        step_size=0.1,
        goal_sample_rate=0.1,
        max_iter=500000
    ):
        self.controller = gen3lite_controller_collision_detection.Gen3LiteArmController()
        self.controller.createBalloonMaze()

        self.start = Node(self.controller.getCurrentJointAngles())
        self.goal = Node(self.controller.goal)
        self.rand_ranges = self.controller.getRanges()
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.start_tree = Tree([self.start],cKDTree([self.start.point]))
        self.goal_tree = Tree([self.goal],cKDTree([self.start.point]))

    def add_node(self,tree):
        rnd_point = self.sample()
        nearest_node = self.nearest_node(rnd_point,tree)
        new_node = self.steer(nearest_node, rnd_point)

        if self.collision_free(nearest_node.point, new_node.point):
            tree.add(new_node)

        return new_node

    # -----------------------------
    # Main planning loop
    #    returns: True when a path has been found
    #             False if we have no path by the set number of iterations
    # -----------------------------
    def plan(self):

        for k in tqdm(range(self.max_iter)):
            if k % 2:
                print("Adding to start")
                new_node = self.add_node(self.start_tree)

                while(new_node is not None):
                    ret = self.add_node(self.goal_tree,new_node)
                    if ret == None:
                        break
                    if ret == self.TREES_CONNECT:
                        print("Plan found!")
                        
            else:
                print("Adding to goal tree")
                new_node = self.add_node(self.goal_tree)

                while(self.add_node(self.start_tree,new_node)):

                if self.reached_goal(new_node):
                    self.path_to_goal = self.extract_path(new_node)
                    return True

        return False

    # -----------------------------
    # Sampling
    # -----------------------------
    def sample(self):
        if random.random() < self.goal_sample_rate:
            return self.goal.point
        point = []
        for i in range(0,len(self.rand_ranges[0])):
            point.append(random.uniform(self.rand_ranges[0][i], self.rand_ranges[1][i]))
        return np.array(point)

    # -----------------------------
    # Nearest node
    # -----------------------------
    def nearest_node(self, point,tree):
         _, idx = tree.kdtree.query(point)
         return tree.node_list[idx]

    # -----------------------------
    # Steer
    # -----------------------------
    def steer(self, from_node, to_point):
        direction = to_point - from_node.point
        distance = np.linalg.norm(direction)
        direction = direction / distance

        new_point = from_node.point + self.step_size * direction
        new_node = Node(new_point)
        new_node.parent = from_node
        return new_node

    # -----------------------------
    # Collision checking
    # -----------------------------
    def collision_free(self, p1, p2):
        return self.controller.collision_free(p1,p2)

    # -----------------------------
    # Goal check
    # -----------------------------
    def reached_goal(self, node):
        return np.linalg.norm(node.point - self.goal.point) < self.step_size

    # -----------------------------
    # Path extraction
    # -----------------------------
    def extract_path(self, node):
        path = [self.goal.point]
        while node is not None:
            path.append(node.point)
            node = node.parent
        return path[::-1]
# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    rrt = RRT()

    success = rrt.plan()
    if success:
        rrt.controller.execPath(rrt.path_to_goal)
        print("Planning complete. Moving the robot to the goal.")
    else:
        print("Failed to find a path to the goal.")
    