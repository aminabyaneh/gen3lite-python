import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import random
import argparse
import gen3lite_controller_collision_detection

class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None

class Tree:
    def __init__(self, node_list, kdtree):
        self.node_list = node_list
        self.kdtree = kdtree
    
    def add(self,new_point):
        self.node_list.append(new_point)
        
        if len(self.node_list) % 50 == 0:
            data = [n.point for n in self.node_list]
            self.kdtree = cKDTree(data)

class RRT:
    def __init__(self, step_size=0.1,max_iter=50000):
        self.controller = gen3lite_controller_collision_detection.Gen3LiteArmController()

        self.start = Node(self.controller.getCurrentJointAngles())
        self.goal = Node(self.controller.goal)
        self.rand_ranges = self.controller.getRanges()
        self.step_size = step_size
        self.max_iter = max_iter
        self.start_tree = Tree([self.start],cKDTree([self.start.point]))
        self.goal_tree = Tree([self.goal],cKDTree([self.goal.point]))
        self.path_to_goal = []

    def add_node(self,tree,target=None):
        if target is None:
            rnd_point = self.sample()
        else:
            rnd_point = target.point
        nearest_node = self.nearest_node(rnd_point,tree)
        new_node = self.steer(nearest_node, rnd_point)

        if self.collision_free(nearest_node.point, new_node.point):
            tree.add(new_node)
            return new_node
        else:
            return None

    def plan(self):

        for k in tqdm(range(self.max_iter)):
            if k % 2:
                new_node = self.add_node(self.start_tree)

                while(new_node is not None):
                    ret = self.add_node(self.goal_tree,new_node)
                    if ret == None:
                        break
                    if self.reached_goal(ret,goal=new_node):
                        self.path_to_goal = self.extract_path(new_node,ret)
                        return True
                        
            else:
                new_node = self.add_node(self.goal_tree)

                while(new_node is not None):
                    ret = self.add_node(self.start_tree,new_node)
                    if ret == None:
                        break

                    if self.reached_goal(ret,goal=new_node):
                        self.path_to_goal = self.extract_path(ret,new_node)
                        return True

        return False

    def sample(self):
        point = []
        for i in range(0,len(self.rand_ranges[0])):
            point.append(random.uniform(self.rand_ranges[0][i], self.rand_ranges[1][i]))
        return np.array(point)

    def nearest_node(self, point, tree):
         _, idx = tree.kdtree.query(point)
         return tree.node_list[idx]

    def steer(self, from_node, to_point):
        direction = to_point - from_node.point
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            new_point = to_point
        else:
            direction = direction / distance
            new_point = from_node.point + self.step_size * direction

        new_node = Node(new_point)
        new_node.parent = from_node
        return new_node

    def collision_free(self, p1, p2):
        return self.controller.collision_free(p1,p2)

    def reached_goal(self, node,goal=None):
        if goal is None:
            goal = self.goal
        
        return np.linalg.norm(node.point - goal.point) < self.step_size and self.collision_free(node.point,goal.point)

    def extract_path(self, start_node,goal_node):
        # Build the start-tree path, leaf to root (backwards) 
        start_tree_path = []
        while start_node is not None:
            start_tree_path.append(start_node.point)
            start_node = start_node.parent
        
        # Build the goal-tree path, leaf to root (forwards)
        goal_tree_path = []
        while goal_node is not None:
            goal_tree_path.append(goal_node.point)
            goal_node = goal_node.parent 

        # First add the start path, reversing
        overall_path = start_tree_path[::-1]
        # Add the goal path
        overall_path.extend(goal_tree_path)
        return overall_path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='arm_rrt',
                    description='Plans and executes paths for arms around obstacles.')
    parser.add_argument('--filename',default='rrt_path.npy')           
    parser.add_argument('-p', '--plan',action='store_true')
    parser.add_argument('-e', '--exec',action='store_true')
    args = parser.parse_args()

    if args.plan:
        rrt = RRT()
        success = rrt.plan()

        if success:
            print("Tree planning reached the goal.")
            np.save(args.filename,rrt.path_to_goal)
        else:
            print("Failed to find a path to the goal.")

        rrt.controller.visTreesAndPaths( 
            [rrt.start_tree,rrt.goal_tree],             
            [rrt.path_to_goal],
            rgbas_in=[[0.5,0.0,0.5,1.0],[0.902,0.106,0.714,1.0]]
        )

    if args.exec:
        path_to_goal = np.load(args.filename)
        rrt = RRT()
        rrt.controller.execPath(path_to_goal)
    