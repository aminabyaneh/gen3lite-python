import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import random
import math
import gen3lite_controller_collision_detection

# -----------------------------
# RRT Node
# -----------------------------
class Node:
    def __init__(self, point):
        self.point = np.array(point)
        self.parent = None

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
        max_iter=5000
    ):
        self.controller = gen3lite_controller_collision_detection.Gen3LiteArmController()
        self.controller.createBalloonMaze()

        self.start = Node(self.controller.getCurrentJointAngles())
        self.goal = Node([0.1026325022237283, -0.2931188624740633, 1.2717083400432991, 0.048794139164578594, 0.07744723004754135, -0.8437927483158898, -0.024709326684397483])
        self.rand_ranges = self.controller.getRanges()
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.kdtree = cKDTree([self.start.point])

    # -----------------------------
    # Main planning loop
    # -----------------------------
    def plan(self):

        for _ in tqdm(range(self.max_iter)):

            rnd_point = self.sample()
            nearest_node = self.nearest_node(rnd_point)
            new_node = self.steer(nearest_node, rnd_point)

            if self.collision_free(nearest_node.point, new_node.point):
                self.node_list.append(new_node)

                if len(self.node_list) % 50 == 0:
                    data = [n.point for n in self.node_list]
                    self.kdtree = cKDTree(data)

                if self.reached_goal(new_node):
                    p = self.extract_path(new_node)
                    self.controller.execPath(p)
                    return p

        return None  # Failed

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
    def nearest_node(self, point):
         _, idx = self.kdtree.query(point)
         return self.node_list[idx]

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
    # Visualization
    # -----------------------------
    def draw(self, path=None):
        plt.figure()
        for node in self.node_list:
            if node.parent:
                plt.plot(
                    [node.point[0], node.parent.point[0]],
                    [node.point[1], node.parent.point[1]],
                    "-g"
                )

        for (ox, oy, r) in self.obstacles:
            circle = plt.Circle((ox, oy), r, color="r")
            plt.gca().add_patch(circle)

        plt.plot(self.start.point[0], self.start.point[1], "bo", label="Start")
        plt.plot(self.goal.point[0], self.goal.point[1], "ro", label="Goal")

        if path:
            px, py = zip(*path)
            plt.plot(px, py, "-b", linewidth=2, label="Path")

        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.show()


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":

    rrt = RRT()

    path = rrt.plan()
    print("Planning complete.")
    print(path)
    #rrt.draw(path)
