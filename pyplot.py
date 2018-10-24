from imsl_dataset import IMSLDataset
gogo_dataset = IMSLDataset('newgogo', '/home/huangjianjun/LandmarkData/Format/', 'map', 'landmark_1')

import numpy as np
import matplotlib.pyplot as plt
from numpy import arccos, dot, pi, cross
from numpy.linalg import norm

# point to segment line
def distance_numpy(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    if all(A == P) or all(B == P):
        return 0
    if arccos(dot((P - A) / norm(P - A), (B - A) / norm(B - A))) > pi / 2:
        return norm(P - A)
    if arccos(dot((P - B) / norm(P - B), (A - B) / norm(A - B))) > pi / 2:
        return norm(P - B)
    return norm(cross(A-B, A-P))/norm(B-A)

def projection(A, B, P):
    """ segment line AB, point P, where each one is an array([x, y]) """
    return dot(P - A, B - A) / (norm(B -A) ** 2) * (B - A) + A

for model in gogo_dataset.maps[0].map_models:
    # print("%d %d %d %d" % (model.start.x, model.start.y, model.end.x, model.end.y))
    plt.plot(np.array([model.start.x, model.end.x]),
             np.array([model.start.y, model.end.y]),
             color="#000000", linewidth=2.0, linestyle="-", )

for landmark in gogo_dataset.get_landmark_info().values():
    mindis = 9999999.0
    minpro = np.zeros(2)
    for model in gogo_dataset.maps[0].map_models:
        # dis, pro = point_to_line_direction(landmark.position.x, landmark.position.y, model.end.x, model.end.y,
        #                                    model.start.x, model.start.y)
        # if dis < mindis:
        #     mindis = dis
        #     minpro = pro
        A = np.array([model.start.x, model.start.y])
        B = np.array([model.end.x, model.end.y])
        P = np.array([landmark.position.x, landmark.position.y])
        dis = distance_numpy(A, B, P)
        pro = projection(A, B, P)
        if dis < mindis:
            mindis = dis
            minpro = pro

    plt.plot(landmark.position.x, landmark.position.y, "ro", color="#FF0000")
    plt.plot(minpro[0], minpro[1], "ro", color="#00FF00")

plt.show()

# np.random.seed(19680801)
#
#
# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii
#
# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()