# -*- coding: utf-8 -*-
"""
Kinematics of continuum robot
@Author: bob

"""
import sys
import time
import numpy as np
import math 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#rotation matrix Z axis
def RMZ(phi):
    cosphi = math.cos(phi)
    sinphi = math.sin(phi)
    return([[cosphi, -sinphi, 0],[sinphi, cosphi, 0],[0, 0, 1]])

# rotation matrix Y axis
def RMY(theta):
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    return ([[costheta, 0, sintheta],[0, 1, 0],[-sintheta, 0, costheta]])

def FKA2J2DOFs(cable_len, robot_len, cable2center_dis):
    phi = math.atan2(robot_len - cable_len[1], robot_len - cable_len[0])
    if np.abs(robot_len - cable_len[0]) > 0.0000001: 
        theta = (robot_len - cable_len[0]) / (cable2center_dis * math.cos(phi))
    elif np.abs(robot_len - cable_len[1]) > 0.0000001:
        theta = (robot_len - cable_len[1]) / (cable2center_dis * math.sin(phi)) 
    else:
        theta = 0  
    return ([phi, theta])

def FKJ2C2DOFs(phi,theta,robot_len):
    RZ = np.mat(RMZ(phi))
    RY = np.mat(RMY(theta))
    if np.abs(theta) > 0.000001: 
        r = robot_len/theta
        p = np.mat([[r - r * math.cos(theta)], [0], [r * math.sin(theta)]])
        position = RZ * p
    else:
        position = [[0], [0], [robot_len]]

    return(position)  

robot_len = 100
cable2center_dis = 3
j = 0
interval1 = 0.01
interval2 = 0.01
len_array1 = len(np.arange(robot_len-math.pi, robot_len+math.pi, interval1))
len_array2 = len(np.arange(robot_len-math.pi, robot_len+math.pi, interval2))
pos = np.mat(np.zeros((3,len_array1 * len_array2)))
cable_len = np.mat(np.zeros((2,len_array1 * len_array2)))

for cable1 in np.arange(robot_len - math.pi, robot_len + math.pi, interval1):
    for cable2 in np.arange(robot_len - math.pi, robot_len + math.pi, interval2):
        cable_len[:,j] = [[cable1], [cable2]]
        phi,theta = FKA2J2DOFs([cable1, cable2], robot_len, cable2center_dis)
        #print('phi=%f, theta=%f, cable1 = %f' %(phi,theta,cable1))
        pos[:,j] = FKJ2C2DOFs(phi, theta, robot_len)
        j = j + 1
'''
interval1 = len(np.arange(- math.pi, math.pi, interval))
interval2 = len(np.arange(0, math.pi/3, interval))
pos = np.mat(np.zeros((3,interval1 * interval2)))
for phi in np.arange(-math.pi, math.pi, interval):
    for theta in np.arange(0, math.pi/3, interval):
        #cable_len = [cable1, cable2]
        #phi,theta = FKA2J2DOFs(cable_len, robot_len, cable2center_dis)
        #print(phi)
        pos[:,j] = FKJ2C2DOFs(phi, theta, robot_len)
        j = j + 1
'''

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[0,:], pos[1,:], pos[2,:])
plt.show()
'''
np.save('input.npy',cable_len)
np.save('output.npy',pos)

inputdata = np.load('input.npy')
outputdata = np.load('output.npy')
print(inputdata)
print(outputdata)
print(pos[:,len_array1 * len_array2 - 1])