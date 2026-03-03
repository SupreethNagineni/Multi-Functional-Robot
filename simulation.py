import pybullet as p
import pybullet_data
import time
import os
import cv2
import numpy as np
import math

# ==========================================
# 1. GENERATE CUSTOM ROBOT URDF
# ==========================================
urdf_content = """<?xml version="1.0"?>
<robot name="mobile_picker">
    <link name="base_link">
        <visual><geometry><box size="0.4 0.3 0.1"/></geometry><material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
        <collision><geometry><box size="0.4 0.3 0.1"/></geometry></collision>
        <inertial><mass value="10.0"/><inertia ixx="0.1" iyy="0.1" izz="0.1"/></inertial>
    </link>
    <link name="arm_base">
        <visual><geometry><cylinder radius="0.05" length="0.1"/></geometry><material name="grey"><color rgba="0.5 0.5 0.5 1"/></material></visual>
        <inertial><mass value="1.0"/><inertia ixx="0.01" iyy="0.01" izz="0.01"/></inertial>
    </link>
    <joint name="joint_base_pan" type="revolute">
        <parent link="base_link"/><child link="arm_base"/>
        <origin xyz="0.15 0 0.1"/><axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>
    <link name="arm_link1">
        <visual><geometry><box size="0.25 0.05 0.05"/></geometry><origin xyz="0.125 0 0"/><material name="yellow"><color rgba="0.8 0.8 0.1 1"/></material></visual>
        <collision><geometry><box size="0.25 0.05 0.05"/></geometry><origin xyz="0.125 0 0"/></collision>
        <inertial><mass value="0.5"/><origin xyz="0.125 0 0"/><inertia ixx="0.01" iyy="0.01" izz="0.01"/></inertial>
    </link>
    <joint name="joint_shoulder" type="revolute">
        <parent link="arm_base"/><child link="arm_link1"/>
        <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>
    <link name="arm_link2">
        <visual><geometry><box size="0.25 0.05 0.05"/></geometry><origin xyz="0.125 0 0"/><material name="yellow"/></visual>
        <collision><geometry><box size="0.25 0.05 0.05"/></geometry><origin xyz="0.125 0 0"/></collision>
        <inertial><mass value="0.2"/><origin xyz="0.125 0 0"/><inertia ixx="0.01" iyy="0.01" izz="0.01"/></inertial>
    </link>
    <joint name="joint_elbow" type="revolute">
        <parent link="arm_link1"/><child link="arm_link2"/>
        <origin xyz="0.25 0 0"/><axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
    </joint>
</robot>
"""

with open("temp_robot.urdf", "w") as f: f.write(urdf_content)

# ==========================================
# 2. SETUP ENVIRONMENT & OVAL PATH
# ==========================================
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=-45, cameraPitch=-50, cameraTargetPosition=[2.0, 1.5, 0])

plane_id = p.loadURDF("plane.urdf")

# Track dimensions
a_radius = 2.0  
b_radius = 1.5  
num_segments = 60

# Generate the oval track
for i in range(num_segments):
    theta = i * (2 * math.pi / num_segments)
    x = a_radius * math.sin(theta)
    y = b_radius - b_radius * math.cos(theta)
    
    dx = a_radius * math.cos(theta)
    dy = b_radius * math.sin(theta)
    angle = math.atan2(dy, dx)
    
    length = math.sqrt(dx**2 + dy**2) * (2 * math.pi / num_segments) * 1.1 
    
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, 0.05, 0.001], rgbaColor=[0.1, 0.1, 0.1, 1])
    p.createMultiBody(baseVisualShapeIndex=visual_shape, basePosition=[x, y, 0.002], 
                      baseOrientation=p.getQuaternionFromEuler([0, 0, angle]))

robot_id = p.loadURDF("temp_robot.urdf", [0, 0, 0.05])
os.remove("temp_robot.urdf") 

# Place THREE obstacles along the curve using a list
obstacle_ids = []
# Angles on the oval where obstacles will be placed
obstacle_angles = [math.pi / 6, math.pi / 2, math.pi * 1.25] 

for obs_theta in obstacle_angles:
    obs_x = a_radius * math.sin(obs_theta)
    obs_y = b_radius - b_radius * math.cos(obs_theta)
    # Add block and append its unique ID to our list
    obs_id = p.loadURDF("cube_small.urdf", [obs_x, obs_y, 0.025])
    obstacle_ids.append(obs_id)

JOINT_PAN, JOINT_SHOULDER, JOINT_ELBOW = 0, 1, 2

# ==========================================
# 3. SENSOR & CONTROL FUNCTIONS
# ==========================================
def move_arm(pan, shoulder, elbow):
    p.setJointMotorControl2(robot_id, JOINT_PAN, p.POSITION_CONTROL, targetPosition=pan)
    p.setJointMotorControl2(robot_id, JOINT_SHOULDER, p.POSITION_CONTROL, targetPosition=shoulder)
    p.setJointMotorControl2(robot_id, JOINT_ELBOW, p.POSITION_CONTROL, targetPosition=elbow)

def read_ultrasonic():
    """Returns both the distance AND the ID of the object hit by the ray."""
    pos, ori = p.getBasePositionAndOrientation(robot_id)
    rot_mat = p.getMatrixFromQuaternion(ori)
    forward = [rot_mat[0], rot_mat[3], rot_mat[6]] 
    
    # Start ray slightly lower (z - 0.02) to ensure it hits the small cubes perfectly
    start = [pos[0] + forward[0]*0.25, pos[1] + forward[1]*0.25, pos[2] - 0.02]
    end = [start[0] + forward[0]*0.6, start[1] + forward[1]*0.6, start[2]]
    
    ray_results = p.rayTest(start, end)
    hit_id = ray_results[0][0]
    hit_dist = ray_results[0][2] * 0.6
    
    if hit_id != -1: 
        return hit_dist, hit_id
    return 0.6, -1

def get_camera_image():
    pos, ori = p.getBasePositionAndOrientation(robot_id)
    rot_mat = p.getMatrixFromQuaternion(ori)
    forward = np.array([rot_mat[0], rot_mat[3], rot_mat[6]])
    up = np.array([rot_mat[2], rot_mat[5], rot_mat[8]])
    
    cam_pos = np.array(pos) + forward * 0.2 + up * 0.15
    target_pos = np.array(pos) + forward * 0.5 
    target_pos[2] = 0.0  
    
    view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos, cameraTargetPosition=target_pos, cameraUpVector=up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=4.0/3.0, nearVal=0.01, farVal=10.0)
    
    _, _, rgbImg, _, _ = p.getCameraImage(160, 120, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    img_rgba = np.array(rgbImg, dtype=np.uint8).reshape((120, 160, 4))
    img_rgb = np.ascontiguousarray(img_rgba[:, :, :3])
    
    return img_rgb

def process_vision(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    M = cv2.moments(thresh)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        error = cx - 80 
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)
        return error, img
    return 0, img

# ==========================================
# 4. MAIN CONTROL LOOP
# ==========================================
state = 0
state_timer = 0
grasp_constraint = None
target_obstacle_id = None # Keeps track of which block we are picking up
move_arm(0, -1.0, 1.0)

Kp = 0.012  
Kd = 0.005  
last_error = 0

for i in range(15000): # Increased time for a full lap with 3 stops
    p.stepSimulation()
    time.sleep(1./240.)
    
    if state == 0:
        dist, hit_id = read_ultrasonic()
        
        # Check if we hit something close AND if it's one of our known obstacles
        if dist < 0.31 and hit_id in obstacle_ids:
            target_obstacle_id = hit_id
            print(f"Obstacle ID {target_obstacle_id} detected! Engaging arm...")
            p.resetBaseVelocity(robot_id, linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0])
            state = 1
            state_timer = 0
        else:
            img = get_camera_image()
            error, display_img = process_vision(img)
            
            cv2.imshow("Robot Camera Feed", cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            
            error_diff = error - last_error
            angular_z = -(Kp * error) - (Kd * error_diff)
            last_error = error
            
            forward_speed = 0.3 
            
            _, ori = p.getBasePositionAndOrientation(robot_id)
            yaw = p.getEulerFromQuaternion(ori)[2]
            vx = forward_speed * math.cos(yaw)
            vy = forward_speed * math.sin(yaw)
            
            p.resetBaseVelocity(robot_id, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, angular_z])

    elif state == 1:
        state_timer += 1
        if state_timer == 1: move_arm(0, 0.45, 0.2)
        elif state_timer > 150: state, state_timer = 2, 0

    elif state == 2:
        state_timer += 1
        if state_timer == 1:
            # Bind the gripper dynamically to whichever block we detected!
            grasp_constraint = p.createConstraint(
                parentBodyUniqueId=robot_id, parentLinkIndex=JOINT_ELBOW,
                childBodyUniqueId=target_obstacle_id, childLinkIndex=-1,
                jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                parentFramePosition=[0.25, 0, 0], childFramePosition=[0, 0, 0]
            )
            move_arm(0, -0.5, 0.5) 
        elif state_timer > 150: state, state_timer = 3, 0

    elif state == 3:
        state_timer += 1
        if state_timer == 1: move_arm(1.57, -0.5, 0.5) 
        elif state_timer > 150: state, state_timer = 4, 0

    elif state == 4:
        state_timer += 1
        if state_timer == 1: move_arm(1.57, 0.45, 0.2) 
        elif state_timer == 100: p.removeConstraint(grasp_constraint)
        elif state_timer > 150: state, state_timer = 5, 0

    elif state == 5:
        state_timer += 1
        if state_timer == 1: move_arm(0, -1.0, 1.0) 
        elif state_timer > 150: state, state_timer = 0, 0 

cv2.destroyAllWindows()
p.disconnect()