# Acacie Song 261182381

import moderngl_window as mglw
import moderngl as mgl
import numpy as np
import random
#idk if can import that?
from scipy.spatial.transform import Rotation as R
#maybe not use it
from scipy.spatial.transform import Slerp

# list of rotation interpolation types to display in this assignment
rotation_type = [ 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', 'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ', 'RL', 'QL', 'QS', 'QLN', 'QLNF', 'QSF','A','B' ]
# list of only the Euler rotation orders, i.e., "if rotorder in euler_orders:" will be useful
euler_orders = [ 'XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX', 'XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ']

# dict mapping letters rotation types letters to colors for labels
letter_colors = { 
            'X': np.array((1,0.2,0.2),dtype='f4'),
            'Y': np.array((0.2,1,0.2),dtype='f4'),
            'Z': np.array((0.4,0.4,1),dtype='f4'),
            'R': np.array((.7,0.9,0.1),dtype='f4'),
            'Q': np.array((0.3,1,1),dtype='f4'),
            'L': np.array((1,0.3,1),dtype='f4'),
            'S': np.array((1,.6,0.1),dtype='f4'),
            'N': np.array((1,.4,.4),dtype='f4'),
            'F': np.array((.5,.5,.5),dtype='f4'),
            'A': np.array((.9,.9,.9),dtype='f4'),
            'B': np.array((.9,.9,.9),dtype='f4')
            }

## Next follows some helper functions you may find useful. Feel free to add more as needed!

def normalize(q):
    return q/np.linalg.norm(q)

def rand_unit_quaternion():
    # generate a random unit length quaternion
    q = np.array([random.gauss(0, 1) for i in range(4)])    
    return normalize(q)

def quaternion_random_axis_angle(angle): 
    # generate a rotation of the given angle about a random axis and return as a unit quaternion
    axis = np.array([random.gauss(0, 1) for i in range(3)])
    axis = axis/np.linalg.norm(axis)
    q = np.append( np.cos(angle/2), np.sin(angle/2)*axis)
    return q

def rand_180_quaternion():
    # generate a random 180 degree rotation and return as a unit quaternion
    return quaternion_random_axis_angle(np.pi)

def quaternion_multiply(q1, q2):
    # return the product of two quaternions
    q1q2 = np.zeros(4)
    q1q2[0] = q1[0]*q2[0] - np.dot(q1[1:],q2[1:])
    q1q2[1:] = q1[0]*q2[1:] + q2[0]*q1[1:] + np.cross(q1[1:],q2[1:])
    return q1q2

def quat_to_R(q):
    # convert a unit quaternion to a3x3 rotation matrix
    return np.array([[1-2*(q[2]**2+q[3]**2), 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])],
                        [2*(q[1]*q[2]+q[0]*q[3]), 1-2*(q[1]**2+q[3]**2), 2*(q[2]*q[3]-q[0]*q[1])],
                        [2*(q[1]*q[3]-q[0]*q[2]), 2*(q[2]*q[3]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)]])

def create_perspective_projection(fovy, aspect, near, far):
    # create a perspective projection matrix (will see this in class soon)
    f = 1.0 / np.tan(fovy * np.pi / 180 / 2.0)
    return np.array([[f / aspect, 0, 0, 0],
                     [0, f, 0, 0],
                     [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                     [0, 0, -1, 0]], dtype='f4')

def create_orthographic_projection(left, right, bottom, top, near, far):
    # create an orthographic projection matrix (will see this in class soon)
    return np.array([[2 / (right - left), 0, 0, -(right + left) / (right - left)],
                     [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                     [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                     [0, 0, 0, 1]], dtype='f4')

def create_look_at(eye, target, up):
    # create a viewing transformation matrix
    eye = np.array(eye, dtype='f4')
    target = np.array(target, dtype='f4')
    up = np.array(up, dtype='f4')
    w = normalize( eye - target )
    u = normalize( np.cross(up, w) )
    v = np.cross(w, u)
    # return inverse of rigid transformation [ u, v, w, eye ]
    return np.array([[u[0], u[1], u[2], -np.dot(u, eye)],
                     [v[0], v[1], v[2], -np.dot(v, eye)],
                     [w[0], w[1], w[2], -np.dot(w, eye)],
                     [0, 0, 0, 1]], dtype='f4')

def near_angle(x, a, b, t):
    m = np.pi
    return abs(x % m - a) <= t or abs(x % m - b) <= t


class InterpolationSceneNode:
	# This class is help organize drawing of a just one monkey head with a given rotation type
    # at a given position within the larger scene. It is called a scene node because it acts
    # much like a node in a scene graph, except there are no child notes, ie., we are not actually 
    # using a scene graph here.

    def __init__(self,pos,rotorder,prog,vao):
        self.rotorder = rotorder    # the rotation order for this node
        self.pos = pos              # the position of where to draw in the world
        self.prog = prog            # the shader program to use for drawing
        self.vao = vao              # the vertex array objects to use for drawing

    def draw_label(self):
        #center
        char_spacing = 0.5
        label_width = len(self.rotorder) * char_spacing - char_spacing

        #Draw label below monkey head
        translation = np.array([char_spacing, 0, 0], dtype ='f4') #space between letters 
        label_pos = self.pos + np.array([-label_width/2, -1.5, 0]) #1.5 unit below monkey and center
        
        #rotation
        for i, char in enumerate(self.rotorder):
            M = np.eye(4, dtype ='f4')
            M[0:3, 3] = label_pos + i * translation #translation

            M[1,1] = 0 #small thickness in y-direction
            #Rotating to lie from x-z to x-y plane. So be viewed from neg z-direction
            M[1,2] = -1
            M[2,1] = 1 
            M[2,2] = 0
            
            #Draw it
            self.prog['M'].write(M.T.flatten())
            self.prog['Color'] = letter_colors[char]
            self.prog['useLight'] = False
            self.vao[char].render()

    #Set A and B
    def set_rotation_euler(self, i):
        #Set angle for each euler_orders
        for item in euler_orders:
            if(i==0):
                self.A_euler_each.append(self.A_rotation.as_euler(item))
            else:
                self.B_euler_each.append(self.B_rotation.as_euler(item))
    
    def set_rotation(self, q, i): #only monkey head rotate euler angles.
        
        #Quarternion of A or B to rotation matrix
        euler_matrix = quat_to_R(q)
        
        #convert rotation matrix to euler rotation
        euler_rotation = R.from_matrix(euler_matrix)
        
        #Set A and B
        if(i==0):
            self.A_matrix = euler_matrix
            self.A_quaternion = q
            self.A_rotation = euler_rotation
            self.A_euler_each = []
        else:
            self.B_matrix = euler_matrix
            self.B_quaternion = q
            self.B_rotation = euler_rotation
            self.B_euler_each = []
            
        self.set_rotation_euler(i)
        ## TODO set the target rotation from the provided quaternion, where i is 0 or 1 for target A or B respectively
        return 
    
    #Euler rotation
    def interpolate_function(self,t):
        self.ipolate_angles = (1-t) * self.A_euler_each[self.rotation_index] + t * self.B_euler_each[self.rotation_index]
        
    def euler_rotation(self,t):
        self.interpolate_function(t)
        ipolate_rotation = R.from_euler(self.rotorder, self.ipolate_angles)
        ipolate_matrix = ipolate_rotation.as_matrix()

        return ipolate_matrix
    

    #Gimbal Lock for Euler
    def check_gimbal_lock(self):
       
        tolerance = np.radians(5)
        is_gimbal_lock = False

        if self.rotorder in euler_orders:
            
            if self.rotorder in ['XYX', 'XZX', 'YXY', 'YZY', 'ZXZ', 'ZYZ']:
                
                if near_angle(self.ipolate_angles[1],0,np.pi,tolerance):
                    is_gimbal_lock = True
            else:
                
                if near_angle(self.ipolate_angles[1],-np.pi/2,np.pi/2,tolerance):
                    is_gimbal_lock = True

        return is_gimbal_lock
   
    #Fix Euler           
    def fix_for_shorter_Euler_interpolation(self):
        ## TODO check if any target angles for Euler rotation are interpolating more than 180 degrees and add 2*pi to the smaller angle to produce a shorter interpolation
        if self.rotorder in euler_orders: 
            
            for i in range(0,len(self.ipolate_angles)): 
                diff = self.A_euler_each[self.rotation_index][i]-self.B_euler_each[self.rotation_index][i]
                #if more than 180 degree
                if(np.abs(diff)>np.pi):
                    #which smaller
                    if self.A_euler_each[self.rotation_index][i]<self.B_euler_each[self.rotation_index][i]:
                        self.A_euler_each[self.rotation_index][i]+= 2*np.pi
                    else:
                        self.B_euler_each[self.rotation_index][i]+= 2*np.pi
                        
                                        
            
    #Other rotation   
    def RL(self, t):
        ipolate_matrix_RL = (1-t) * self.A_matrix + t * self.B_matrix
        return ipolate_matrix_RL
    
    def QL(self, t):
        ipolate_angles_quat = (1-t) * self.A_quaternion + t * self.B_quaternion
        ipolate_matrix_ql = quat_to_R(ipolate_angles_quat)
        return ipolate_matrix_ql

    def QS(self, t, B_quaternion):
        angle_spherical = np.arccos(np.dot(self.A_quaternion, B_quaternion))
        if angle_spherical == 0 or angle_spherical == np.pi: #if = 0 can't divide by 0 to not crash.
            slerp = self.A_quaternion
        else:
            slerpA = np.sin((1-t)*angle_spherical)/np.sin(angle_spherical)* self.A_quaternion
            slerpB = np.sin((t)*angle_spherical)/np.sin(angle_spherical)* B_quaternion
            slerp = slerpA + slerpB
        spherical_matrix = quat_to_R(slerp)
            
        return spherical_matrix
    def QLN(self, t, B_quaternion):
        ipolate_ang_quat = (1-t) * self.A_quaternion + t * B_quaternion
        ipolate_ang_quat_normalize = normalize(ipolate_ang_quat)
        ipolate_matrix_ql = quat_to_R(ipolate_ang_quat_normalize)
        return ipolate_matrix_ql
    def fix_quaternion_rotation(self, B_quaternion):
        if(np.dot(self.A_quaternion, self.B_quaternion)<0):
                B_quaternion = -self.B_quaternion
        return B_quaternion
        
    def QSF(self, t, B_quaternion):
        B_quaternion = self.fix_quaternion_rotation(B_quaternion)
        return self.QS(t, B_quaternion) 
    
    
    def QLNF(self, t, B_quaternion):
        B_quaternion = self.fix_quaternion_rotation(B_quaternion)
        return self.QLN(t, B_quaternion)
   

    def render(self, t):
        # interpolation parameter t is a float between 0 and 1, 
        self.draw_label() ## TODO draw the label for the specified rotaiton type
        self.rotation_index = rotation_type.index(self.rotorder)
        
        M = np.eye(4,dtype='f4')
        M[0:3,3] = self.pos

        ## TODO compute the interpolation between target orientations and use it to set the rotation
        #Euler Rotation
        if self.rotorder in euler_orders:
            M[0:3,0:3] = self.euler_rotation(t)

        #Others
        if self.rotorder == "RL":
            M[0:3,0:3] = self.RL(t)
        if self.rotorder == "QL":
            M[0:3,0:3] = self.QL(t)
        if self.rotorder == "QS":
            M[0:3,0:3] = self.QS(t, self.B_quaternion) 
        if self.rotorder == "QLN":
            M[0:3,0:3] = self.QLN(t, self.B_quaternion)
        if self.rotorder == "QLNF":
            M[0:3,0:3] = self.QLNF(t, self.B_quaternion)
        if self.rotorder == "QSF":
            M[0:3,0:3] = self.QSF(t, self.B_quaternion)
        if self.rotorder == "A":
            M[0:3,0:3] = self.A_matrix
        if self.rotorder == "B":
            M[0:3,0:3] = self.B_matrix

        #Draw Monkey
        self.prog['M'].write( M.T.flatten() ) # transpose and flatten to get in Opengl Column-major format
        
        if self.check_gimbal_lock()==True:
            self.prog['Color'] = (1,0,0)
        else:
            self.prog['Color'] = (0.5,0.5,0.5)
        self.prog['useLight'] = True
        self.vao['monkey'].render()


class HelloInterpolation(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Assignment 1 - Acacie Song 261182381"
    window_size = (1280, 720)
    aspect_ratio = 16.0 / 9.0
    resizable = True
    resource_dir = 'data'

    def setup_wire_box(self):
        # create cube vertices and indices for drawing a wire cube of size 2
        vertices = np.array([
            -1.0, -1.0, -1.0,  
             1.0, -1.0, -1.0,
             1.0,  1.0, -1.0, 
            -1.0,  1.0, -1.0,
            -1.0, -1.0,  1.0, 
             1.0, -1.0,  1.0,
             1.0,  1.0,  1.0, 
            -1.0,  1.0,  1.0,
        ], dtype='f4')
        # create cube edges
        indices = np.array([ 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7], dtype='i4')
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        ibo= self.ctx.buffer(indices.astype("i4").tobytes())
        # note: nothing for the "normal" attribute, as we will ingore it with the lighting disabled
        self.cube_vao = self.ctx.vertex_array(self.prog, [(vbo, '3f', 'in_position')], index_buffer=ibo, mode=mgl.LINES)

    def draw_wire_box(self):
        self.cube_vao.render()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        random.seed(0) # set random seed for deterministic reproducibility
        self.prog = self.ctx.program( 
            vertex_shader =open('glsl/vert.glsl').read(), 
            fragment_shader = open('glsl/frag.glsl').read() )
        self.prog['Light'] = (10,25,25) # set the light direction

        # load obj files from resource_dir for drawing the monkey and letters
        self.scene = {}
        self.vao = {}
        for a in ['monkey','X','Y','Z','R','L','N','Q','S','F','A','B']:
            self.scene[a] = self.load_scene(a+".obj")
            self.vao[a] = self.scene[a].root_nodes[0].mesh.vao.instance(self.prog)        
        self.setup_wire_box()

        # setup a grid of different interpolation sub-scenes, nicely spaced for viewing
        self.nodes = []
        for i in range(len(rotation_type)):
            c = 4*((i % 5) - 2)
            r = 4*(-(i // 5) + 1.75)
            self.nodes.append( InterpolationSceneNode( np.array([c,r,0]), rotation_type[i], self.prog, self.vao ) )

        # initialize the target orientations
        self.A = np.array([1,0,0,0])
        self.set_new_rotations(self.A,0)
        self.B = np.array([1,0,0,0])
        self.set_new_rotations(self.B,1)

        # Setup the 1st person and 3rd person viewing and projection matrices        
        self.V_1st_person = create_look_at(eye=(0, 0, 40), target=(0, 0, 0), up=(0, 1, 0))
        self.V_3rd_person = create_look_at(eye=(30, 10, 55), target=(0, 0, 20), up=(0, 1, 0))

        self.P_1st_person_perspective = create_perspective_projection(25.0, self.aspect_ratio, 10, 45)
        self.P_1st_person_orthographic = create_orthographic_projection(-16, 16,-9,9,37,45)        
        self.P_3rd_person = create_perspective_projection(40.0, self.aspect_ratio, 10, 1000)

        self.view = '1st_person'        
        self.P_1st_person = self.P_1st_person_perspective.copy()
        self.V_current = self.V_1st_person.copy()
        self.P_current = self.P_1st_person.copy()

    def set_new_rotations(self, target, i): 
        # set the target rotations for all the nodes, where i is 0 or 1 for target A or B respectively
        for b in self.nodes: b.set_rotation(target,i)
        
    def key_event(self, key, action, modifiers):
        # handle key events, see the assignment spec for additional description
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.A:                
                self.A = rand_unit_quaternion()
                self.set_new_rotations( self.A, 0 )
                
            if key == self.wnd.keys.B:                
                self.B = rand_unit_quaternion()
                self.set_new_rotations( self.B, 1 )
            if key == self.wnd.keys.Z:
                q = quaternion_random_axis_angle( np.pi/180*3 )
                q = quaternion_multiply( q, self.A )
                self.B = -q
                self.set_new_rotations( self.B, 1 )
            if key == self.wnd.keys.X:                
                q = quaternion_random_axis_angle( np.pi )
                self.B = quaternion_multiply( q, self.A )
                self.set_new_rotations( self.B, 1 )
            if key == self.wnd.keys.I:                
                self.A = np.array([1,0,0,0])
                self.B = np.array([1,0,0,0])
                self.set_new_rotations( self.A, 0 )
                self.set_new_rotations( self.B, 1 )
            if key == self.wnd.keys.F:
                for node in self.nodes:
                    node.fix_for_shorter_Euler_interpolation()
            if key == self.wnd.keys.NUMBER_1:
                self.view = '1st_person'	                
            elif key == self.wnd.keys.NUMBER_3:
                self.view = '3rd_person'
            elif key == self.wnd.keys.P:
                self.P_1st_person = self.P_1st_person_perspective
                self.projection_1st_person = 'perspective'
            elif key == self.wnd.keys.O:
                self.P_1st_person = self.P_1st_person_orthographic
                self.projection_1st_person = 'orthographic'

    def draw_viewer_as_monkey(self):
        inverse_matrix = np.linalg.inv(self.V_1st_person)

        #Draw 
        M = np.eye(4,dtype='f4')
        M[:4,:4] = inverse_matrix[0:4,0:4]
        angle = np.pi

        #Rotation to look correct angle. negative z axis
        R = np.array([[np.cos(angle),0,np.sin(angle),0],[0,1,0,0],[-np.sin(angle),0,np.cos(angle),0],[0,0,0,1]], dtype ='f4') 
        M = M @ R
        
        #Draw a Monkey
        self.prog['M'].write( M.T.flatten() ) # transpose and flatten to get in Opengl Column-major format
        self.prog['useLight'] = True
        self.vao['monkey'].render()
    def draw_frustum(self):
        #Size, transform, rotation
        inverse_matrix = np.linalg.inv(self.V_1st_person)
        inverse_projection = np.linalg.inv(self.P_1st_person)
        frustum_transform = inverse_matrix @ inverse_projection

        #draw frustum box
        M = np.eye(4,dtype='f4')
        M[:4,:4] = frustum_transform[:4,:4]
        self.prog['M'].write( M.T.flatten() )
        self.prog['useLight'] = False
        self.draw_wire_box()
        self.prog['Color'] = (1,1,1)

    def render(self, time, frame_time):
        self.ctx.clear(0,0,0)
        self.ctx.enable(mgl.DEPTH_TEST)

        # Interpolate the current and target viewing and projection matrices
        # Here we use a simple exponential decay filter to smooth the interpolation
        # (i.e., different than linear interpolation used for the rotations, and
        # likewise different than ease-in/ease-out interpolation that one might use)
        if self.view == '1st_person':
            V_target = self.V_1st_person
            P_target = self.P_1st_person            
        elif self.view == '3rd_person':
            V_target = self.V_3rd_person
            P_target = self.P_3rd_person

        self.V_current = self.V_current * 0.9 + V_target*0.1
        self.P_current = self.P_current * 0.9 + P_target*0.1

        self.prog['P'].write( self.P_current.T.flatten() )
        self.prog['V'].write( self.V_current.T.flatten() )

        # We create an interpolation parameter based on the time in seconds, 
        # going from 0 to 1, pausing for a second, then back to zero, and pausing again, 
        # and repeating in a 4 second cycle.
        time_mod_4 = time%4
        if time_mod_4 < 1: t = time_mod_4
        elif time_mod_4 < 2: t = 1
        elif time_mod_4 <3: t = 3 - time_mod_4
        else: t = 0

        # draw the interpolation sub-scenes
        for b in self.nodes:
            b.render(t)
       
        ## TODO you'll also need some code to draw the viewer and frustum!
        ## Hint: use the `self.prog['useLight']` uniform to switch between lighting and non-lighting modes
        self.draw_viewer_as_monkey()
        self.draw_frustum()


HelloInterpolation.run()