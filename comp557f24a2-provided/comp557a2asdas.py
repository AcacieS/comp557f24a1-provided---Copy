## TODO Enter your name and McGill ID here

import numpy as np
from pyrr import Matrix44
from pyrr import Vector3
from pyrr import Vector4
from pyrr import Quaternion
import moderngl as mgl
import moderngl_window as mglw
from enum import Enum
import gpytoolbox

ground_name = 'ground' # ground plane is a special case for cheap shadows

object_name = { ground_name, 'monkey3', 'monkey2', 'monkey1', 'tree1', 'tree2'}
object_colors = {'monkey1': (0.97, 0.09, 0.0, 1), 
                 'monkey2': (0.06, 0.9, 0.02, 1), 
                 'monkey3': (0.07, 0.04, 0.9, 1), 
                 'tree1': (0.09, 0.67, 0.09, 1), 
                 'tree2': (0.09, 0.87, 0.09, 1), 
                 ground_name: (0.69, 0.5, 0.49, 1)}

class CameraView(Enum):
    MAIN_VIEW = 1
    LIGHT_VIEW = 2
    THIRD_PERSON_VIEW = 3
    POST_PROJECTION_VIEW = 4

class ShadowMappingSample(mglw.WindowConfig):
    title = "Assignment 2 - YOUR NAME AND ID" # TODO: ENTER YOUR NAME AND ID
    window_size = (1280, 720)
    gl_version = (3, 3)    
    aspect_ratio = 16 / 9
    resizable = True
    resource_dir = 'data'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # load the GLSL program for drawing the camera view with shadow map
        self.prog_render_scene_with_sm = self.ctx.program(
            vertex_shader =open('glsl/render_with_sm_vert.glsl').read(), 
            fragment_shader = open('glsl/render_with_sm_frag.glsl').read() )

        # load the GLSL program for drawing the depth map form the light view
        self.prog_depth = self.ctx.program(
            vertex_shader =open('glsl/depth_vert.glsl').read(), 
            fragment_shader = open('glsl/depth_frag.glsl').read() )

        # assign textures unit ID to samplers in GLSL programs
        self.prog_render_scene_with_sm['u_sampler_shadow'].value = 0
        self.prog_render_scene_with_sm['u_sampler_shadow_map_raw'].value = 1

        self.objects = {} # keep a copy of objects to draw with the camera view
        self.objects_shadow = {}  # keep a copy of objects for drawing the depth map (no normals, just positions)
        # We'll also keep a buffer of scene verts (for computing bounds) starting with the origin in list of points in scene     
        # for efficiency would be better just to keep convex hull of scene points
        self.scene_verts = np.array([[0,0,0,1]]).T   
        for name in object_name:
            v,f = gpytoolbox.read_mesh(f'data/{name}.obj')            
            n = gpytoolbox.per_vertex_normals(v, f)            
            verts = np.ones( (4, v.shape[0] ) )
            verts[:3,:] = v.T
            # append verts as additional columns to scene_verts
            self.scene_verts = np.hstack( (self.scene_verts, verts) ) 
            if name == ground_name: 
                # TODO: OBJECTIVE 3: compute the ground plane equation for cheap shadows


                
                self.ground_plane = np.array( (0,0,0,0) )

            # make vertex and index buffers
            vb = self.ctx.buffer(v.astype('f4').tobytes())
            nb = self.ctx.buffer(n.astype('f4').tobytes())
            fb = self.ctx.buffer(f.astype('i4').tobytes())           
            # create vertex array object
            self.objects[name] = self.ctx.vertex_array( self.prog_render_scene_with_sm, [(vb, '3f', 'in_position'), (nb, '3f', 'in_normal')], index_buffer=fb, mode=mgl.TRIANGLES )
            self.objects_shadow[name] = self.ctx.vertex_array( self.prog_depth, [(vb, '3f', 'in_position')], index_buffer=fb, mode=mgl.TRIANGLES )

        self.setup_box_axis_and_grid()

        shadow_size = ( 2 << 7, 2 << 7 ) # 512²
        self.tex_depth = self.ctx.depth_texture(shadow_size)
        self.tex_color_depth = self.ctx.texture(shadow_size, components=1, dtype='f4')
        self.fbo_depth = self.ctx.framebuffer( color_attachments=[self.tex_color_depth], depth_attachment=self.tex_depth )
        self.sampler_depth = self.ctx.sampler( filter=(mgl.LINEAR, mgl.LINEAR), compare_func='>=', repeat_x=False, repeat_y=False, texture=self.tex_depth )
        self.sampler_depth_map_raw = self.ctx.sampler( filter=(mgl.NEAREST, mgl.NEAREST), repeat_x=False, repeat_y=False, texture=self.tex_depth )
        self.sampler_depth.use(location=0)          # Assign the texture and sampling parameters to the texture unit
        self.sampler_depth_map_raw.use(location=1)  # Assign the texture and sampling parameters to the texture unit

        self.ctx.disable(mgl.CULL_FACE)  # have thin non-closed objets, so disable culling by default
        self.ctx.enable(mgl.DEPTH_TEST)  # always use depth test!

        # Attributes to control camera and light movement
        self.move_light = False
        self.light_rotation_angle = 0.0        
        self.light_d = 1    # Light distance multiplier (to control its distance from origin)
        self.light_h = 8    # Leight height above xz plane

        # Default camera rotations to be controlled by mouse interactions
        self.cam1_R = Matrix44.from_x_rotation(-0.1)
        self.cam1_d = 10            # Camera 1 distance from origin
        self.cam3_R = Matrix44.from_y_rotation(np.pi/2)       
        self.cam3_d = 30            # Camera 3 distance from origin
        self.cam4_R = Matrix44.from_y_rotation(-np.pi/2)
        self.cam4_d = 8             # Camera 4 distance from origin

        # Flags and values conrolled by keyboard to adjust viewing and rendering options
        self.show_CAM1 = True       # show frustum and camera location for camera 1
        self.show_CAM2 = True       # show frustum and camera location for camera 2
        self.use_perspective_shadow_map = False
        self.use_linear_filter = False  # shadow map filtering 
        self.use_culling = False        # front face culling  in light view to reduce self-shadowing
        self.cheap_shadows = False
        self.draw_depth = False         # draw the depth of fragments with respect to light position
        self.draw_depth_map = False     # draw the depth recorded from the light position
        self.use_shadow_map = True      
        self.manual_light_fov = True    # TODO: Set to False after completing OBJECTIVE 2
        self.light_fov = 45
        self.camera = CameraView.MAIN_VIEW

    def make_vao(self, vertices, indices):
        # helper function to create a vertex array object from vertex and index buffer for line geometry
        vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        ibo = self.ctx.buffer(indices.astype("i4").tobytes())
        return self.ctx.vertex_array( self.prog_render_scene_with_sm, [(vbo, '3f', 'in_position')], index_buffer=ibo, mode=mgl.LINES )

    def setup_box_axis_and_grid(self):
        # set up wire cube, axis lines, and a near plane grid
        # create cube vertices and indices for drawing a wire cube of size 2
        vertices = np.array([-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1], dtype='f4')
        # create cube edges
        indices = np.array([ 0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7], dtype='i4')
        self.cube_vao = self.make_vao(vertices, indices)
        self.line_x_vao = self.make_vao(np.array([0,0,0,1,0,0]), np.array([0,1]))
        self.line_y_vao = self.make_vao(np.array([0,0,0,0,1,0]), np.array([0,1]))
        self.line_z_vao = self.make_vao(np.array([0,0,0,0,0,1]), np.array([0,1]))
        n = 8
        vertices = -np.ones( (3,n*4), dtype='f4')
        coords = np.linspace(-1, 1, n)
        vertices[0:2,1::2] = 1
        vertices[0,0:n*2:2] = coords
        vertices[0,1:n*2:2] = coords
        vertices[1,n*2::2] = coords
        vertices[1,n*2+1::2] = coords
        indices = np.array( range(0,40), dtype='i4' )
        self.grid_vao = self.make_vao(vertices.T, indices)

    def draw_axis(self):
        # draw a coordinate frame with red green blue axis colours (note that lighting should be disabled when using  this function)
        self.prog_render_scene_with_sm['u_color'] = (1,0,0,1)
        self.line_x_vao.render()
        self.prog_render_scene_with_sm['u_color'] = (0,1,0,1)
        self.line_y_vao.render()
        self.prog_render_scene_with_sm['u_color'] = (0,0,1,1)
        self.line_z_vao.render()

    # TODO: OBJECTIVE 1: implement trackball rotations and apply to the current camera view
    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):        
        # TODO: compute a rotation update and apply, as opposed to doing nothing below!



        if self.camera == CameraView.MAIN_VIEW:
            self.cam1_R = self.cam1_R
        if self.camera == CameraView.THIRD_PERSON_VIEW:
            self.cam3_R = self.cam3_R
        if self.camera == CameraView.POST_PROJECTION_VIEW:
            self.cam4_R = self.cam4_R
    
    def mouse_scroll_event(self, x_offset: float, y_offset: float):        
        # dolly the camera by increasing and decreasing the distance from the origin
        if self.camera == CameraView.MAIN_VIEW:
            self.cam1_d = self.cam1_d * np.power( 1.1, y_offset )
        if self.camera == CameraView.THIRD_PERSON_VIEW:
            self.cam3_d = self.cam3_d * np.power( 1.1, y_offset )
        if self.camera == CameraView.POST_PROJECTION_VIEW:
            self.cam4_d = self.cam4_d * np.power( 1.1, y_offset )

    def key_event(self, key, action, modifiers):
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.NUMBER_1:                       # Control view in the applicaiton window
                self.camera = CameraView.MAIN_VIEW
            if key == self.wnd.keys.NUMBER_2:
                self.camera = CameraView.LIGHT_VIEW
            if key == self.wnd.keys.NUMBER_3:
                self.camera = CameraView.THIRD_PERSON_VIEW
            if key == self.wnd.keys.NUMBER_4: 
                self.camera = CameraView.POST_PROJECTION_VIEW
            if key == self.wnd.keys.F:                       
                self.use_linear_filter = not self.use_linear_filter # shadow map filtering
            if key == self.wnd.keys.C:
                self.use_culling = not self.use_culling             # front face culling  in light view to reduce self-shadowing
            if key == self.wnd.keys.O:
                self.cheap_shadows = not self.cheap_shadows         # cheap shadows using a planar projection
            if key == self.wnd.keys.P:                              
                self.use_perspective_shadow_map = not self.use_perspective_shadow_map   
            if key == self.wnd.keys.U:
                self.use_shadow_map = not self.use_shadow_map
            if key == self.wnd.keys.D:                          # cycle through drawing depth or depth map
                if ( self.draw_depth ):
                    self.draw_depth = False
                    self.draw_depth_map = True
                elif ( self.draw_depth_map ):
                    self.draw_depth = False
                    self.draw_depth_map = False
                else:
                    self.draw_depth = True                            
            if key == self.wnd.keys.A:                  # Toggle animate light
                self.move_light = not self.move_light
            if key == self.wnd.keys.MINUS:              # Adjust light distance and height 
                self.light_d = self.light_d / 1.1
            if key == self.wnd.keys.EQUAL:
                self.light_d = self.light_d * 1.1
            if key == self.wnd.keys.COMMA:              
                self.light_h = self.light_h / 1.1
            if key == self.wnd.keys.PERIOD:
                self.light_h = self.light_h * 1.1       
            if key == self.wnd.keys.E: # Toggle display of main camera (i.e., "eye" view)
                self.show_CAM1 = not self.show_CAM1
            if key == self.wnd.keys.L: # Toggle display of light camera
                self.show_CAM2 = not self.show_CAM2
            if key == self.wnd.keys.M:                              # Manual light FOV control 
                self.manual_light_fov = not self.manual_light_fov   # (this only makes sense in the absence of tilting and shifting the light view)
            if key == self.wnd.keys.NUMBER_9:
                self.light_fov = self.light_fov / 1.1
                if self.light_fov < 1: self.light_fov = 1
            if key == self.wnd.keys.NUMBER_0:
                self.light_fov = self.light_fov * 1.1
                if self.light_fov > 179: self.light_fov = 179

    # TODO: OBJECTIVE 2a: implement a method to compute the near and far planes of the view frustum
    def compute_nf_from_view(self, V, verts):
        n,f = 4, 12




        return n,f

    # TODO: OBJECTIVE 2b: implement a method to compute bounds for the frustum
    def compute_lrtb_for_projection(self, V, n, f, verts):        
        l,r,t,b = -1,1,1,-1
        
        
        
        
        
        return l,r,t,b

    def render(self, time: float, _frame_time: float):

        # Update the light position, if it is animating, and set the GLSL program uniform to the world position
        if self.move_light:
            self.light_rotation_angle += _frame_time
        light_pos = Matrix44.from_y_rotation(self.light_rotation_angle) * Vector3((0, self.light_d *self.light_h, self.light_d *6)) 
        self.prog_render_scene_with_sm['u_light'] = light_pos

        # Setup camera 1, the 1st person view
        V1 = Matrix44.from_translation((0,0,-self.cam1_d), dtype='f4') * self.cam1_R        
        n,f = self.compute_nf_from_view(V1, self.scene_verts)
        if n < 0: n = 1e-3 # make sure near and far stay in front of camera
        P1 = Matrix44.perspective_projection( 45.0, self.aspect_ratio, n, f )

        # Setup camera 2, the light view (possibly to be used post camera 1 projection) 
        if self.use_perspective_shadow_map:
            # TODO: OBJECTIVE 4: Implement the persepctive shadow map!
            V2 = Matrix44.identity(dtype='f4') # temporary value... 
            P2 = Matrix44.identity(dtype='f4') # temporary value... 




        else:
            # Create light view has arbitrary up and is direted at origin
            V2 = Matrix44.look_at( light_pos, target=(0, 0, 0), up=(0.0, 1.0, 0.0) )            
            n,f = self.compute_nf_from_view(V2, self.scene_verts)
            if self.manual_light_fov:
                P2 = Matrix44.perspective_projection(self.light_fov, 1, n, f)
            else:
                l,r,t,b = self.compute_lrtb_for_projection(V2, n, f, self.scene_verts)
                P2 = Matrix44.perspective_projection_bounds(l,r,t,b,n,f)
 
        # Setup camera 3, the 3rd person view
        V3 = Matrix44.from_translation((0,0,-self.cam3_d), dtype='f4') * self.cam3_R
        P3 = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000)

        # Setup camera 4, the post perspective view
        V4 = Matrix44.from_translation((0,0,-self.cam4_d), dtype='f4') * self.cam4_R
        P4 = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000)

        # Set the camera based on user selection.  We set cam_mv to help with specular lighting 
        if self.camera == CameraView.MAIN_VIEW:
            cam_mvp = P1 * V1
            cam_mv = V1
        elif self.camera == CameraView.LIGHT_VIEW:
            cam_mvp = P2 * V2
            cam_mv = V2
            # TODO: OBJECTIVE 4: ADJUST THIS IF USING PERSPECTIVE SHADOW MAP    



        elif self.camera == CameraView.THIRD_PERSON_VIEW:
            cam_mvp = P3 * V3
            cam_mv = V3
        elif self.camera == CameraView.POST_PROJECTION_VIEW:            
            cam_mvp = P4 * V4 * P1 * V1
            cam_mv = V4 * P1 * V1
        
        # Set the the MVP transformation for the main view
        # note that the lighting is done in world coordinates, so object normals are used without transformaiton
        # and the M in the MVP transformation is the identity matrix
        self.prog_render_scene_with_sm['u_mvp'].write( cam_mvp.astype('f4').tobytes() )
        
        # Lighting computations are done in world coordinates, so we must provide the camera position in world coordinates to GLSL
        # to permit the view direction to be computed at each fragment
        pos = cam_mv.inverse * Vector4((0,0,0,1)) # transform origin of camera frame to world with inverse viewing transformation
        pos = Vector3(pos.xyz) / pos.w            # normalize by w 
        self.prog_render_scene_with_sm['u_cam_pos'].write( pos.astype('f4').tobytes() )
        
        # set up light camear view and projection, along with the light space transformation to use during rendering with shadows
        mvp_light = P2 * V2
        # TODO: OBJECTIVE 4: ADJUST THIS IF USING PERSPECTIVE SHADOW MAP




        # Windowing transform to convert [-1,1]^3 canonical viewing volume to texture coordinates in [0,1] range
        W = Matrix44.from_translation((0.5, 0.5, 0.5), dtype='f4') * Matrix44.from_scale((0.5, 0.5, 0.5), dtype='f4')
        light_space_transform = W * mvp_light        
        
        # Set the MVP for the light view, along with the tranformation to use for rendering with shadows
        self.prog_render_scene_with_sm['u_light_space_transform'].write( light_space_transform.astype('f4').tobytes() )
        self.prog_depth['u_mvp'].write(mvp_light.astype('f4').tobytes())

        ########################################################
        # pass 1: render shadow-map (depth framebuffer -> texture) from light view
        self.fbo_depth.use()
        self.fbo_depth.clear(1,1,1,1, depth = 1)   
        self.ctx.depth_func = '<'
        self.ctx.front_face = 'ccw'

        # TODO: OBJECTIVE 4 and 5: Adjustments to depth clear, depth test, front face definitions, and culling may be necessary when using perspective shadow map
     


        if self.use_culling:
            self.ctx.enable(mgl.CULL_FACE)
            self.ctx.cull_face = 'front' 

        for name in object_name:
            self.objects_shadow[name].render()       

        self.ctx.cull_face = 'back'
        self.ctx.disable(mgl.CULL_FACE)

        ########################################################
        # pass 2: render the scene with the specified camera and viewing parameters
        self.ctx.screen.use()
        self.ctx.clear(0,0,0,0,depth=1)
        self.ctx.depth_func = '<'

        if self.use_linear_filter:
            self.sampler_depth.filter = (mgl.LINEAR, mgl.LINEAR)      
        else:
            self.sampler_depth.filter = (mgl.NEAREST, mgl.NEAREST)            
        
        self.prog_render_scene_with_sm['u_draw_depth'] = self.draw_depth
        self.prog_render_scene_with_sm['u_draw_depth_map'] = self.draw_depth_map
        self.prog_render_scene_with_sm['u_use_shadow_map'] = (not self.camera == CameraView.LIGHT_VIEW) and self.use_shadow_map
        self.prog_render_scene_with_sm['u_use_lighting'] = True

        # change definition of front face for post projection view, and the light view when using a post projection shadow map.
        self.ctx.front_face = 'ccw'
        if self.camera == CameraView.POST_PROJECTION_VIEW:
            # Projection flips handedness, so we need to change our definition of front faces for lighting
            self.ctx.front_face = 'cw' 
        
        # TODO: OBJECTIVE 4 and 5: extra work will be needed to get a correct lightview when using a perspective shadow map
        #if self.camera == CameraView.LIGHT_VIEW and self.use_perspective_shadow_map:
         



                        
        for name in object_name:
            self.prog_render_scene_with_sm['u_color'] = object_colors[name]            
            self.objects[name].render()       

        # TODO: OBJECTIVE 3: You'll want to draw cheap shadows if the flag is set





        # reset the depth function and front face definition
        self.ctx.depth_func = '<'
        self.ctx.front_face = 'ccw'
        
        ######################################
        # draw frustums, and coordinate frames
                
        self.prog_render_scene_with_sm['u_draw_depth'] = False
        self.prog_render_scene_with_sm['u_draw_depth_map'] = False
        self.prog_render_scene_with_sm['u_use_shadow_map'] = False
        self.prog_render_scene_with_sm['u_use_lighting'] = False
        self.ctx.enable(mgl.BLEND)

        if self.show_CAM2:
            if self.use_perspective_shadow_map:
                M = cam_mvp * mvp_light.inverse
                self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
                self.prog_render_scene_with_sm['u_color'] = (1,1,0,0.5)            
                self.cube_vao.render()
                self.grid_vao.render()
                M = cam_mvp * (V2 * P1 * V1).inverse
                self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
                self.draw_axis()
            else:
                M = cam_mvp * mvp_light.inverse
                self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
                self.prog_render_scene_with_sm['u_color'] = (1,1,0,0.5)            
                self.cube_vao.render()
                self.grid_vao.render()
                M = cam_mvp * V2.inverse
                self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
                self.draw_axis()

        if self.show_CAM1:
            M = cam_mvp * (P1*V1).inverse            
            self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
            self.prog_render_scene_with_sm['u_color'] = (1,1,1,0.5) # make light frustum yellow
            self.cube_vao.render()
            M = cam_mvp * V1.inverse 
            self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
            self.draw_axis()

        if self.camera == CameraView.POST_PROJECTION_VIEW:        
            M = P4 * V4  
            self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
            self.draw_axis()
        else:
            # draw world frame
            M = cam_mvp
            self.prog_render_scene_with_sm['u_mvp'].write(M.astype('f4').tobytes())
            self.draw_axis()

        self.ctx.disable(mgl.BLEND)

ShadowMappingSample.run()