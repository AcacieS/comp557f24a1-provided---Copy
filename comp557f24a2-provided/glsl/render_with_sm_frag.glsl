#version 330

uniform vec3 u_light; // light position in world coordinates
uniform vec3 u_cam_pos; // camera position in world coordinates
uniform vec4 u_color; // k_d material parameter, otherwise, the color to draw if lighting disabled

uniform sampler2DShadow u_sampler_shadow;
uniform sampler2D       u_sampler_shadow_map_raw;

uniform bool u_use_lighting;
uniform bool u_use_shadow_map;
uniform bool u_draw_depth;
uniform bool u_draw_depth_map;

uniform bool u_invert_shadow_test;

in vec3 v_vert; // vertex position in world coordiantes
in vec3 v_norm; // normal in world coordinates
in vec4 v_shadow_coord;

out vec4 f_color;

const float LIGHT_AMBIENT = 0.3;
const vec4 LIGHT = vec4( 0.8, 0.8, 0.8, 1.0 );
const vec4 k_s   = vec4( 1, 1, 1, 1 );

float compute_visibility(in float cos_theta) {
	vec2 shadow_coord_ls = v_shadow_coord.xy / v_shadow_coord.w; // normalize for shadow coordinates in light space texture
	float bias = 0.005 * tan(acos(cos_theta)); // bias according to the slope (this function doesn't make a lot of sense)
	bias = clamp(bias, 0, 0.01) * (u_invert_shadow_test ? -1 : 1);	
	float z_from_cam = v_shadow_coord.z / v_shadow_coord.w - bias;
	vec3 shadow_coord = vec3( shadow_coord_ls, z_from_cam );
	float shadow_value = texture( u_sampler_shadow, shadow_coord );	
	if ( u_invert_shadow_test ) {
		shadow_value = 1.0 - shadow_value;
	}
	return 1.0 - shadow_value;
}
				
void main() {
	if ( u_use_lighting == false ) {
		f_color = u_color; 
		return;
	}
	if ( u_draw_depth == true ) {
		f_color = vec4( v_shadow_coord.z / v_shadow_coord.w );
		return;
	}
	if ( u_draw_depth_map == true ) {
		vec2 shadow_coord_ls = v_shadow_coord.xy / v_shadow_coord.w;			
		float d = texture( u_sampler_shadow_map_raw, shadow_coord_ls ).r;
		f_color = vec4( d,d,d,1 );
		return;
	}

	// Setup vectors for computing lighting, and flip the normal if the face is backfacing
	// Note that all these vectors are in world coordinates
	vec3 normal_vector = normalize( v_norm ) * (gl_FrontFacing ? 1 : -1);
	vec3 light_vector = normalize( u_light - v_vert );
	vec3 view_vector = normalize( u_cam_pos - v_vert );
	vec3 half_vector = normalize( light_vector + view_vector );

	// Compute lighting contributions
	float cos_theta = dot( light_vector, normal_vector );
	vec4 Ld = u_color * LIGHT * max( cos_theta, 0.0 );	
	vec4 Ls = k_s * LIGHT *  pow( max( dot( half_vector, normal_vector ), 0.0 ), 50.0 );
	vec4 La = u_color * LIGHT_AMBIENT;
	
	if ( u_use_shadow_map ) {
		 f_color = compute_visibility( cos_theta ) * (Ld + Ls) + La;
		 return;
	}
	f_color = Ld + Ls + La;
}