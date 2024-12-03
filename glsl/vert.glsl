#version 330 core

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_vert;
out vec3 v_norm;

void main() {
	mat3 MinvT = transpose(inverse(mat3(M)));
	vec4 v = V * M * vec4(in_position, 1.0);
	v_vert = v.xyz;
	v_norm = mat3(V) * MinvT * in_normal;
	gl_Position = P * v;
}