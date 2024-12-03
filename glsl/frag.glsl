#version 330 core

uniform mat4 V;

uniform vec3 Color;
uniform vec3 Light;
uniform bool useLight;

in vec3 v_norm;

out vec4 f_color;

void main() {
	if (!useLight) {
		f_color = vec4(Color, 1.0);
		return;
	}
	vec3 l = normalize(Light);
	vec3 n = normalize(v_norm);
	float lum = max(dot(n,l),0) + 0.3;
	f_color = vec4(Color * lum, 1.0);
}