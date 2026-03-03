precision highp float;
precision mediump sampler3D;

uniform vec3 u_size;
uniform int u_renderstyle;
uniform float u_renderthreshold;
uniform vec2 u_clim;

uniform sampler3D u_data;
uniform sampler2D u_cmdata;

varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

// The maximum distance through our rendering volume is sqrt(3).
const int MAX_STEPS = 887;	// 887 for 512^3, 1774 for 1024^3
const int REFINEMENT_STEPS = 2;
const float relative_step_size = 1.0;
const float shininess = 40.0;

void cast_mip(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray);
void cast_volume(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray);

float sample1(vec3 texcoords);
vec4 apply_colormap(float val);

float sample1(vec3 texcoords) {
        return texture(u_data, texcoords.xyz).r;
}

void main() {
        // Normalize clipping plane info
        vec3 farpos = v_farpos.xyz / v_farpos.w;
        vec3 nearpos = v_nearpos.xyz / v_nearpos.w;

        // Calculate unit vector pointing in the view direction through this fragment.
        vec3 view_ray = normalize(nearpos.xyz - farpos.xyz);

        // Compute the (negative) distance to the front surface or near clipping plane.
        // v_position is the back face of the cuboid, so the initial distance calculated in the dot
        // product below is the distance from near clip plane to the back of the cuboid
        float distance_t = dot(nearpos - v_position, view_ray);
        distance_t = max(distance_t, min((-0.5 - v_position.x) / view_ray.x,
                                                                (u_size.x - 0.5 - v_position.x) / view_ray.x));
        distance_t = max(distance_t, min((-0.5 - v_position.y) / view_ray.y,
                                                                (u_size.y - 0.5 - v_position.y) / view_ray.y));
        distance_t = max(distance_t, min((-0.5 - v_position.z) / view_ray.z,
                                                                (u_size.z - 0.5 - v_position.z) / view_ray.z));

        // Now we have the starting position on the front surface
        vec3 front = v_position + view_ray * distance_t;

        // Decide how many steps to take
        int nsteps = int(-distance_t / relative_step_size + 0.5);
        if ( nsteps < 1 )
                discard;

        // Get starting location and step vector in texture coordinates
        vec3 step = ((v_position - front) / u_size) / float(nsteps);
        vec3 start_loc = front / u_size;

        cast_volume(start_loc, step, nsteps, view_ray);
        if (gl_FragColor.a < 0.05)
                discard;
}

vec4 apply_colormap(float val) {
        val = (val - u_clim[0]) / (u_clim[1] - u_clim[0]);
        val = clamp(val, 0.0, 1.0);
        vec4 color_vec4 = texture2D(u_cmdata, vec2(val, 0.5));
        return color_vec4;
}

void cast_volume(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray) {
    vec4 accumulatedColor = vec4(0.0); // Initialize accumulated color and transparency
    vec3 loc = start_loc;

    for (int iter = 0; iter < MAX_STEPS; iter++) {
        if (iter >= nsteps) break;

        float val = sample1(loc);
        // val = smoothstep(0.1, 0.9, val);

        vec4 sampleColor = apply_colormap(val);

        sampleColor.rgb *= sampleColor.a; // Pre-multiply alpha
        accumulatedColor.rgb += (1.0 - accumulatedColor.a) * sampleColor.rgb;
        accumulatedColor.a += (1.0 - accumulatedColor.a) * sampleColor.a;

        // Early exit if fully opaque
        if (accumulatedColor.a >= 0.95) break;

        loc += step; // Move along the ray
    }
    gl_FragColor = accumulatedColor;
    // Discard if completely transparent
    if (gl_FragColor.a == 0.0) discard;
}

void cast_mip(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray) {
        float max_val = -1e6;
        vec3 max_loc = start_loc;
        int max_i = 100;
        vec3 loc = start_loc;

        // Enter the raycasting loop. In WebGL 1 the loop index cannot be compared with
        // non-constant expression. So we use a hard-coded max, and an additional condition
        // inside the loop.
        for (int iter=0; iter<MAX_STEPS; iter++) {
                if (iter >= nsteps)
                        break;
                // Sample from the 3D texture
                float val = sample1(loc);   //  获取数据中对应值
                // Apply MIP operation
                if (val > max_val) {
                        max_val = val;
                        max_i = iter;
                }

                // Advance location deeper into the volume
                loc += step;
        }

        // Refine location, gives crispier images
        vec3 iloc = start_loc + step * (float(max_i) - 0.5);
        vec3 istep = step / float(REFINEMENT_STEPS);
        // 从视线射线出发获取 max_val
        for (int i=0; i<REFINEMENT_STEPS; i++) {
            max_val = max(max_val, sample1(iloc));
            iloc += istep;
        }
        // Resolve final color
        vec4 color_res = apply_colormap(max_val);
        if (max_val == 0.0){
            discard;
        }
        else gl_FragColor = color_res;
}