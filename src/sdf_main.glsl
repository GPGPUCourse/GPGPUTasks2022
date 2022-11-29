
// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}

// see https://iquilezles.org/articles/distfunctions/
float sdCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

// smooth minimum function to create gradual transitions between SDFs
// https://iquilezles.org/articles/smin/
float smoothmin(float d0, float d1, float k)
{
    float h = max( k-abs(d0-d1), 0.0 )/k;
    return min( d0, d1 ) - h*h*k*(1.0/4.0);
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;

    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }

    return 1.0;
}

vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // body, two spheres with smoothmin
    d = sdSphere((p - vec3(0.0, 0.37, -0.7)), 0.35);
    d = smoothmin(d, sdSphere(p - vec3(0.0, 0.66, -0.7), 0.22), 0.35);

    // hands, two capsules, can wave with lazycos
    float h1 = sdCapsule(p, vec3(-0.22, 0.33, -0.9), vec3(-0.4, 0.33 + 0.07*lazycos(7. * iTime), -0.4), 0.06);
    d = min(d, h1);
    // решил сделать вторую руку тоже анимированной для разнообразия
    float h2 = sdCapsule(p, vec3(0.22, 0.33, -0.9), vec3(0.4, 0.33 + -0.07*lazycos(7. * iTime), -0.4), 0.06);
    d = min(d, h2);

    // legs, two capsules
    float l1 = sdCapsule(p, vec3(0.2, -0.05, -0.8), vec3(0.1, 0.2, -0.9), 0.07);
    d = min(d, l1);
    float l2 = sdCapsule(p, vec3(-0.2, -0.05, -0.8), vec3(-0.1, 0.2, -0.9), 0.07);
    d = min(d, l2);

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEyeBall(vec3 p)
{

    float d0 = sdSphere(p - vec3(0., 0.52, -0.5), 0.21);

    // return distance and color
    return vec4(d0, vec3(1.0, 1.0, 1.0));

}

vec4 sdEyePupil(vec3 p)
{

    float d0 = sdSphere(p - vec3(0., 0.52, -0.38), 0.1);

    // return distance and color
    return vec4(d0, vec3(0.0, 0.0, 0.0));

}

vec4 sdEyeIris(vec3 p)
{

    float d0 = sdSphere(p - vec3(0., 0.52, -0.489), 0.2);

    // return distance and color
    return vec4(d0, vec3(0.0, 1.0, 1.0));

}

vec4 sdEye(vec3 p)
{

    vec4 res = sdEyeBall(p);

    vec4 ep = sdEyePupil(p);
    if (ep.x < res.x) res = ep;

    vec4 ei = sdEyeIris(p);
    if (ei.x < res.x) res = ei;

    
    return res;
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 res = sdBody(p);
    
    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }
    
    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);
    
    
    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, vec3(1.0, 0.0, 0.0));
    }
    
    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                           sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                           sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    
    float EPS = 1e-3;
    
    
    // p = ray_origin + t * ray_direction;
    
    float t = 0.0;
    
    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, vec3(0.0, 0.0, 0.0));
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{
    
    vec3 light_dir = normalize(light_source - p);
    
    float shading = dot(light_dir, normal);
    
    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);
    
    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{
    
    vec3 light_dir = p - light_source;
    
    float target_dist = length(light_dir);
    
    
    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }
    
    return 1.0;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.y;
    
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);
    

    vec3 ray_origin = vec3(0.0, 0.5, 1.0);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));
    

    vec4 res = raycast(ray_origin, ray_direction);
    
    
    
    vec3 col = res.yzw;
    
    
    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);
    
    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);
    
    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;
    
    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;
    
    
    
    // Output to screen
    fragColor = vec4(col, 1.0);
}
