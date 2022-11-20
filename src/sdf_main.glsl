#define ITERS 320
#define EPS 1.e-4

#define T_VOID 1.e10
#define M_PI 3.14159265359


void with(inout vec4 cur, vec4 oth) {
    if (oth.x < cur.x) {
        cur = oth;
    }
}

void with(inout vec4 cur, float othDist, vec3 othCol) {
    with(cur, vec4(othDist, othCol));
}

const vec4 SD_VOID = vec4(T_VOID, 0., 0., 0.);


// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdSphereNoised(vec3 p, float r, float n) {
    vec3 ref1 = vec3(0., 0., 1.0);
    vec3 ref2 = vec3(0., 1., .0);
    float k1 = dot(ref1, normalize(p));
    float k2 = dot(ref2, normalize(p));
    return length(p) - r + n * (sin(k1 * 100.) * 0.4 + sin(k2 * 50.));
}

// XZ plane
float sdPlane(vec3 p) {
    return p.y;
}

// see https://iquilezles.org/articles/distfunctions/
float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

// smooth minimum function to create gradual transitions between SDFs
// https://iquilezles.org/articles/smin/
float sminE(float a, float b, float k) {
    float res = exp(-k * a) + exp2(-k * b);
    return -log(res) / k;
}

float sminP(float a, float b, float k) {
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*k*(1.0/4.0);
}


// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle) {
    int nsleep = 10;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }
    
    return 1.0;
}


float sdHand(vec3 p, float a) {
    return sdCapsule(p, vec3(0., 0., 0.), vec3(-0.25 * cos(a), 0.25 * sin(a), 0.1), 0.07);
}

float sdLeg(vec3 p) {
    return sdCapsule(p, vec3(0., 0., 0.), vec3(0., -0.25, 0.), 0.07);
}

vec4 sdBody(vec3 p) {
    float d = T_VOID;

    // body, two spheres with smoothmin
    float d1 = sdSphereNoised((p - vec3(0.0, 0.4, -0.7)), 0.35, .001);
    float d2 = sdSphere((p - vec3(0.0, 0.8, -0.65)), 0.2);
    // d = sminE(d1, d2, 12.);
    d = sminP(d1, d2, .33);
    
    // hands, two capsules, can wave with lazycos
    vec3 hPos = vec3(-0.3, 0.4, -0.7);
    float lazyTime = lazycos(iTime * 20.);
    float dh1 = sdHand(vec3(-p.x, p.yz) - hPos, -1.);
    float dh2 = sdHand(p - hPos, -lazyTime * 0.8 - 0.2);
    d = min(d, min(dh1, dh2));
    
    // legs, two capsules
    vec3 lPos = vec3(0.2, 0.22, -0.7);
    float dl1 = sdLeg(vec3(-p.x, p.yz) - lPos);
    float dl2 = sdLeg(p - lPos);
    d = min(d, min(dl1, dl2));
    
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEyeBall(vec3 p) {
    float d = sdSphere(p, 0.18);
    return vec4(d, vec3(1.0, 1.0, 1.0));

}

vec4 sdEyePupil(vec3 p) {
    float d = sdSphere(p - vec3(0., 0., 0.06), 0.13);
    return vec4(d, vec3(0.0, 0.0, 0.0));

}

vec4 sdEyeIris(vec3 p) {
    float d = sdSphere(p - vec3(0., 0., 0.038), 0.15);
    return vec4(d, vec3(0.0, 1.0, 1.0));
}

vec4 sdEye(vec3 p) {
    vec4 res = SD_VOID;
    p = p - vec3(0.0, 0.75, -0.5);
    
    with(res, sdEyeBall(p));
    with(res, sdEyePupil(p));
    with(res, sdEyeIris(p));
    
    return res;
}

vec4 sdMonster(vec3 p) {
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 res = SD_VOID;
    
    with(res, sdBody(p));
    with(res, sdEye(p));
    
    return res;
}


vec4 sdTotal(vec3 p) {
    vec4 res = SD_VOID;
    
    with(res, sdMonster(p));
    with(res, sdPlane(p), vec3(1.0, 0.0, 0.0));
    
    return res;
}

// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const vec2 h = vec2(EPS,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                           sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                           sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction) {
    // p = ray_origin + t * ray_direction;
    
    float t = 0.0;
    
    for (int iter = 0; iter < ITERS; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(T_VOID, vec3(0.0, 0.0, 0.0));
}


float shading(vec3 p, vec3 light_source, vec3 normal) {
    vec3 light_dir = normalize(light_source - p);
    
    float shading = dot(light_dir, normal);
    
    return clamp(shading, 0.3, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness) {
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);
    
    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source) {    
    vec3 light_dir = p - light_source;

    float target_dist = length(light_dir);
    
    if (raycast(light_source, normalize(light_dir)).x + EPS < target_dist) {
        return 0.4;
    }

    return 1.0;
}

float fog(float dist) {
    dist = clamp(dist - .8, 0., 3.2);
    return exp(-dist * 0.08);
}


void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord/iResolution.y;
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);
    
    vec3 ray_origin = vec3(0.0, 0.7, 1.1);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1) - vec3(0., 0.06, 0.));
    
    vec4 res = raycast(ray_origin, ray_direction);
    float t = res.x;
    vec3 col = res.yzw;
    
    if (t != T_VOID) {
        vec3 surface_point = ray_origin + t * ray_direction;
        vec3 normal = calcNormal(surface_point);

        vec3 light_source = vec3(1.0 + 2.5 * sin(iTime), 10.0, 10.0);

        float shad = shading(surface_point, light_source, normal);
        shad = min(shad, castShadow(surface_point, light_source));
        col *= shad;

        float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
        col += vec3(1.0, 1.0, 1.0) * spec;
    } else {
        col = vec3(0.3 - 0.2*sin(uv.x / wh.x * M_PI) * sin(uv.y * M_PI), 0.1, 0.1);
    }
    
    fragColor = vec4(col * fog(t), 1.0);
}
