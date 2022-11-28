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
float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba)/dot(ba, ba), 0.0, 1.0);
    return length(pa - ba*h) - r;
}

// smooth minimum function to create gradual transitions between SDFs
// https://iquilezles.org/articles/smin/
float smoothmin(float d0, float d1, float k)
{
    float res = exp2(-k*d0) + exp2(-k*d1);
    return -log2(res)/k;
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

float easeInSine(float x) {
    return 1.0 - cos((x * 3.14159265359) / 2.0);
}

vec4 sdBody(vec3 p)
{
    float d = 1e10;

    // body, two spheres with smoothmin
    d = sdSphere((p - vec3(0.0, 0.35, -0.7)), 0.35);
    d = smoothmin(d, sdSphere(p - vec3(0.0, 0.67, -0.7), 0.25), 15.0);

    // hands, two capsules, can wave with lazycos
    float ang = (lazycos(7.0 * iTime) + 1.0) / 2.0;
    ang = easeInSine(ang);
    d = min(d, sdCapsule(p, vec3(-0.30, 0.42, -0.7), vec3(-0.42, 0.30 + 0.14 * (1.0 - ang), -0.7), 0.05));
    d = min(d, sdCapsule(p, vec3(0.30, 0.42, -0.7), vec3(0.42, 0.30, -0.7), 0.05));

    // legs, two capsules
    d = min(d, sdCapsule(p, vec3(-0.1, 0.1, -0.7), vec3(-0.1, -0.03, -0.7), 0.05));
    d = min(d, sdCapsule(p, vec3(0.1, 0.1, -0.7), vec3(0.1, -0.03, -0.7), 0.05));

    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEyeBall(vec3 p)
{
    float d0 = sdSphere(p, 0.20);

    // return distance and color
    return vec4(d0, vec3(1.0, 1.0, 1.0));

}

vec4 sdEyePupil(vec3 p)
{
    float d0 = sdSphere(p, 0.08);

    // return distance and color
    return vec4(d0, vec3(0.0, 0.0, 0.0));

}

vec4 sdEyeIris(vec3 p)
{
    float d0 = sdSphere(p, 0.15);

    // return distance and color
    return vec4(d0, vec3(0.0, 1.0, 1.0));

}

vec4 sdEye(vec3 p)
{

    vec4 res = sdEyeBall(p - vec3(0.0, 0.63, -0.52));
    vec4 iris = sdEyeIris(p - vec3(0.0, 0.64, -0.46));
    if (iris.x < res.x) {
        res = iris;
    }
    vec4 pupil = sdEyePupil(p - vec3(0.0, 0.65, -0.38));
    if (pupil.x < res.x) {
        res = pupil;
    }

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
vec3 calcNormal(in vec3 p)// for function f(p)
{
    const float eps = 0.0001;// or some other value
    const vec2 h = vec2(eps, 0);
    return normalize(vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
    sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
    sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x));
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

float easeOutBounce(float x) {
    float n1 = 7.5625;
    float d1 = 2.75;

    if (x < (1.0 / d1)) {
        return n1 * x * x;
    } else if (x < (2.0 / d1)) {
        return n1 * (x - (1.5 / d1)) * x + 0.75;
    } else if (x < (2.5 / d1)) {
        return n1 * (x - (2.25 / d1)) * x + 0.9375;
    } else {
        return n1 * (x - (2.625 / d1)) * x + 0.984375;
    }
}

float easeInOutBounce(float x) {
    return x < 0.5 ? ((1.0 - easeOutBounce(1.0 - 2.0 * x)) / 2.0) : ((1.0 + easeOutBounce(2.0 * x - 1.0)) / 2.0);
}

float lightPower() {
    float x = lazycos(10.0 * iTime);
    x = (1.0 + x) / 2.0;
    x = easeInOutBounce(x);
    x = 0.5 + (x * 0.5);
    return x;
}


void mainImage(out vec4 fragColor, in vec2 fragCoord)
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
    shad *= lightPower();
    col *= shad;

    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;


    // Output to screen
    fragColor = vec4(col, 1.0);
}
