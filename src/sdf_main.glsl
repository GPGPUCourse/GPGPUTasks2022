#define PI 3.14159265359
#define PIm2 (PI * 2.0)

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}

// plane is moved as if the monstrik was jumping towards us
float planeZ() {
    int nsleep = 9;
    float iter = iTime*5.0 / PIm2 + 3.0;
    int phase = int(iter) % nsleep;
    
    float baseTime = float(int(iter)/nsleep*nsleep - 3) * PIm2/5.0;
    float remTime = iTime - baseTime; // when phase < 3, [-1.2 pi; 0)
    
    // when phase < 3, [0; 3); int(jumpPhase) == phase
    float jumpPhase = 3.0 + remTime / (1.2 * PI) * 3.0;
    float speedPhase = float(phase) + cos(PI * (jumpPhase - float(phase)));
    float remZ = (speedPhase - 3.0) / 3.0 * 1.2 * PI;
    
    return baseTime + 
        (phase < 3 ? remZ : 0.0);
}
// XZ plane
float sdPlane(vec3 p)
{
    return p.y + sin(p.x * 10.0) * sin((p.z + planeZ()) * 10.0) * 0.01;
}

// see https://iquilezles.org/articles/distfunctions/
float sdCapsule( vec3 p, vec3 a, vec3 fr, float r )
{
    vec3 b = a + fr;
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

vec4 chcol(vec4 dCol1, vec4 dCol2)
{
    
    return vec4(
        min(dCol1.x, dCol2.x),
        (dCol1.x < dCol2.x) ? dCol1.yzw : dCol2.yzw
    );
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 9;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }
    
    return 1.0;
}

float handsWavingTime() { // 0 to 1, with sleeping at 0
    return -lazycos(iTime*5.0)/2.0 + 0.5;
}

float sitStandLoopTime() { // 0 to 1, with sleeping at 0
    return -lazycos(iTime*5.0 + 3.0 * 6.28318530718)/2.0 + 0.5;
}

bool isWaving() {
    int nsleep = 9;
    int iperiod = int(iTime*5.0 / 6.28318530718) % nsleep;
    return iperiod < 3;
}

bool isSquatting() {
    int nsleep = 9;
    int iperiod = int(iTime*5.0 / 6.28318530718 + 3.0) % nsleep;
    return iperiod < 3;
}

vec4 sdBody(vec3 p)
{
    float d = 1e10;
    
    float bodyY = sitStandLoopTime() * 0.30 + 0.35;

    // body, two spheres with smoothmin
    d = sdSphere((p - vec3(0.0, bodyY, -0.7)), 0.35);
    d = smoothmin(d, sdSphere((p - vec3(0.0, bodyY + 0.3, -0.65)), 0.25), 0.2);
    
    // hands, two capsules, can wave with lazycos
    vec3 limbsP = p - vec3(0.0, bodyY, -0.65);
    
    // leftArm
    float armAngle = -1.0;
    if (isWaving()) {
        armAngle = handsWavingTime() * 0.6 - 1.0;
    } else if (isSquatting()) {
        armAngle = sitStandLoopTime() * -1.8 - 1.0;
    }
    float armX = sin(armAngle) * 0.3;
    float armY = cos(armAngle) * 0.3;
    
    d = smoothmin(d, 
        sdCapsule(
            limbsP, 
            vec3(-0.25, 0.1, 0.0),
            vec3(armX, armY, 0.0),
            0.05
        ), 
        0.025);
    d = smoothmin(d, 
        sdCapsule(
            limbsP, 
            vec3(0.25, 0.1, 0.0),
            vec3(-armX, armY, 0.0),
            0.05
        ), 
        0.025);
    
    // legs, two capsules
    // or four?
    
    // left knee, relative to the hip
    float kneeX = -0.45 / 2.0 + (-0.45 / 2.0) * (1.0 - sitStandLoopTime());
    float kneeY = -0.15 + (-0.3 / 2.0) * sitStandLoopTime();
    float kneeZ = 0.15 - 0.15 * sitStandLoopTime();
    
    d = smoothmin(d, 
        sdCapsule(
            limbsP,
            vec3(-0.2, -0.2, 0.0),
            vec3(kneeX + 0.2, kneeY + 0.2, kneeZ),
            0.05
        ), 
        0.025);
    d = smoothmin(d, 
        sdCapsule(
            limbsP,
            vec3(0.2, -0.2, 0.0),
            vec3(-kneeX - 0.2, kneeY + 0.2, kneeZ),
            0.05
        ), 
        0.025);
    
    // left foot, relative to the knee
    float feetX = 0.5 + -0.5 * sitStandLoopTime();
    float feetY = -0.20 + (-0.3 / 2.0) * sitStandLoopTime();
    float feetZ = 0.15 - 0.15 * sitStandLoopTime();
    
    d = smoothmin(d, 
        sdCapsule(
            limbsP,
            vec3(kneeX, kneeY, kneeZ),
            vec3(feetX, feetY, feetZ),
            0.05
        ), 
        0.015);
    d = smoothmin(d, 
        sdCapsule(
            limbsP,
            vec3(-kneeX, kneeY, kneeZ),
            vec3(-feetX, feetY, feetZ),
            0.05
        ), 
        0.015);
    
    // return distance and color
    return vec4(d, vec3(0.0, 1.0, 0.0));
}

vec4 sdEyeBall(vec3 p)
{
    float d0 = sdSphere((p - vec3(0.0, 0.60, -0.50)), 0.20);
    
    // return distance and color
    return vec4(d0, vec3(1.0, 1.0, 1.0));

}

vec4 sdEyePupil(vec3 p)
{
    float d0 = sdSphere((p - vec3(0.0, 0.60, -0.33)), 0.07);
    
    // return distance and color
    return vec4(d0, vec3(0.0, 0.0, 0.0));

}

vec4 sdEyeIris(vec3 p)
{

    float d0 = sdSphere((p - vec3(0.0, 0.60, -0.44)), 0.15);
    
    // return distance and color
    return vec4(d0, vec3(0.0, 1.0, 1.0));

}

vec4 sdEye(vec3 p)
{
    // it's not a bug that Y coord is fixed :)
    vec4 ball = sdEyeBall(p);
    vec4 iris = sdEyeIris(p);
    vec4 pupil = sdEyePupil(p);
    
    return chcol(ball, chcol(iris, pupil));
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 body = sdBody(p);
    vec4 eye = sdEye(p);
    
    vec3 col = chcol(body, eye).yzw;
    
    
    return vec4(smoothmin(body.x, eye.x, 0.03), col);
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
