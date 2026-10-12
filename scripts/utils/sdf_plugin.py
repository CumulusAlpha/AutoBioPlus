import warp as wp
from mujoco_warp._src.types import vec_pluginattr

# =========================
# Attribute layout
# =========================

ATTR_PITCH = 0
ATTR_RADIUS = 1
ATTR_LOW = 2
ATTR_HIGH = 3
ATTR_GAUGE = 4


# =========================
# Helix distance (exact)
# =========================


@wp.func
def helix_curve_sdf(
    p: wp.vec3, radius: float, pitch: float, low: float, high: float
) -> float:

    # match C implementation scaling
    pitch_scaled = pitch / (2.0 * wp.pi)
    low_scaled = low * (2.0 * wp.pi)
    high_scaled = high * (2.0 * wp.pi)

    x = p[0]
    y = p[1]
    z = p[2]

    phi = wp.atan2(y, x)

    # integer winding index
    k = wp.round((z / pitch_scaled - phi) / (2.0 * wp.pi))

    l = wp.ceil((low_scaled - phi) / (2.0 * wp.pi))
    h = wp.floor((high_scaled - phi) / (2.0 * wp.pi))

    # initialize distance as large value
    d = 1e10

    # =========================
    # boundary cases
    # =========================

    if k < l:
        k = l

        t = low_scaled
        lx = radius * wp.cos(t)
        ly = radius * wp.sin(t)
        lz = t * pitch_scaled

        dx = lx - x
        dy = ly - y
        dz = lz - z

        d = wp.sqrt(dx * dx + dy * dy + dz * dz)

    elif k > h:
        k = h

        t = high_scaled
        hx = radius * wp.cos(t)
        hy = radius * wp.sin(t)
        hz = t * pitch_scaled

        dx = hx - x
        dy = hy - y
        dz = hz - z

        d = wp.sqrt(dx * dx + dy * dy + dz * dz)

    # =========================
    # main helix distance
    # =========================

    t = 2.0 * wp.pi * k + phi

    dz = z - pitch_scaled * t
    r = wp.sqrt(x * x + y * y)
    dr = r - radius

    d_helix = wp.sqrt(dr * dr + dz * dz)

    return wp.min(d, d_helix)


# =========================
# Thread SDF
# =========================


@wp.func
def thread(p: wp.vec3, attr: vec_pluginattr) -> float:

    pitch = attr[ATTR_PITCH]
    radius = attr[ATTR_RADIUS]
    low = attr[ATTR_LOW]
    high = attr[ATTR_HIGH]
    gauge = attr[ATTR_GAUGE]

    d = helix_curve_sdf(p, radius, pitch, low, high)

    return d - gauge


# =========================
# Gradient
# =========================


@wp.func
def thread_sdf_grad(p: wp.vec3, attr: vec_pluginattr) -> wp.vec3:

    eps = 1e-8

    xp = wp.vec3(p[0] + eps, p[1], p[2])
    xn = wp.vec3(p[0] - eps, p[1], p[2])
    dx = thread(xp, attr) - thread(xn, attr)

    yp = wp.vec3(p[0], p[1] + eps, p[2])
    yn = wp.vec3(p[0], p[1] - eps, p[2])
    dy = thread(yp, attr) - thread(yn, attr)

    zp = wp.vec3(p[0], p[1], p[2] + eps)
    zn = wp.vec3(p[0], p[1], p[2] - eps)
    dz = thread(zp, attr) - thread(zn, attr)

    return wp.vec3(dx, dy, dz) / (2.0 * eps)
