import { WG_SIZE, MAX_WALLS } from './config.js';

const WGSL_COMMON = /* wgsl */`
// Cohesive (negative) EOS pressure floor, gated by density: fluid near rest
// density keeps a slight cohesion for droplet cohesiveness, but sparse particles
// (below the gate) get a hard 0 floor — otherwise low-particle-count fluid
// attracts itself into stable clusters (tensile instability).
const COHESION_PRESSURE: f32     = -0.1;   // pressure floor where cohesion is allowed
const COHESION_DENSITY_GATE: f32 = 0.7;    // fraction of rest_density above which cohesion applies

const FIXED: f32 = 100000.0;

// i32 max is ~2.147e9, so FIXED=1e5 caps the encodable range at ~±21474. Clamp a
// bit inside that so a stray huge value (velocity spike, exploding density) saturates
// instead of overflowing i32 on the *FIXED multiply (UB / wraps to garbage, which
// atomicAdd would then bake into the grid as a corrupt cell).
fn encode(v: f32) -> i32 {
    let clamped = clamp(v, -20000.0, 20000.0);
    return i32(clamped * FIXED);
}
fn decode(v: i32) -> f32 { return f32(v) / FIXED; }

struct Params {
    grid_X     : u32,
    grid_Y     : u32,
    grid_Z     : u32,
    particle_num: u32,
    dt         : f32,
    gravity    : f32,
    rest_density: f32,
    stiffness  : f32,
    eos_power  : f32,
    viscosity  : f32,
    hard_min   : f32,
    hard_max_x : f32,
    hard_max_y : f32,
    hard_max_z : f32,
    // Splash方式の予測位置バネ補正 (ref/boundary-condition-splash-style.md ステップ3)
    wall_min       : f32,
    wall_stiffness : f32,
    lookahead_k    : f32,
    _pad0      : f32,
    _pad1      : f32,
    _pad2      : f32,
}

struct Weights3 { wx: vec3<f32>, wy: vec3<f32>, wz: vec3<f32>, }

fn quadratic_weights(f: vec3<f32>) -> Weights3 {
    return Weights3(
        vec3<f32>(0.5*(0.5-f.x)*(0.5-f.x), 0.75-f.x*f.x, 0.5*(0.5+f.x)*(0.5+f.x)),
        vec3<f32>(0.5*(0.5-f.y)*(0.5-f.y), 0.75-f.y*f.y, 0.5*(0.5+f.y)*(0.5+f.y)),
        vec3<f32>(0.5*(0.5-f.z)*(0.5-f.z), 0.75-f.z*f.z, 0.5*(0.5+f.z)*(0.5+f.z)),
    );
}

// Per-particle cell-neighborhood data (base grid cell + fractional offset + quadratic
// B-spline weights) — computed once per substep by WGSL_COMPUTE_WEIGHTS and shared by
// P2G_MASS/P2G_MOM/G2P instead of each of the three re-deriving it from particle_pos
// (same pos throughout a substep, so base/f/weights are identical across all three).
struct BaseWeights {
    bx: i32, by: i32, bz: i32,
    fx: f32, fy: f32, fz: f32,
    wx0: f32, wx1: f32, wx2: f32,
    wy0: f32, wy1: f32, wy2: f32,
    wz0: f32, wz1: f32, wz2: f32,
}

// Both RayParams-consuming shaders (WGSL_COMPACT, WGSL_RAYCAST) test against the
// infinite cylinder around the camera ray (origin + t*dir), exactly like
// WGSL_APPLY_HAND's cell test — this makes them correct regardless of the actual
// fluid depth, unlike testing against a single point projected onto a fixed depth
// plane (which misses when the fluid surface isn't at that plane's depth).
struct RayParams { ox: f32, oy: f32, oz: f32, radius: f32, dx: f32, dy: f32, dz: f32, _pad: f32 }

fn ray_perp_dist(pos: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let to_p = pos - origin;
    let perp = to_p - dot(to_p, dir) * dir;
    return length(perp);
}

// Flattened grid index for (cx,cy,cz), or -1 if outside [0,gx)×[0,gy)×[0,gz).
// Collapses the "3× bounds-check + continue" scaffolding that recurs wherever a
// shader gathers/scatters over a particle's 3×3×3 cell neighborhood without a
// precomputed baseIdx (P2G_MASS, diffuse advect's sample_grid_velocity). The
// perf-critical inner loops that DO have a precomputed baseIdx (P2G_MOM, G2P)
// intentionally skip this and use baseIdx + offset arithmetic instead — see
// their comments.
fn cell_idx_checked(cx: i32, cy: i32, cz: i32, gx: u32, gy: u32, gz: u32) -> i32 {
    if (cx < 0 || cx >= i32(gx) || cy < 0 || cy >= i32(gy) || cz < 0 || cz >= i32(gz)) {
        return -1;
    }
    return (cz * i32(gy) + cy) * i32(gx) + cx;
}

// Hard-clamps one axis of a predicted position to [lo, hi], zeroing that axis'
// velocity component when clamped. Returns (clamped_pos, clamped_vel).
// Used by G2P's final boundary safety valve, applied once per axis.
fn hard_clamp_axis(p: f32, v: f32, lo: f32, hi: f32) -> vec2<f32> {
    var np = p; var nv = v;
    if (np < lo) { np = lo; nv = 0.0; }
    if (np > hi) { np = hi; nv = 0.0; }
    return vec2<f32>(np, nv);
}

// ── 動く壁 (AABB障害物) ────────────────────────────────────────────────
// Movable AABB obstacles, coupled to the fluid via grid boundary conditions
// (AABB IF-test in UPDATE_GRID + G2P). Layout kept in sync with fluid-gpu.js's
// 192B uniform (MAX_WALLS × 48B) and the renderer's box pass (config.js MAX_WALLS
// is the single source of truth for the count).
// min_active.w = active flag: 0.0 = inactive (ignored everywhere), 1.0 = active.
// "active" means w >= 0.5. vel is derived CPU-side each simFrame as
// (new min − prev-frame min)/DT.
struct Wall {
    min_active : vec4<f32>,   // xyz = min corner,  w = active flag (0=off / 1=on)
    max_pad    : vec4<f32>,   // xyz = max corner
    vel_pad    : vec4<f32>,   // xyz = wall velocity (grid units / sim time)
}
struct WallParams { walls: array<Wall, ${MAX_WALLS}> }

// AABB signed distance + outward normal, packed as vec4 (xyz = normal, w = sdf).
// Outside (w > 0): normal points from the nearest box face toward the query point
//   (= face normal in the face region, corner-diagonal past a corner).
// Inside  (w < 0): normal is the SHALLOWEST-penetration face's outward normal, so
//   pushing along it by -w escapes the box by the least distance.
// Shared by UPDATE_GRID (velocity BC), G2P (position push-out) and DIFFUSE_ADVECT.
fn wall_sdf_normal(p: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>) -> vec4<f32> {
    let dmin = mn - p;   // per-axis: >0 ⇒ p is below the min face
    let dmax = p - mx;   // per-axis: >0 ⇒ p is above the max face
    let q    = max(dmin, dmax);
    let qmax = max(q.x, max(q.y, q.z));
    if (qmax > 0.0) {
        // Outside on at least one axis: distance/normal from the nearest surface point.
        let nearest = clamp(p, mn, mx);
        let diff    = p - nearest;
        let len     = length(diff);
        let n = select(vec3<f32>(0.0, 1.0, 0.0), diff / len, len > 1e-6);
        return vec4<f32>(n, length(max(q, vec3<f32>(0.0))));
    }
    // Fully inside: qmax (the least-negative axis gap) is the shallowest penetration.
    var n = vec3<f32>(0.0, 1.0, 0.0);
    if (qmax == q.x)      { n = vec3<f32>(select(-1.0, 1.0, dmax.x > dmin.x), 0.0, 0.0); }
    else if (qmax == q.y) { n = vec3<f32>(0.0, select(-1.0, 1.0, dmax.y > dmin.y), 0.0); }
    else                  { n = vec3<f32>(0.0, 0.0, select(-1.0, 1.0, dmax.z > dmin.z)); }
    return vec4<f32>(n, qmax);   // qmax <= 0 = signed penetration depth
}
`;

export const WGSL_CLEAR = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> cell_mv        : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> cell_mass      : array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             params         : Params;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.grid_X * params.grid_Y * params.grid_Z) { return; }
    atomicStore(&cell_mass[idx],           0);
    atomicStore(&cell_mv[idx * 3u + 0u],   0);
    atomicStore(&cell_mv[idx * 3u + 1u],   0);
    atomicStore(&cell_mv[idx * 3u + 2u],   0);
}
`;

// Computes per-particle base cell + fractional offset + quadratic B-spline weights
// once per substep (particle_pos is constant across CLEAR→...→G2P within a substep),
// so P2G_MASS/P2G_MOM/G2P can read it instead of each redundantly recomputing
// floor()/quadratic_weights() from scratch (same result all 3 times).
export const WGSL_COMPUTE_WEIGHTS = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read>       particle_pos : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out_weights  : array<BaseWeights>;
@group(0) @binding(2) var<uniform>             params       : Params;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pos   = particle_pos[p].xyz;
    let cellI = vec3<i32>(floor(pos));
    let base  = cellI - vec3<i32>(1);
    let f     = pos - vec3<f32>(cellI) - vec3<f32>(0.5);
    let wt    = quadratic_weights(f);

    var o: BaseWeights;
    o.bx = base.x; o.by = base.y; o.bz = base.z;
    o.fx = f.x;    o.fy = f.y;    o.fz = f.z;
    o.wx0 = wt.wx.x; o.wx1 = wt.wx.y; o.wx2 = wt.wx.z;
    o.wy0 = wt.wy.x; o.wy1 = wt.wy.y; o.wy2 = wt.wy.z;
    o.wz0 = wt.wz.x; o.wz1 = wt.wz.y; o.wz2 = wt.wz.z;
    out_weights[p] = o;
}
`;

export const WGSL_P2G_MASS = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read>       particle_mass : array<f32>;
@group(0) @binding(1) var<storage, read_write> cell_mass     : array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             params        : Params;
@group(0) @binding(3) var<storage, read>       weights       : array<BaseWeights>;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pm  = particle_mass[p];
    let bw  = weights[p];
    let base = vec3<i32>(bw.bx, bw.by, bw.bz);
    let wx = vec3<f32>(bw.wx0, bw.wx1, bw.wx2);
    let wy = vec3<f32>(bw.wy0, bw.wy1, bw.wy2);
    let wz = vec3<f32>(bw.wz0, bw.wz1, bw.wz2);

    for (var k: i32 = 0; k < 3; k++) {
        let cz = base.z + k;
        for (var j: i32 = 0; j < 3; j++) {
            let cy = base.y + j;
            for (var i: i32 = 0; i < 3; i++) {
                let cx = base.x + i;
                let idx = cell_idx_checked(cx, cy, cz, params.grid_X, params.grid_Y, params.grid_Z);
                if (idx < 0) { continue; }
                let w = wx[i] * wy[j] * wz[k];
                atomicAdd(&cell_mass[u32(idx)], encode(w * pm));
            }
        }
    }
}
`;

export const WGSL_P2G_MOM = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read>       particle_vel    : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       particle_affine : array<f32>;
@group(0) @binding(2) var<storage, read>       particle_mass   : array<f32>;
@group(0) @binding(3) var<storage, read>       cell_mass_f32    : array<f32>;
@group(0) @binding(4) var<storage, read_write> cell_mv          : array<atomic<i32>>;
@group(0) @binding(5) var<uniform>             params           : Params;
@group(0) @binding(6) var<storage, read_write> particle_density : array<f32>;
@group(0) @binding(7) var<storage, read>       weights          : array<BaseWeights>;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let vel  = particle_vel[p].xyz;
    let mass = particle_mass[p];
    let off  = p * 9u;
    let a00 = particle_affine[off+0u]; let a01 = particle_affine[off+1u]; let a02 = particle_affine[off+2u];
    let a10 = particle_affine[off+3u]; let a11 = particle_affine[off+4u]; let a12 = particle_affine[off+5u];
    let a20 = particle_affine[off+6u]; let a21 = particle_affine[off+7u]; let a22 = particle_affine[off+8u];

    let bw = weights[p];
    let base = vec3<i32>(bw.bx, bw.by, bw.bz);
    let f    = vec3<f32>(bw.fx, bw.fy, bw.fz);
    let wt   = Weights3(vec3<f32>(bw.wx0,bw.wx1,bw.wx2), vec3<f32>(bw.wy0,bw.wy1,bw.wy2), vec3<f32>(bw.wz0,bw.wz1,bw.wz2));

    let SLICE   = params.grid_X * params.grid_Y;
    let baseIdx = u32((base.z * i32(params.grid_Y) + base.y) * i32(params.grid_X) + base.x);

    var density = 0.0;
    for (var k: i32 = 0; k < 3; k++) {
        for (var j: i32 = 0; j < 3; j++) {
            for (var i: i32 = 0; i < 3; i++) {
                let cidx = baseIdx + u32(k) * SLICE + u32(j) * params.grid_X + u32(i);
                density += wt.wx[i] * wt.wy[j] * wt.wz[k] * cell_mass_f32[cidx];
            }
        }
    }
    particle_density[p] = density;
    if (density <= 1e-8) { return; }

    // Clamp the density used for the volume estimate: an isolated/sparse particle
    // (e.g. after real-time deletion thins the fluid) can see a near-zero local
    // density, and vol = mass / density blows up into a huge force even though
    // the pressure term itself is already floored below (density-gated cohesion —
    // sparse particles get a hard 0 floor, only near-rest density keeps cohesion)
    // — visible as particles spuriously bouncing at rest. particle_density above
    // stays unclamped since the renderer uses it (isolation size shrink).
    let vol = mass / max(density, params.rest_density * 0.15);
    var pressure = params.stiffness * (pow(density / params.rest_density, params.eos_power) - 1.0);
    let pFloor = select(0.0, COHESION_PRESSURE, density > params.rest_density * COHESION_DENSITY_GATE);
    pressure = max(pressure, pFloor);

    let visc = params.viscosity;
    let s00 = -pressure + visc * (a00 + a00);
    let s11 = -pressure + visc * (a11 + a11);
    let s22 = -pressure + visc * (a22 + a22);
    let s01 = visc * (a01 + a10);
    let s02 = visc * (a02 + a20);
    let s12 = visc * (a12 + a21);

    let coef = -vol * 4.0 * params.dt;

    var dxs: array<f32, 3>;
    dxs[0] = -f.x - 1.0; dxs[1] = -f.x; dxs[2] = 1.0 - f.x;
    var dys: array<f32, 3>;
    dys[0] = -f.y - 1.0; dys[1] = -f.y; dys[2] = 1.0 - f.y;
    var dzs: array<f32, 3>;
    dzs[0] = -f.z - 1.0; dzs[1] = -f.z; dzs[2] = 1.0 - f.z;

    for (var k: i32 = 0; k < 3; k++) {
        let wzk = wt.wz[k]; let dzk = dzs[k];
        for (var j: i32 = 0; j < 3; j++) {
            let wyzk = wt.wy[j] * wzk; let dyj = dys[j];
            for (var i: i32 = 0; i < 3; i++) {
                let w    = wt.wx[i] * wyzk;
                let dxi  = dxs[i];
                let cidx = baseIdx + u32(k) * SLICE + u32(j) * params.grid_X + u32(i);

                let Qx = a00*dxi + a01*dyj + a02*dzk;
                let Qy = a10*dxi + a11*dyj + a12*dzk;
                let Qz = a20*dxi + a21*dyj + a22*dzk;

                let fxf = coef * (s00*dxi + s01*dyj + s02*dzk);
                let fyf = coef * (s01*dxi + s11*dyj + s12*dzk);
                let fzf = coef * (s02*dxi + s12*dyj + s22*dzk);

                atomicAdd(&cell_mv[cidx*3u+0u], encode(w * (mass * (vel.x + Qx) + fxf)));
                atomicAdd(&cell_mv[cidx*3u+1u], encode(w * (mass * (vel.y + Qy) + fyf)));
                atomicAdd(&cell_mv[cidx*3u+2u], encode(w * (mass * (vel.z + Qz) + fzf)));
            }
        }
    }
}
`;

export const WGSL_DECODE_MASS = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> cell_mass     : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> cell_mass_f32 : array<f32>;
@group(0) @binding(2) var<uniform>             params        : Params;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.grid_X * params.grid_Y * params.grid_Z) { return; }
    cell_mass_f32[idx] = decode(atomicLoad(&cell_mass[idx]));
}
`;

// APPLY_HAND pushes fluid along an infinite cylinder around the camera ray through
// hand_pos (depth-independent, like deleteNear/raycastFluid) — so its dispatch can't
// simply be limited to a small box around hand_pos itself, the cylinder can run
// through the whole grid along that ray. Instead FluidGPU.updateHand() computes a
// bounding box (bbox_min/bbox_max, in grid cells) of (cylinder ∩ grid) on the CPU
// each frame and this shader is dispatched in 3D only over that box — still a large
// reduction over the whole grid, since the box is tight in the two axes roughly
// perpendicular to the ray even though it may span most of the grid along it.
export const WGSL_APPLY_HAND = WGSL_COMMON + /* wgsl */`
struct HandParams {
    px:f32, py:f32, pz:f32, radius:f32,
    vx:f32, vy:f32, vz:f32, is_on:u32,
    strength:f32, ex:f32, ey:f32, ez:f32,
    bbox_min_x:i32, bbox_min_y:i32, bbox_min_z:i32,
    bbox_max_x:i32, bbox_max_y:i32, bbox_max_z:i32,
    _pad0:i32, _pad1:i32,
};
@group(0) @binding(0) var<storage, read_write> cell_mv_f32 : array<f32>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var<uniform> hand   : HandParams;

// Fixed 4×4×4=64 shape (independent of config.js's WG_SIZE) — this shader's dispatch
// extent is already bbox-sized on the CPU side, so it doesn't need to match the 1D
// per-particle/per-cell workgroup convention used elsewhere.
@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (hand.is_on == 0u) { return; }
    let cx = hand.bbox_min_x + i32(gid.x);
    let cy = hand.bbox_min_y + i32(gid.y);
    let cz = hand.bbox_min_z + i32(gid.z);
    if (cx > hand.bbox_max_x || cy > hand.bbox_max_y || cz > hand.bbox_max_z) { return; }
    if (cx < 0 || cy < 0 || cz < 0 ||
        cx >= i32(params.grid_X) || cy >= i32(params.grid_Y) || cz >= i32(params.grid_Z)) { return; }

    let idx  = u32((cz * i32(params.grid_Y) + cy) * i32(params.grid_X) + cx);
    let cell = vec3<f32>(f32(cx) + 0.5, f32(cy) + 0.5, f32(cz) + 0.5);
    let hand_pos = vec3<f32>(hand.px, hand.py, hand.pz);
    let ray_dir  = normalize(hand_pos - vec3<f32>(hand.ex, hand.ey, hand.ez));
    let to_cell  = cell - hand_pos;
    let perp     = to_cell - dot(to_cell, ray_dir) * ray_dir;
    let d = length(perp);
    if (d >= hand.radius) { return; }
    let t = 1.0 - d / hand.radius;
    let w = t * t * hand.strength;
    cell_mv_f32[idx*3u+0u] += hand.vx * w;
    cell_mv_f32[idx*3u+1u] += hand.vy * w;
    cell_mv_f32[idx*3u+2u] += hand.vz * w;
}`;

export const WGSL_UPDATE_GRID = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> cell_mv        : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> cell_mass      : array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             params         : Params;
@group(0) @binding(3) var<storage, read_write> cell_mv_f32    : array<f32>;
@group(0) @binding(4) var<uniform>             wallp          : WallParams;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.grid_X * params.grid_Y * params.grid_Z) { return; }

    let cm = decode(atomicLoad(&cell_mass[idx]));
    if (cm <= 0.0) {
        cell_mv_f32[idx*3u+0u] = 0.0;
        cell_mv_f32[idx*3u+1u] = 0.0;
        cell_mv_f32[idx*3u+2u] = 0.0;
        return;
    }

    var vx = decode(atomicLoad(&cell_mv[idx*3u+0u])) / cm;
    var vy = decode(atomicLoad(&cell_mv[idx*3u+1u])) / cm;
    var vz = decode(atomicLoad(&cell_mv[idx*3u+2u])) / cm;

    vy += params.gravity * params.dt;

    let cx = i32(idx % params.grid_X);
    let cy = i32((idx / params.grid_X) % params.grid_Y);
    let cz = i32(idx / (params.grid_X * params.grid_Y));

    // Splash方式: 壁際2セル以内は方向に関係なく無条件に速度ゼロ (ref/boundary-condition-splash-style.md ステップ1)
    // y(重力方向)は薄い層だと床際の帯がほぼ全高を覆ってしまい、密度過多を解消する
    // 上向きの圧力速度まで握りつぶして粒子が同一座標に圧縮される→暴発の原因になったため、
    // 方向限定ロジックに戻す(床に向かう速度だけ止め、押し返す速度は生かす)。
    if (cx < 2 || cx > i32(params.grid_X) - 3) { vx = 0.0; }
    if (cy < 2                       && vy < 0.0) { vy = 0.0; }
    if (cy > i32(params.grid_Y) - 3  && vy > 0.0) { vy = 0.0; }
    if (cz < 2 || cz > i32(params.grid_Z) - 3) { vz = 0.0; }

    // 床(y_min)際のセルはXZ方向速度に摩擦減衰をかけ、床に沿って粒子が滑って
    // 角に集まるのを抑える。
    if (cy < 2) { vx *= 0.85; vz *= 0.9; }

    // 動く壁 (AABB障害物) の速度境界条件。外壁Splash処理・床摩擦の後に適用。
    // セル中心 (index+0.5、APPLY_HAND と同じ規約) で壁SDFを評価:
    //  - 壁内部 (sdf<0): v = wall_vel を強制。壁が動くとこれが流体を押す駆動力になる。
    //  - 表面1セル以内: 相対速度 vrel=v-wall_vel の法線成分が壁に「接近する」向き(<0)の
    //    ときのみその成分を除去 (外壁と同じ方向限定ロジック=押し返しは生かす。
    //    接線成分は保持するので壁面を滑る)。
    //    帯幅は2セル→1セルに縮小済み (2セルだと隙間を閉じる微小な接近速度まで毎ステップ
    //    殺され、壁際の隙間が上ほど広い不均一として定着する — ヘッドレス計測で確認。
    //    1セルなら全高で隙間~0に密着し、貫通・漏出もゼロ)。
    let cellPos = vec3<f32>(f32(cx) + 0.5, f32(cy) + 0.5, f32(cz) + 0.5);
    var wv = vec3<f32>(vx, vy, vz);
    for (var wi = 0u; wi < ${MAX_WALLS}u; wi++) {
        let wall = wallp.walls[wi];
        if (wall.min_active.w < 0.5) { continue; }   // 非activeはスキップ
        let wvel = wall.vel_pad.xyz;
        let sn   = wall_sdf_normal(cellPos, wall.min_active.xyz, wall.max_pad.xyz);
        if (sn.w < 0.0) {
            wv = wvel;
        } else if (sn.w < 1.0) {
            let vrel = wv - wvel;
            let vn   = dot(vrel, sn.xyz);
            if (vn < 0.0) { wv = wv - vn * sn.xyz; }
        }
    }
    vx = wv.x; vy = wv.y; vz = wv.z;

    cell_mv_f32[idx*3u+0u] = vx;
    cell_mv_f32[idx*3u+1u] = vy;
    cell_mv_f32[idx*3u+2u] = vz;
}
`;

// Stream-compacts particles outside the deletion cylinder to the front of the array.
// Affine (velocity-gradient) data is deliberately NOT carried through compaction —
// G2P fully recomputes particle_affine[p] every substep regardless of its previous
// value, so a stale value for one substep only perturbs the viscosity term
// transiently. Skipping it keeps storage-buffer bindings (7) under the default
// WebGPU maxStorageBuffersPerShaderStage limit (8) without requesting higher limits.
export const WGSL_COMPACT = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read>       pos_in   : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       vel_in   : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       mass_in  : array<f32>;
@group(0) @binding(3) var<storage, read_write> pos_out  : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel_out  : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> mass_out : array<f32>;
@group(0) @binding(6) var<storage, read_write> counter  : atomic<u32>;
@group(0) @binding(7) var<uniform>             params   : Params;
@group(0) @binding(8) var<uniform>             del      : RayParams;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pos    = pos_in[p].xyz;
    let origin = vec3<f32>(del.ox, del.oy, del.oz);
    let dir    = vec3<f32>(del.dx, del.dy, del.dz);
    if (ray_perp_dist(pos, origin, dir) < del.radius) { return; }

    let idx = atomicAdd(&counter, 1u);
    pos_out[idx]  = pos_in[p];
    vel_out[idx]  = vel_in[p];
    mass_out[idx] = mass_in[p];
}
`;

// Finds where the camera ray first meets the particle cloud: the minimum distance-
// along-ray (t) among particles within `radius` of the ray axis. Used to place
// spawnParticles() at the existing fluid surface instead of a fixed-depth plane.
// atomicMin runs on the f32 bit pattern reinterpreted as u32 — valid because IEEE754
// positive floats preserve ordering when compared as unsigned integers. The JS side
// initializes hit_t to 0xFFFFFFFF (sentinel "no hit"; also happens to be the max
// possible u32, so any real hit's bit pattern is smaller) and decodes a non-sentinel
// result by bitcasting back to f32.
export const WGSL_RAYCAST = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read>       pos_in : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> hit_t  : atomic<u32>;
@group(0) @binding(2) var<uniform>             params : Params;
@group(0) @binding(3) var<uniform>             ray    : RayParams;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pos    = pos_in[p].xyz;
    let origin = vec3<f32>(ray.ox, ray.oy, ray.oz);
    let dir    = vec3<f32>(ray.dx, ray.dy, ray.dz);
    let t      = dot(pos - origin, dir);
    if (t <= 0.0) { return; }
    if (ray_perp_dist(pos, origin, dir) >= ray.radius) { return; }

    atomicMin(&hit_t, bitcast<u32>(t));
}
`;

export const WGSL_G2P = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> particle_pos    : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> particle_vel    : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> particle_affine : array<f32>;
@group(0) @binding(3) var<storage, read>       cell_mv_f32     : array<f32>;
@group(0) @binding(4) var<uniform>             params          : Params;
@group(0) @binding(5) var<storage, read>       weights         : array<BaseWeights>;
@group(0) @binding(6) var<uniform>             wallp           : WallParams;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pos = particle_pos[p].xyz;
    let bw  = weights[p];
    let base = vec3<i32>(bw.bx, bw.by, bw.bz);
    let wt   = Weights3(vec3<f32>(bw.wx0,bw.wx1,bw.wx2), vec3<f32>(bw.wy0,bw.wy1,bw.wy2), vec3<f32>(bw.wz0,bw.wz1,bw.wz2));

    let SLICE   = params.grid_X * params.grid_Y;
    let baseIdx = u32((base.z * i32(params.grid_Y) + base.y) * i32(params.grid_X) + base.x);

    var gvx = 0.0; var gvy = 0.0; var gvz = 0.0;
    var B00 = 0.0; var B01 = 0.0; var B02 = 0.0;
    var B10 = 0.0; var B11 = 0.0; var B12 = 0.0;
    var B20 = 0.0; var B21 = 0.0; var B22 = 0.0;

    for (var k: i32 = 0; k < 3; k++) {
        let wzk = wt.wz[k];
        let dzk = f32(base.z + k) - pos.z + 0.5;
        for (var j: i32 = 0; j < 3; j++) {
            let wyzk = wt.wy[j] * wzk;
            let dyj  = f32(base.y + j) - pos.y + 0.5;
            for (var i: i32 = 0; i < 3; i++) {
                let w    = wt.wx[i] * wyzk;
                let dxi  = f32(base.x + i) - pos.x + 0.5;
                let cidx = baseIdx + u32(k)*SLICE + u32(j)*params.grid_X + u32(i);

                let cvx = cell_mv_f32[cidx*3u+0u];
                let cvy = cell_mv_f32[cidx*3u+1u];
                let cvz = cell_mv_f32[cidx*3u+2u];

                let wvx = w * cvx; let wvy = w * cvy; let wvz = w * cvz;
                gvx += wvx; gvy += wvy; gvz += wvz;

                B00 += wvx*dxi; B01 += wvx*dyj; B02 += wvx*dzk;
                B10 += wvy*dxi; B11 += wvy*dyj; B12 += wvy*dzk;
                B20 += wvz*dxi; B21 += wvz*dyj; B22 += wvz*dzk;
            }
        }
    }

    let off = p * 9u;
    particle_affine[off+0u] = B00*4.0; particle_affine[off+1u] = B01*4.0; particle_affine[off+2u] = B02*4.0;
    particle_affine[off+3u] = B10*4.0; particle_affine[off+4u] = B11*4.0; particle_affine[off+5u] = B12*4.0;
    particle_affine[off+6u] = B20*4.0; particle_affine[off+7u] = B21*4.0; particle_affine[off+8u] = B22*4.0;

    var np = pos + params.dt * vec3<f32>(gvx, gvy, gvz);
    var nv = vec3<f32>(gvx, gvy, gvz);

    // Splash方式: 予測位置ベースの弱いバネ補正 (ref/boundary-condition-splash-style.md ステップ3)
    // 床(Y-min)からwall_min(3マス)の高さまでのゾーンに絞る。ハードクランプ「より前」の
    // 生のnp/nvを使う。Y >= floorZoneY で補正なし、floorZoneY から hard_min に向かって線形に補正強度を増やす。
    let floorZoneY = params.hard_min + params.wall_min;
    let xn = np + nv * params.dt * params.lookahead_k;
    if (xn.y < floorZoneY) {
        let zoneHeight = floorZoneY - params.hard_min;
        let distBelowFloor = floorZoneY - xn.y;
        let blend = clamp(distBelowFloor / zoneHeight, 0.0, 1.0);
        nv.y += params.wall_stiffness * blend * distBelowFloor;
    }

    // 動く壁 (AABB) の押し出し。床バネと同じ場所 (ハードクランプの前) で生のnp/nvを使う。
    for (var wi = 0u; wi < ${MAX_WALLS}u; wi++) {
        let wall = wallp.walls[wi];
        if (wall.min_active.w < 0.5) { continue; }   // 非activeのみスキップ
        let wmin = wall.min_active.xyz;
        let wmax = wall.max_pad.xyz;
        let wvel = wall.vel_pad.xyz;

        // バネ補正: lookahead予測位置が「壁に食い込む」(sdf<0) ときだけ法線方向へ減速材として
        // 効かせる。以前は近接ゾーン(wall_min=3セル)で常時押し出していたが、それだと壁際に
        // 「バネ vs 水圧」の釣り合いで決まる隙間ができ、静水圧が最大の床付近だけ隙間が潰れて
        // 不自然に狭く見える (上ほど広い=不均一)。外壁・床と同じく定常時の安定化は
        // UPDATE_GRID のIF速度帯に任せ、粒子は壁面に密着させる (全高で隙間ゼロ=均一)。
        // lookahead があるので、動く壁へ突っ込む粒子は実接触の前に減速される。
        let xnw = np + nv * params.dt * params.lookahead_k;
        let sxn = wall_sdf_normal(xnw, wmin, wmax);
        if (sxn.w < 0.0) {
            let dist  = -sxn.w;
            let blend = clamp(dist / params.wall_min, 0.0, 1.0);
            nv += sxn.xyz * (params.wall_stiffness * blend * dist);
        }

        // 安全弁: 更新後位置がなお壁内部なら最近傍表面へ投影し、壁との相対速度の
        // 法線成分 (壁に向かう向き=負のみ) をゼロ化。sn.w<0 なので np - n*sn.w が外向き押し出し。
        // 投影量は1ステップ0.5セルまでに制限する。深く埋まった粒子を一気に表面へテレポート
        // すると界面に密度スパイクが立ち、EOS圧力が粒子を吹き上げる自励噴出ループになる。
        // 0.5セル/ステップの漸進排出なら圧力場が追従できる (深く埋まった粒子のテレポートによる
        // 密度スパイク噴出を防ぐ汎用の安定化)。
        let snp = wall_sdf_normal(np, wmin, wmax);
        if (snp.w < 0.0) {
            np = np - snp.xyz * max(snp.w, -0.5);
            let vn = dot(nv - wvel, snp.xyz);
            if (vn < 0.0) { nv -= vn * snp.xyz; }
        }
    }

    // ハードクランプは最後に安全弁として適用 (バネで押し戻された後もなお壁を超える場合のみ発動)
    let rx = hard_clamp_axis(np.x, nv.x, params.hard_min, params.hard_max_x);
    let ry = hard_clamp_axis(np.y, nv.y, params.hard_min, params.hard_max_y);
    let rz = hard_clamp_axis(np.z, nv.z, params.hard_min, params.hard_max_z);
    np = vec3<f32>(rx.x, ry.x, rz.x);
    nv = vec3<f32>(rx.y, ry.y, rz.y);

    particle_pos[p] = vec4<f32>(np, 0.0);
    particle_vel[p] = vec4<f32>(nv, 0.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────
// Diffuse particles (spray/foam/bubble) — free-list pool, see
// ref/diffuse-particles-gpu-migration.md. One thread = one main-fluid particle
// (generate) or one pool slot (advect); dead slots are recycled via an atomic
// stack (free_list / free_list_top) instead of CPU-style swap-remove, which
// would race across threads.
// ─────────────────────────────────────────────────────────────────────────

// generate reads the main fluid's already-written particle_density (from
// P2G_MOM) and recomputes the EOS pressure from it in place, rather than
// storing a redundant particle_pressure buffer — the pressure term has no
// viscosity dependence, so it's an exact reproduction of P2G_MOM's value.
// This keeps this shader's storage-buffer count at 8, the WebGPU-guaranteed
// minimum (see WGSL_COMPACT's comment for the same constraint elsewhere).
export const WGSL_DIFFUSE_GENERATE = WGSL_COMMON + /* wgsl */`
const DIFFUSE_SPRAY: u32 = 0u;
const DIFFUSE_FOAM: u32 = 1u;

struct DiffuseParticle {
    pos      : vec4<f32>,
    vel      : vec4<f32>,
    ptype    : u32,
    alive    : u32,
    lifetime : f32,
    _pad     : f32,
}

struct GenParams {
    ke_threshold        : f32,
    spawn_rate_k        : f32,
    density_spray_max   : f32,
    density_bubble_min  : f32,
    lifetime_foam       : f32,
    pressure_threshold  : f32,
    crest_dot_threshold : f32,
    seed                : u32,
}

@group(0) @binding(0) var<uniform>             params            : Params;
@group(0) @binding(1) var<uniform>             gen               : GenParams;
@group(0) @binding(2) var<storage, read>       particle_pos      : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       particle_vel      : array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       particle_mass     : array<f32>;
@group(0) @binding(5) var<storage, read>       particle_density  : array<f32>;
@group(0) @binding(6) var<storage, read>       cell_mass_f32     : array<f32>;
@group(0) @binding(7) var<storage, read_write> diffuse_particles : array<DiffuseParticle>;
@group(0) @binding(8) var<storage, read_write> free_list         : array<u32>;
@group(0) @binding(9) var<storage, read_write> free_list_top     : atomic<i32>;

fn diffuse_rand(seed: u32) -> f32 {
    var x = seed;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    return f32(x) / 4294967295.0;
}

// Density-gradient normal estimate (CPU _estimateNormal). w=1 marks a valid
// result in place of the CPU version's null return.
fn estimate_normal(pos: vec3<f32>) -> vec4<f32> {
    let g = vec3<i32>(floor(pos));
    let X = i32(params.grid_X); let Y = i32(params.grid_Y); let Z = i32(params.grid_Z);
    if (g.x < 1 || g.x >= X - 1 || g.y < 1 || g.y >= Y - 1 || g.z < 1 || g.z >= Z - 1) {
        return vec4<f32>(0.0);
    }
    let idxL = u32((g.z * Y + g.y) * X + (g.x - 1));
    let idxR = u32((g.z * Y + g.y) * X + (g.x + 1));
    let idxD = u32((g.z * Y + (g.y - 1)) * X + g.x);
    let idxU = u32((g.z * Y + (g.y + 1)) * X + g.x);
    let idxB = u32(((g.z - 1) * Y + g.y) * X + g.x);
    let idxF = u32(((g.z + 1) * Y + g.y) * X + g.x);

    var n = vec3<f32>(
        cell_mass_f32[idxL] - cell_mass_f32[idxR],
        cell_mass_f32[idxD] - cell_mass_f32[idxU],
        cell_mass_f32[idxB] - cell_mass_f32[idxF],
    );
    let len = length(n);
    if (len < 1e-6) { return vec4<f32>(0.0); }
    return vec4<f32>(n / len, 1.0);
}

// Pops one free slot off the stack; returns -1 if the pool is exhausted
// (and restores the count it speculatively took).
fn try_spawn() -> i32 {
    let prevTop = atomicSub(&free_list_top, 1);
    if (prevTop <= 0) {
        atomicAdd(&free_list_top, 1);
        return -1;
    }
    return i32(free_list[u32(prevTop - 1)]);
}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let density = particle_density[p];
    if (density <= 0.0 || density > gen.density_bubble_min) { return; }

    let vel    = particle_vel[p].xyz;
    let speed2 = dot(vel, vel);
    let speed  = sqrt(speed2);
    let ke     = 0.5 * particle_mass[p] * speed2;

    var pressure = params.stiffness * (pow(density / params.rest_density, params.eos_power) - 1.0);
    let pFloor = select(0.0, COHESION_PRESSURE, density > params.rest_density * COHESION_DENSITY_GATE);
    pressure = max(pressure, pFloor);
    let trappedAir = max(0.0, pressure - gen.pressure_threshold);

    let pos = particle_pos[p].xyz;
    let normalResult = estimate_normal(pos);
    var waveCrest = 0.0;
    if (speed > 1e-4 && normalResult.w > 0.5) {
        let dotVal = dot(vel / speed, normalResult.xyz);
        if (dotVal > gen.crest_dot_threshold) { waveCrest = dotVal * speed; }
    }

    let score = ke + trappedAir * 2.0 + waveCrest * 1.5;
    if (score < gen.ke_threshold) { return; }

    let randSeed    = p * 9781u + gen.seed;
    let spawnChance = min(1.0, (score - gen.ke_threshold) * gen.spawn_rate_k * params.dt);
    if (diffuse_rand(randSeed) > spawnChance) { return; }

    let slot = try_spawn();
    if (slot < 0) { return; }

    let jitter = 0.3;
    let jx = (diffuse_rand(randSeed + 1u) - 0.5) * jitter * speed;
    let jy = (diffuse_rand(randSeed + 2u) - 0.5) * jitter * speed;
    let jz = (diffuse_rand(randSeed + 3u) - 0.5) * jitter * speed;

    var ptype: u32 = DIFFUSE_FOAM;
    if (density < gen.density_spray_max) { ptype = DIFFUSE_SPRAY; }

    let lifetime = select(999.0, gen.lifetime_foam * min(1.0, score / gen.ke_threshold), ptype == DIFFUSE_FOAM);

    var newP: DiffuseParticle;
    newP.pos      = vec4<f32>(pos, 0.0);
    newP.vel      = vec4<f32>(vel + vec3<f32>(jx, jy, jz), 0.0);
    newP.ptype    = ptype;
    newP.alive    = 1u;
    newP.lifetime = lifetime;
    newP._pad     = 0.0;

    diffuse_particles[u32(slot)] = newP;
}
`;

export const WGSL_DIFFUSE_ADVECT = WGSL_COMMON + /* wgsl */`
const DIFFUSE_SPRAY: u32 = 0u;
const DIFFUSE_FOAM: u32 = 1u;
const DIFFUSE_BUBBLE: u32 = 2u;

struct DiffuseParticle {
    pos      : vec4<f32>,
    vel      : vec4<f32>,
    ptype    : u32,
    alive    : u32,
    lifetime : f32,
    _pad     : f32,
}

struct AdvectParams {
    diffuse_max : u32,
    buoyancy    : f32,
    drag        : f32,
    _pad        : f32,
}

@group(0) @binding(0) var<uniform>             params            : Params;
@group(0) @binding(1) var<uniform>             adv               : AdvectParams;
@group(0) @binding(2) var<storage, read_write> diffuse_particles : array<DiffuseParticle>;
@group(0) @binding(3) var<storage, read>       cell_mv_f32       : array<f32>;
@group(0) @binding(4) var<storage, read_write> free_list         : array<u32>;
@group(0) @binding(5) var<storage, read_write> free_list_top     : atomic<i32>;
@group(0) @binding(6) var<uniform>             wallp             : WallParams;

// Same trilinear quadratic-B-spline sample as G2P, against the main fluid's
// already gravity-and-boundary-applied grid velocity (cell_mv_f32).
fn sample_grid_velocity(pos: vec3<f32>) -> vec3<f32> {
    let cellI = vec3<i32>(floor(pos));
    let base  = cellI - vec3<i32>(1);
    let f     = pos - vec3<f32>(cellI) - vec3<f32>(0.5);
    let wt    = quadratic_weights(f);

    var v = vec3<f32>(0.0);
    for (var k: i32 = 0; k < 3; k++) {
        let cz = base.z + k;
        for (var j: i32 = 0; j < 3; j++) {
            let cy = base.y + j;
            for (var i: i32 = 0; i < 3; i++) {
                let cx = base.x + i;
                let idx = cell_idx_checked(cx, cy, cz, params.grid_X, params.grid_Y, params.grid_Z);
                if (idx < 0) { continue; }
                let w = wt.wx[i] * wt.wy[j] * wt.wz[k];
                v.x += w * cell_mv_f32[u32(idx) * 3u + 0u];
                v.y += w * cell_mv_f32[u32(idx) * 3u + 1u];
                v.z += w * cell_mv_f32[u32(idx) * 3u + 2u];
            }
        }
    }
    return v;
}

fn kill_particle(i: u32) {
    diffuse_particles[i].alive = 0u;
    let slot = atomicAdd(&free_list_top, 1);
    free_list[u32(slot)] = i;
}

// Damped reflection off one axis' walls (x/z side walls, y_max ceiling).
const DIFFUSE_RESTITUTION: f32 = 0.3;
fn reflect_axis(p: f32, v: f32, lo: f32, hi: f32) -> vec2<f32> {
    var np = p; var nv = v;
    if (np < lo) { np = lo + (lo - np); nv = -nv * DIFFUSE_RESTITUTION; }
    else if (np > hi) { np = hi - (np - hi); nv = -nv * DIFFUSE_RESTITUTION; }
    return vec2<f32>(clamp(np, lo, hi), nv);
}

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= adv.diffuse_max) { return; }
    if (diffuse_particles[i].alive == 0u) { return; }

    var p = diffuse_particles[i];
    let pos3 = p.pos.xyz;

    if (p.ptype == DIFFUSE_SPRAY) {
        p.vel.y += params.gravity * params.dt;
        p.pos = vec4<f32>(pos3 + params.dt * p.vel.xyz, 0.0);

    } else if (p.ptype == DIFFUSE_FOAM) {
        let gv = sample_grid_velocity(pos3);
        p.pos = vec4<f32>(pos3 + params.dt * gv, 0.0);
        p.lifetime -= params.dt;
        if (p.lifetime <= 0.0) {
            kill_particle(i);
            return;
        }

    } else if (p.ptype == DIFFUSE_BUBBLE) {
        let gv = sample_grid_velocity(pos3);
        p.vel.x += params.dt * adv.drag * (gv.x - p.vel.x);
        p.vel.y += params.dt * (adv.buoyancy * -params.gravity + adv.drag * (gv.y - p.vel.y));
        p.vel.z += params.dt * adv.drag * (gv.z - p.vel.z);
        p.pos = vec4<f32>(pos3 + params.dt * p.vel.xyz, 0.0);
    }

    // 動く壁 (AABB) に入った diffuse 粒子を最も浅い貫通軸で押し出し、その軸(=法線)方向の
    // 速度成分を減衰反射 (外壁の reflect_axis と同じ DIFFUSE_RESTITUTION、壁速度基準)。
    // 全active壁 (w >= 0.5) に適用。diffuse は装飾トレーサーで壁を突き抜けると見栄えが悪い。
    for (var wi = 0u; wi < ${MAX_WALLS}u; wi++) {
        let wall = wallp.walls[wi];
        if (wall.min_active.w < 0.5) { continue; }
        let sn = wall_sdf_normal(p.pos.xyz, wall.min_active.xyz, wall.max_pad.xyz);
        if (sn.w < 0.0) {
            p.pos = vec4<f32>(p.pos.xyz - sn.xyz * sn.w, 0.0);   // 最近傍表面へ押し出し
            let wvel = wall.vel_pad.xyz;
            let vrel = p.vel.xyz - wvel;
            let vn   = dot(vrel, sn.xyz);
            if (vn < 0.0) {
                let newvrel = vrel - (1.0 + DIFFUSE_RESTITUTION) * vn * sn.xyz;
                p.vel = vec4<f32>(wvel + newvrel, 0.0);
            }
        }
    }

    // Boundary: same [hard_min, hard_max] box as the main fluid's hard clamp.
    // Floor (y_min) despawns the particle (it has "landed"/been absorbed).
    // Side walls (x/z) and the ceiling (y_max) reflect instead, since those
    // exits are just simulation overshoot, not a natural resting place.
    if (p.pos.y < params.hard_min) {
        kill_particle(i);
        return;
    }
    let rx = reflect_axis(p.pos.x, p.vel.x, params.hard_min, params.hard_max_x);
    let ry = reflect_axis(p.pos.y, p.vel.y, params.hard_min, params.hard_max_y);
    let rz = reflect_axis(p.pos.z, p.vel.z, params.hard_min, params.hard_max_z);
    p.pos = vec4<f32>(rx.x, ry.x, rz.x, 0.0);
    p.vel = vec4<f32>(rx.y, ry.y, rz.y, 0.0);

    diffuse_particles[i] = p;
}
`;
