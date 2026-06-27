// MLS-MPM 3D Fluid Simulation — WebGPU Compute Shader version
// Self-contained: FluidGPU (compute) + Fluid (CPU setup only) + renderer + loop

const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.innerWidth < 768;
const WG_SIZE = isMobile ? 32 : 64;
// Mobile WebGPU implementations may not handle DPR-scaled canvases reliably; use 1x there.
const dpr = isMobile ? 1 : Math.min(window.devicePixelRatio || 1, 2);
let logicalW = window.innerWidth, logicalH = window.innerHeight;

const WGSL_COMMON = /* wgsl */`
const FIXED: f32 = 100000.0;

fn encode(v: f32) -> i32 { return i32(v * FIXED); }
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
    _pad0      : f32,
    _pad1      : f32,
}
`;

// ── clear grid ──────────────────────────────────────────────────────────────
const WGSL_CLEAR = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> cell_mv   : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> cell_mass : array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             params    : Params;

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

// ── p2g mass ─────────────────────────────────────────────────────────────────
const WGSL_P2G_MASS = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read>       particle_pos  : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       particle_mass : array<f32>;
@group(0) @binding(2) var<storage, read_write> cell_mass     : array<atomic<i32>>;
@group(0) @binding(3) var<uniform>             params        : Params;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pos = particle_pos[p].xyz;
    let pm  = particle_mass[p];

    let cellI = vec3<i32>(floor(pos));
    let base  = cellI - vec3<i32>(1);
    let f     = pos - vec3<f32>(cellI) - vec3<f32>(0.5);

    var wx: array<f32, 3>;
    wx[0] = 0.5 * (0.5 - f.x) * (0.5 - f.x);
    wx[1] = 0.75 - f.x * f.x;
    wx[2] = 0.5 * (0.5 + f.x) * (0.5 + f.x);
    var wy: array<f32, 3>;
    wy[0] = 0.5 * (0.5 - f.y) * (0.5 - f.y);
    wy[1] = 0.75 - f.y * f.y;
    wy[2] = 0.5 * (0.5 + f.y) * (0.5 + f.y);
    var wz: array<f32, 3>;
    wz[0] = 0.5 * (0.5 - f.z) * (0.5 - f.z);
    wz[1] = 0.75 - f.z * f.z;
    wz[2] = 0.5 * (0.5 + f.z) * (0.5 + f.z);

    for (var k: i32 = 0; k < 3; k++) {
        let cz = base.z + k;
        if (cz < 0 || cz >= i32(params.grid_Z)) { continue; }
        for (var j: i32 = 0; j < 3; j++) {
            let cy = base.y + j;
            if (cy < 0 || cy >= i32(params.grid_Y)) { continue; }
            for (var i: i32 = 0; i < 3; i++) {
                let cx = base.x + i;
                if (cx < 0 || cx >= i32(params.grid_X)) { continue; }
                let w   = wx[i] * wy[j] * wz[k];
                let idx = u32((cz * i32(params.grid_Y) + cy) * i32(params.grid_X) + cx);
                atomicAdd(&cell_mass[idx], encode(w * pm));
            }
        }
    }
}
`;

// ── p2g momentum ─────────────────────────────────────────────────────────────
const WGSL_P2G_MOM = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read>       particle_pos    : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       particle_vel    : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       particle_affine : array<f32>;
@group(0) @binding(3) var<storage, read>       particle_mass   : array<f32>;
@group(0) @binding(4) var<storage, read_write> cell_mass       : array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> cell_mv         : array<atomic<i32>>;
@group(0) @binding(6) var<uniform>             params          : Params;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pos  = particle_pos[p].xyz;
    let vel  = particle_vel[p].xyz;
    let mass = particle_mass[p];
    let off  = p * 9u;
    let a00 = particle_affine[off+0u]; let a01 = particle_affine[off+1u]; let a02 = particle_affine[off+2u];
    let a10 = particle_affine[off+3u]; let a11 = particle_affine[off+4u]; let a12 = particle_affine[off+5u];
    let a20 = particle_affine[off+6u]; let a21 = particle_affine[off+7u]; let a22 = particle_affine[off+8u];

    let cellI = vec3<i32>(floor(pos));
    let base  = cellI - vec3<i32>(1);
    let f     = pos - vec3<f32>(cellI) - vec3<f32>(0.5);

    var wx: array<f32, 3>;
    wx[0] = 0.5 * (0.5 - f.x) * (0.5 - f.x);
    wx[1] = 0.75 - f.x * f.x;
    wx[2] = 0.5 * (0.5 + f.x) * (0.5 + f.x);
    var wy: array<f32, 3>;
    wy[0] = 0.5 * (0.5 - f.y) * (0.5 - f.y);
    wy[1] = 0.75 - f.y * f.y;
    wy[2] = 0.5 * (0.5 + f.y) * (0.5 + f.y);
    var wz: array<f32, 3>;
    wz[0] = 0.5 * (0.5 - f.z) * (0.5 - f.z);
    wz[1] = 0.75 - f.z * f.z;
    wz[2] = 0.5 * (0.5 + f.z) * (0.5 + f.z);

    // gather density
    let SLICE = params.grid_X * params.grid_Y;
    let baseIdx = u32((base.z * i32(params.grid_Y) + base.y) * i32(params.grid_X) + base.x);
    var density = 0.0;
    for (var k: i32 = 0; k < 3; k++) {
        for (var j: i32 = 0; j < 3; j++) {
            for (var i: i32 = 0; i < 3; i++) {
                let cidx = baseIdx + u32(k) * SLICE + u32(j) * params.grid_X + u32(i);
                density += wx[i] * wy[j] * wz[k] * decode(atomicLoad(&cell_mass[cidx]));
            }
        }
    }
    if (density <= 1e-8) { return; }

    let vol = mass / density;
    var pressure = params.stiffness * (pow(density / params.rest_density, params.eos_power) - 1.0);
    if (pressure < -0.1) { pressure = -0.1; }

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
        let wzk = wz[k]; let dzk = dzs[k];
        for (var j: i32 = 0; j < 3; j++) {
            let wyzk = wy[j] * wzk; let dyj = dys[j];
            for (var i: i32 = 0; i < 3; i++) {
                let w    = wx[i] * wyzk;
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

const WGSL_APPLY_HAND = WGSL_COMMON + /* wgsl */`
struct HandParams {
    px:f32, py:f32, pz:f32, radius:f32,
    vx:f32, vy:f32, vz:f32, is_on:u32,
    strength:f32, _q0:f32, _q1:f32, _q2:f32,
};
@group(0) @binding(0) var<storage, read_write> cell_mv : array<atomic<i32>>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var<uniform> hand   : HandParams;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.grid_X * params.grid_Y * params.grid_Z;
    if (idx >= total || hand.is_on == 0u) { return; }
    let SLICE = params.grid_X * params.grid_Y;
    let cx = f32(idx % params.grid_X) + 0.5;
    let cy = f32((idx / params.grid_X) % params.grid_Y) + 0.5;
    let cz = f32(idx / SLICE) + 0.5;
    let d = distance(vec3<f32>(cx, cy, cz), vec3<f32>(hand.px, hand.py, hand.pz));
    if (d >= hand.radius) { return; }
    let t = 1.0 - d / hand.radius;
    let w = t * t * hand.strength;
    atomicAdd(&cell_mv[idx*3u+0u], encode(hand.vx * w));
    atomicAdd(&cell_mv[idx*3u+1u], encode(hand.vy * w));
    atomicAdd(&cell_mv[idx*3u+2u], encode(hand.vz * w));
}`;

// ── update grid ───────────────────────────────────────────────────────────────
const WGSL_UPDATE_GRID = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> cell_mv   : array<atomic<i32>>;
@group(0) @binding(1) var<storage, read_write> cell_mass : array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             params    : Params;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.grid_X * params.grid_Y * params.grid_Z) { return; }

    let cm = decode(atomicLoad(&cell_mass[idx]));
    if (cm <= 0.0) { return; }

    var vx = decode(atomicLoad(&cell_mv[idx*3u+0u])) / cm;
    var vy = decode(atomicLoad(&cell_mv[idx*3u+1u])) / cm;
    var vz = decode(atomicLoad(&cell_mv[idx*3u+2u])) / cm;

    vy += params.gravity * params.dt;

    let cx = i32(idx % params.grid_X);
    let cy = i32((idx / params.grid_X) % params.grid_Y);
    let cz = i32(idx / (params.grid_X * params.grid_Y));

    if (cx < 2                       && vx < 0.0) { vx = 0.0; }
    if (cx > i32(params.grid_X) - 3  && vx > 0.0) { vx = 0.0; }
    if (cy < 2                       && vy < 0.0) { vy = 0.0; }
    if (cy > i32(params.grid_Y) - 3  && vy > 0.0) { vy = 0.0; }
    if (cz < 2                       && vz < 0.0) { vz = 0.0; }
    if (cz > i32(params.grid_Z) - 3  && vz > 0.0) { vz = 0.0; }

    atomicStore(&cell_mv[idx*3u+0u], encode(vx));
    atomicStore(&cell_mv[idx*3u+1u], encode(vy));
    atomicStore(&cell_mv[idx*3u+2u], encode(vz));
}
`;

// ── g2p ───────────────────────────────────────────────────────────────────────
const WGSL_G2P = WGSL_COMMON + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> particle_pos    : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> particle_vel    : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> particle_affine : array<f32>;
@group(0) @binding(3) var<storage, read_write> cell_mv         : array<atomic<i32>>;
@group(0) @binding(4) var<uniform>             params          : Params;

@compute @workgroup_size(${WG_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.particle_num) { return; }

    let pos   = particle_pos[p].xyz;
    let cellI = vec3<i32>(floor(pos));
    let base  = cellI - vec3<i32>(1);
    let f     = pos - vec3<f32>(cellI) - vec3<f32>(0.5);

    var wx: array<f32, 3>;
    wx[0] = 0.5 * (0.5 - f.x) * (0.5 - f.x);
    wx[1] = 0.75 - f.x * f.x;
    wx[2] = 0.5 * (0.5 + f.x) * (0.5 + f.x);
    var wy: array<f32, 3>;
    wy[0] = 0.5 * (0.5 - f.y) * (0.5 - f.y);
    wy[1] = 0.75 - f.y * f.y;
    wy[2] = 0.5 * (0.5 + f.y) * (0.5 + f.y);
    var wz: array<f32, 3>;
    wz[0] = 0.5 * (0.5 - f.z) * (0.5 - f.z);
    wz[1] = 0.75 - f.z * f.z;
    wz[2] = 0.5 * (0.5 + f.z) * (0.5 + f.z);

    let SLICE   = params.grid_X * params.grid_Y;
    let baseIdx = u32((base.z * i32(params.grid_Y) + base.y) * i32(params.grid_X) + base.x);

    var gvx = 0.0; var gvy = 0.0; var gvz = 0.0;
    var B00 = 0.0; var B01 = 0.0; var B02 = 0.0;
    var B10 = 0.0; var B11 = 0.0; var B12 = 0.0;
    var B20 = 0.0; var B21 = 0.0; var B22 = 0.0;

    for (var k: i32 = 0; k < 3; k++) {
        let wzk = wz[k];
        let dzk = f32(base.z + k) - pos.z + 0.5;
        for (var j: i32 = 0; j < 3; j++) {
            let wyzk = wy[j] * wzk;
            let dyj  = f32(base.y + j) - pos.y + 0.5;
            for (var i: i32 = 0; i < 3; i++) {
                let w    = wx[i] * wyzk;
                let dxi  = f32(base.x + i) - pos.x + 0.5;
                let cidx = baseIdx + u32(k)*SLICE + u32(j)*params.grid_X + u32(i);

                let cvx = decode(atomicLoad(&cell_mv[cidx*3u+0u]));
                let cvy = decode(atomicLoad(&cell_mv[cidx*3u+1u]));
                let cvz = decode(atomicLoad(&cell_mv[cidx*3u+2u]));

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

    if (np.x < params.hard_min)  { np.x = params.hard_min;  nv.x = 0.0; }
    if (np.x > params.hard_max_x){ np.x = params.hard_max_x; nv.x = 0.0; }
    if (np.y < params.hard_min)  { np.y = params.hard_min;  nv.y = 0.0; }
    if (np.y > params.hard_max_y){ np.y = params.hard_max_y; nv.y = 0.0; }
    if (np.z < params.hard_min)  { np.z = params.hard_min;  nv.z = 0.0; }
    if (np.z > params.hard_max_z){ np.z = params.hard_max_z; nv.z = 0.0; }

    particle_pos[p] = vec4<f32>(np, 0.0);
    particle_vel[p] = vec4<f32>(nv, 0.0);
}
`;

// ─────────────────────────────────────────────────────────────────────────────

class FluidGPU {
    constructor(device, fluidCPU) {
        this.device = device;

        this.grid_X_num = fluidCPU.grid_X_num;
        this.grid_Y_num = fluidCPU.grid_Y_num;
        this.grid_Z_num = fluidCPU.grid_Z_num;
        this.grid_num = fluidCPU.grid_num;
        this.particle_num = fluidCPU.particle_num;
        this.active_particle_num = fluidCPU.active_particle_num;

        this.DT = fluidCPU.DT;
        this.REST_DENSITY = fluidCPU.REST_DENSITY;
        this.STIFFNESS = fluidCPU.STIFFNESS;
        this.EOS_POWER = fluidCPU.EOS_POWER;
        this.VISCOSITY = fluidCPU.VISCOSITY;
        this.GRAVITY = fluidCPU.GRAVITY;
        this.SUBSTEPS = fluidCPU.SUBSTEPS;

        const HM = 2;
        this.HARD_MIN = HM;
        this.HARD_MAX_X = this.grid_X_num - HM;
        this.HARD_MAX_Y = this.grid_Y_num - HM;
        this.HARD_MAX_Z = this.grid_Z_num - HM;

        this._createBuffers(fluidCPU);
        this._createPipelines();
    }

    _pack3to4(arr3) {
        const n = arr3.length / 3;
        const arr4 = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
            arr4[i * 4] = arr3[i * 3];
            arr4[i * 4 + 1] = arr3[i * 3 + 1];
            arr4[i * 4 + 2] = arr3[i * 3 + 2];
        }
        return arr4;
    }

    _createBuffers(fluidCPU) {
        const d = this.device;
        const N = this.particle_num;
        const G = this.grid_num;
        const SV = GPUBufferUsage.STORAGE;
        const CD = GPUBufferUsage.COPY_DST;
        const CS = GPUBufferUsage.COPY_SRC;
        const VX = GPUBufferUsage.VERTEX;
        const UN = GPUBufferUsage.UNIFORM;

        this.particlePosBuffer = d.createBuffer({ size: N * 16, usage: SV | CD | CS | VX });
        this.particleVelBuffer = d.createBuffer({ size: N * 16, usage: SV | CD | CS | VX });
        this.particleAffineBuffer = d.createBuffer({ size: N * 9 * 4, usage: SV | CD | CS });
        this.particleMassBuffer = d.createBuffer({ size: N * 4, usage: SV | CD });
        this.cellMvBuffer = d.createBuffer({ size: G * 3 * 4, usage: SV | CD | CS });
        this.cellMassBuffer = d.createBuffer({ size: G * 4, usage: SV | CD | CS });
        this.paramsBuffer = d.createBuffer({ size: 64, usage: UN | CD });
        this.handParamsBuffer = d.createBuffer({ size: 48, usage: UN | CD });

        // pre-allocated CPU buffers reused every frame to avoid GC pressure
        this._paramsBuf = new ArrayBuffer(64);
        this._paramsU32 = new Uint32Array(this._paramsBuf);
        this._paramsF32 = new Float32Array(this._paramsBuf);
        this._handBuf = new ArrayBuffer(48);
        this._handF32 = new Float32Array(this._handBuf);
        this._handU32 = new Uint32Array(this._handBuf);
        this.handActive = false;

        d.queue.writeBuffer(this.particlePosBuffer, 0, this._pack3to4(fluidCPU.particle_pos));
        d.queue.writeBuffer(this.particleVelBuffer, 0, this._pack3to4(fluidCPU.particle_vel));
        d.queue.writeBuffer(this.particleAffineBuffer, 0, fluidCPU.particle_affine);
        d.queue.writeBuffer(this.particleMassBuffer, 0, fluidCPU.particle_mass);
    }

    _writeParamsBuffer(dt) {
        this._paramsU32[0] = this.grid_X_num; this._paramsU32[1] = this.grid_Y_num; this._paramsU32[2] = this.grid_Z_num;
        this._paramsU32[3] = this.active_particle_num;
        this._paramsF32[4] = dt; this._paramsF32[5] = this.GRAVITY; this._paramsF32[6] = this.REST_DENSITY;
        this._paramsF32[7] = this.STIFFNESS; this._paramsF32[8] = this.EOS_POWER; this._paramsF32[9] = this.VISCOSITY;
        this._paramsF32[10] = this.HARD_MIN; this._paramsF32[11] = this.HARD_MAX_X; this._paramsF32[12] = this.HARD_MAX_Y;
        this._paramsF32[13] = this.HARD_MAX_Z;
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this._paramsBuf);
    }

    _createPipelines() {
        const d = this.device;
        const mkP = (code) => d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code }), entryPoint: 'main' },
        });
        this._clearPipeline = mkP(WGSL_CLEAR);
        this._p2gMassPipeline = mkP(WGSL_P2G_MASS);
        this._p2gMomPipeline = mkP(WGSL_P2G_MOM);
        this._updateGridPipeline = mkP(WGSL_UPDATE_GRID);
        this._g2pPipeline = mkP(WGSL_G2P);
        this._applyHandPipeline = mkP(WGSL_APPLY_HAND);

        const mkBG = (pipeline, entries) =>
            d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const e = (binding, buffer) => ({ binding, resource: { buffer } });

        this._clearBG = mkBG(this._clearPipeline, [
            e(0, this.cellMvBuffer), e(1, this.cellMassBuffer), e(2, this.paramsBuffer),
        ]);
        this._p2gMassBG = mkBG(this._p2gMassPipeline, [
            e(0, this.particlePosBuffer), e(1, this.particleMassBuffer),
            e(2, this.cellMassBuffer), e(3, this.paramsBuffer),
        ]);
        this._p2gMomBG = mkBG(this._p2gMomPipeline, [
            e(0, this.particlePosBuffer), e(1, this.particleVelBuffer),
            e(2, this.particleAffineBuffer), e(3, this.particleMassBuffer),
            e(4, this.cellMassBuffer), e(5, this.cellMvBuffer), e(6, this.paramsBuffer),
        ]);
        this._updateGridBG = mkBG(this._updateGridPipeline, [
            e(0, this.cellMvBuffer), e(1, this.cellMassBuffer), e(2, this.paramsBuffer),
        ]);
        this._g2pBG = mkBG(this._g2pPipeline, [
            e(0, this.particlePosBuffer), e(1, this.particleVelBuffer),
            e(2, this.particleAffineBuffer), e(3, this.cellMvBuffer), e(4, this.paramsBuffer),
        ]);
        this._applyHandBG = mkBG(this._applyHandPipeline, [
            e(0, this.cellMvBuffer), e(1, this.paramsBuffer), e(2, this.handParamsBuffer),
        ]);
    }

    updateHand(pos, vel, radius, strength, active) {
        this.handActive = active;
        this._handF32[0] = pos[0]; this._handF32[1] = pos[1]; this._handF32[2] = pos[2]; this._handF32[3] = radius;
        this._handF32[4] = vel[0]; this._handF32[5] = vel[1]; this._handF32[6] = vel[2]; this._handU32[7] = active ? 1 : 0;
        this._handF32[8] = strength;
        this.device.queue.writeBuffer(this.handParamsBuffer, 0, this._handBuf);
    }

    simFrame(cmd) {
        const sub_dt = this.DT / this.SUBSTEPS;
        const wgP = Math.ceil(this.active_particle_num / WG_SIZE);
        const wgG = Math.ceil(this.grid_num / WG_SIZE);
        this._writeParamsBuffer(sub_dt);
        for (let step = 0; step < this.SUBSTEPS; step++) {
            const run = (pip, bg, wg) => {
                const pass = cmd.beginComputePass();
                pass.setPipeline(pip); pass.setBindGroup(0, bg);
                pass.dispatchWorkgroups(wg); pass.end();
            };
            run(this._clearPipeline, this._clearBG, wgG);
            run(this._p2gMassPipeline, this._p2gMassBG, wgP);
            run(this._p2gMomPipeline, this._p2gMomBG, wgP);
            run(this._updateGridPipeline, this._updateGridBG, wgG);
            if (this.handActive) run(this._applyHandPipeline, this._applyHandBG, wgG);
            run(this._g2pPipeline, this._g2pBG, wgP);
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Canvas / overlay setup
// ─────────────────────────────────────────────────────────────
const c = document.querySelector("#gpu");
const overlay = document.querySelector("#overlay");
const octx = overlay.getContext("2d");
overlay.style.touchAction = "none";

function resizeCanvases() {
    logicalW = window.innerWidth;
    logicalH = window.innerHeight;
    c.width = overlay.width = Math.round(logicalW * dpr);
    c.height = overlay.height = Math.round(logicalH * dpr);
    // Scale the 2D context so all overlay drawing uses CSS-pixel coordinates
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
resizeCanvases();
window.addEventListener("resize", resizeCanvases);


// ─────────────────────────────────────────────────────────────
//  CPU Fluid class (used only for particle initialisation)
// ─────────────────────────────────────────────────────────────
class Fluid {
    constructor(aspectX, aspectY, aspectZ, particle_radius, particle_num) {
        this.aspectX = aspectX;
        this.aspectY = aspectY;
        this.aspectZ = aspectZ;
        this.particle_radius = particle_radius;
        this.particle_num = particle_num;
        this.active_particle_num = 0;
        this.norm_to_grid = 1 / (particle_radius * 4);
        this.grid_X_num = Math.ceil(aspectX * this.norm_to_grid);
        this.grid_Y_num = Math.ceil(aspectY * this.norm_to_grid);
        this.grid_Z_num = Math.ceil(aspectZ * this.norm_to_grid);
        this.grid_num = this.grid_X_num * this.grid_Y_num * this.grid_Z_num;

        this.particle_pos = new Float32Array(particle_num * 3);
        this.particle_vel = new Float32Array(particle_num * 3);
        this.particle_affine = new Float32Array(particle_num * 9);
        this.particle_mass = new Float32Array(particle_num);

        this.DT = 0.1;
        this.REST_DENSITY = 8.0;
        this.STIFFNESS = 7.0;
        this.EOS_POWER = 7;
        this.VISCOSITY = 0.01;
        this.GRAVITY = -0.98;
        this.SUBSTEPS = 2;
    }

    // x0..z1: normalized 0–1 per axis (0 = near edge, 1 = far edge of each dimension)
    // Particles are placed every 0.5 grid cells starting from the block interior.
    // Continues from the current active_particle_num so multiple calls can stack.
    fill_block(x0, y0, z0, x1, y1, z1) {
        const step = 0.5, jitter = 0.05;
        const gx0 = x0 * this.grid_X_num, gx1 = x1 * this.grid_X_num;
        const gy0 = y0 * this.grid_Y_num, gy1 = y1 * this.grid_Y_num;
        const gz0 = z0 * this.grid_Z_num, gz1 = z1 * this.grid_Z_num;
        let p = this.active_particle_num;
        outer:
        for (let gz = gz0 + step / 2; gz < gz1; gz += step) {
            for (let gy = gy0 + step / 2; gy < gy1; gy += step) {
                for (let gx = gx0 + step / 2; gx < gx1; gx += step) {
                    if (p >= this.particle_num) break outer;
                    this.particle_pos[p * 3]     = gx + (Math.random() - 0.5) * jitter;
                    this.particle_pos[p * 3 + 1] = gy + (Math.random() - 0.5) * jitter;
                    this.particle_pos[p * 3 + 2] = gz + (Math.random() - 0.5) * jitter;
                    this.particle_mass[p] = 1.0;
                    p++;
                }
            }
        }
        this.active_particle_num = p;
        return p;
    }
}

// ─────────────────────────────────────────────────────────────
//  Particle initialisation
// ─────────────────────────────────────────────────────────────
// Fixed 1.8:2:1.8 domain — independent of screen orientation so portrait phones get the same grid size as landscape
const fluid = new Fluid(1.8, 2, 1.8, isMobile ? 0.02 : 0.0125, isMobile ? 30000 : 120000);
fluid.fill_block(0.15, 0.05, 0.15,  0.85, 0.65, 0.85);

// ─────────────────────────────────────────────────────────────
//  Orbit camera
// ─────────────────────────────────────────────────────────────
const camera = {
    target: [fluid.grid_X_num / 2, fluid.grid_Y_num / 2, fluid.grid_Z_num / 2],
    theta: Math.PI / 2,
    phi: 0.15,
    radius: Math.max(fluid.grid_X_num, fluid.grid_Y_num, fluid.grid_Z_num) * 1.6,
    fovy: Math.PI / 4,
    near: 0.1,
    far: 10000,
};

// Hand interaction state (right-drag or Shift+left-drag)
const handState = {
    pos: [0, 0, 0], vel: [0, 0, 0],
    radius: 6.0, strength: 0.5, active: false,
};

const orbit = { active: false, pointerId: null, lastX: 0, lastY: 0 };
const iact = { active: false, pointerId: null };

overlay.addEventListener("contextmenu", (e) => e.preventDefault());

overlay.addEventListener("pointerdown", (e) => {
    const cv = cameraVectors(camera);
    const vp = mat4Multiply(
        mat4Perspective(camera.fovy, c.width / c.height, camera.near, camera.far),
        mat4LookAt(cv.eye, camera.target, [0, 1, 0])
    );

    // On mobile, tapping outside the projected simulation box triggers orbit instead of push
    let wantOrbit = e.button === 2;
    if (!wantOrbit && isMobile) {
        const b = simBoundsOnScreen(
            [fluid.grid_X_num, fluid.grid_Y_num, fluid.grid_Z_num],
            vp, logicalW, logicalH
        );
        wantOrbit = b !== null && (
            e.offsetX < b[0] || e.offsetX > b[2] ||
            e.offsetY < b[1] || e.offsetY > b[3]
        );
    }

    if (wantOrbit) {
        // Right drag (PC) or tap outside sim (mobile) = orbit
        orbit.active = true; orbit.pointerId = e.pointerId;
        orbit.lastX = e.clientX; orbit.lastY = e.clientY;
        overlay.setPointerCapture(e.pointerId);
        e.preventDefault();
    } else {
        // Left drag (PC) or tap inside sim (mobile) = push fluid
        iact.active = true; iact.pointerId = e.pointerId;
        // Unproject click position onto the plane through camera.target
        const p0 = screenToWorld(e.offsetX, e.offsetY, camera, cv);
        handState.pos = [
            Math.max(0, Math.min(fluid.grid_X_num, p0[0])),
            Math.max(0, Math.min(fluid.grid_Y_num, p0[1])),
            Math.max(0, Math.min(fluid.grid_Z_num, p0[2])),
        ];
        handState.active = true;
        overlay.setPointerCapture(e.pointerId);
    }
});

overlay.addEventListener("pointermove", (e) => {
    if (orbit.active && e.pointerId === orbit.pointerId) {
        const dx = e.clientX - orbit.lastX, dy = e.clientY - orbit.lastY;
        orbit.lastX = e.clientX; orbit.lastY = e.clientY;
        camera.theta -= dx * 0.01;
        camera.phi = Math.max(-(Math.PI / 2 - 0.01), Math.min(Math.PI / 2 - 0.01, camera.phi + dy * 0.01));
    }
    if (iact.active && e.pointerId === iact.pointerId) {
        // Recompute absolute 3D position from current mouse — never accumulate drift
        const cv = cameraVectors(camera);
        const np = screenToWorld(e.offsetX, e.offsetY, camera, cv);
        const vs = 8.0;
        handState.vel[0] = (np[0] - handState.pos[0]) * vs;
        handState.vel[1] = (np[1] - handState.pos[1]) * vs;
        handState.vel[2] = (np[2] - handState.pos[2]) * vs;
        handState.pos = np;   // no clamping — shader only iterates valid cells
    }
});

const endPointer = (e) => {
    if (e.pointerId === iact.pointerId) {
        iact.active = false; iact.pointerId = null;
        handState.active = false; handState.vel[0] = handState.vel[1] = handState.vel[2] = 0;
    }
    if (e.pointerId === orbit.pointerId) { orbit.active = false; orbit.pointerId = null; }
};
overlay.addEventListener("pointerup", endPointer);
overlay.addEventListener("pointercancel", endPointer);
overlay.addEventListener("wheel", (e) => {
    e.preventDefault();
    camera.radius = Math.max(1, Math.min(100000, camera.radius * Math.exp(e.deltaY * 0.001)));
}, { passive: false });

// ─────────────────────────────────────────────────────────────
//  4×4 matrix utilities (column-major, WebGPU depth [0,1])
// ─────────────────────────────────────────────────────────────
function mat4Perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0]);
}
function mat4LookAt(eye, center, up) {
    let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    let xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
        -(xx * eye[0] + xy * eye[1] + xz * eye[2]), -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
        -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
}
// Unproject screen (offsetX/Y) onto the plane through camera.target, normal = view direction
// sx, sy must be in CSS pixels (e.offsetX / e.offsetY)
function screenToWorld(sx, sy, cam, cv) {
    const ndcX = (sx / logicalW) * 2 - 1;
    const ndcY = 1 - (sy / logicalH) * 2;
    const aspect = logicalW / logicalH;
    const th = Math.tan(cam.fovy / 2);
    // camera front direction (eye → target, normalized)
    const fx = cam.target[0] - cv.eye[0], fy = cam.target[1] - cv.eye[1], fz = cam.target[2] - cv.eye[2];
    const fl = Math.hypot(fx, fy, fz) || 1;
    const fd = [fx / fl, fy / fl, fz / fl];
    // ray direction through NDC pixel
    let rdx = fd[0] + ndcX * th * aspect * cv.right[0] + ndcY * th * cv.up[0];
    let rdy = fd[1] + ndcX * th * aspect * cv.right[1] + ndcY * th * cv.up[1];
    let rdz = fd[2] + ndcX * th * aspect * cv.right[2] + ndcY * th * cv.up[2];
    const rl = Math.hypot(rdx, rdy, rdz) || 1;
    rdx /= rl; rdy /= rl; rdz /= rl;
    // intersect ray with plane: dot(P - target, fd) = 0
    const denom = rdx * fd[0] + rdy * fd[1] + rdz * fd[2];
    if (Math.abs(denom) < 1e-6) return [...cam.target];
    const ttx = cam.target[0] - cv.eye[0], tty = cam.target[1] - cv.eye[1], ttz = cam.target[2] - cv.eye[2];
    const t = (ttx * fd[0] + tty * fd[1] + ttz * fd[2]) / denom;
    return [cv.eye[0] + t * rdx, cv.eye[1] + t * rdy, cv.eye[2] + t * rdz];
}
function worldToScreen(pos, viewProj, w, h) {
    const x = pos[0], y = pos[1], z = pos[2];
    const cx = viewProj[0] * x + viewProj[4] * y + viewProj[8] * z + viewProj[12];
    const cy = viewProj[1] * x + viewProj[5] * y + viewProj[9] * z + viewProj[13];
    const cw = viewProj[3] * x + viewProj[7] * y + viewProj[11] * z + viewProj[15];
    if (cw <= 0) return null;
    return [(cx / cw + 1) / 2 * w, (1 - cy / cw) / 2 * h];
}
// Returns [minX, minY, maxX, maxY] of the simulation box in screen space,
// or null if any corner is behind the camera.
function simBoundsOnScreen(gridDims, viewProj, w, h) {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < 8; i++) {
        const sc = worldToScreen(
            [(i & 1) ? gridDims[0] : 0, (i & 2) ? gridDims[1] : 0, (i & 4) ? gridDims[2] : 0],
            viewProj, w, h
        );
        if (!sc) return null;
        if (sc[0] < minX) minX = sc[0]; if (sc[0] > maxX) maxX = sc[0];
        if (sc[1] < minY) minY = sc[1]; if (sc[1] > maxY) maxY = sc[1];
    }
    return [minX, minY, maxX, maxY];
}
function mat4Multiply(a, b) {
    const out = new Float32Array(16);
    for (let col = 0; col < 4; col++) {
        const b0 = b[col * 4], b1 = b[col * 4 + 1], b2 = b[col * 4 + 2], b3 = b[col * 4 + 3];
        out[col * 4] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
        out[col * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
        out[col * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
        out[col * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
    }
    return out;
}
function cameraVectors(cam) {
    const cp = Math.cos(cam.phi), sp = Math.sin(cam.phi), ct = Math.cos(cam.theta), st = Math.sin(cam.theta);
    const dir = [cp * st, sp, cp * ct];
    const eye = [cam.target[0] + cam.radius * dir[0], cam.target[1] + cam.radius * dir[1], cam.target[2] + cam.radius * dir[2]];
    const fx = -dir[0], fy = -dir[1], fz = -dir[2], wup = [0, 1, 0];
    let rx = fy * wup[2] - fz * wup[1], ry = fz * wup[0] - fx * wup[2], rz = fx * wup[1] - fy * wup[0];
    const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    return { eye, right: [rx, ry, rz], up: [ux, uy, uz] };
}

// ─────────────────────────────────────────────────────────────
//  WebGPU renderer (billboard instanced particles)
// ─────────────────────────────────────────────────────────────
async function initRenderer(device, particlePosBuffer, particleVelBuffer) {
    const context = c.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    // uniform: viewProj(64) + camRight+halfSize(16) + camUp+_pad(16) = 96 bytes
    const uniformBuffer = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const uni = new Float32Array(24);

    const depthTexture = device.createTexture({
        size: [c.width, c.height], format: "depth32float", usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    const shader = device.createShaderModule({
        code: /* wgsl */`
        struct Uniforms {
            viewProj : mat4x4<f32>,
            camRight : vec3<f32>, halfSize : f32,
            camUp    : vec3<f32>, _pad0    : f32,
        };
        @group(0) @binding(0) var<uniform> u : Uniforms;

        struct VsOut {
            @builtin(position) pos   : vec4<f32>,
            @location(0)       speed : f32,
        };

        @vertex
        fn vs(@location(0) pos: vec3<f32>, @location(1) vel: vec3<f32>,
              @builtin(vertex_index) vi: u32) -> VsOut {
            // lower 2 bits encode the corner: works with local (0-3) or global
            // (instance*4+local) vertex_index, so buggy mobile drivers are fine
            let cx = f32(vi & 1u) * 2.0 - 1.0;
            let cy = f32((vi >> 1u) & 1u) * 2.0 - 1.0;
            let world = pos + (cx*u.camRight + cy*u.camUp) * u.halfSize;
            var out: VsOut;
            out.pos   = u.viewProj * vec4<f32>(world, 1.0);
            out.speed = length(vel);
            return out;
        }

        // blue(slow) -> cyan -> green -> yellow -> red(fast)
        fn speedColor(speed: f32) -> vec3<f32> {
            let t = clamp(speed / 8.0, 0.0, 1.0);
            if (t < 0.25) {
                return mix(vec3<f32>(0.1, 0.2, 0.9), vec3<f32>(0.0, 0.9, 0.9), t * 4.0);
            } else if (t < 0.5) {
                return mix(vec3<f32>(0.0, 0.9, 0.9), vec3<f32>(0.1, 0.9, 0.1), (t-0.25) * 4.0);
            } else if (t < 0.75) {
                return mix(vec3<f32>(0.1, 0.9, 0.1), vec3<f32>(1.0, 0.85, 0.0), (t-0.5) * 4.0);
            }
            return mix(vec3<f32>(1.0, 0.85, 0.0), vec3<f32>(1.0, 0.1, 0.0), (t-0.75) * 4.0);
        }

        @fragment
        fn fs(in: VsOut) -> @location(0) vec4<f32> {
            return vec4<f32>(speedColor(in.speed), 1.0);
        }
    ` });

    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shader, entryPoint: "vs",
            buffers: [
                {
                    arrayStride: 16, stepMode: "instance",
                    attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }]
                },
                {
                    arrayStride: 16, stepMode: "instance",
                    attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }]
                },
            ],
        },
        fragment: { module: shader, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-strip" },
        depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    const particleGridRadius = 0.25;
    let depthTex = depthTexture, depthView = depthTexture.createView();
    let depthW = depthTexture.width, depthH = depthTexture.height;

    return function render(cmd, count, cv, viewProj) {
        // Always match depth texture to the actual swap-chain texture size.
        // Some mobile WebGPU implementations return a texture whose size differs
        // from canvas.width/height, so we track the actual size instead of c.width/c.height.
        const colorTex = context.getCurrentTexture();
        const tw = colorTex.width, th = colorTex.height;
        if (tw !== depthW || th !== depthH) {
            depthTex.destroy();
            depthTex = device.createTexture({
                size: [tw, th], format: "depth32float",
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
            depthView = depthTex.createView();
            depthW = tw; depthH = th;
        }
        uni.set(viewProj, 0);
        uni[16] = cv.right[0]; uni[17] = cv.right[1]; uni[18] = cv.right[2]; uni[19] = particleGridRadius;
        uni[20] = cv.up[0]; uni[21] = cv.up[1]; uni[22] = cv.up[2];
        device.queue.writeBuffer(uniformBuffer, 0, uni);

        const pass = cmd.beginRenderPass({
            colorAttachments: [{
                view: colorTex.createView(),
                clearValue: { r: 0.08, g: 0.08, b: 0.08, a: 1 }, loadOp: "clear", storeOp: "store",
            }],
            depthStencilAttachment: {
                view: depthView, depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store",
            },
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.setVertexBuffer(0, particlePosBuffer);
        pass.setVertexBuffer(1, particleVelBuffer);
        pass.draw(4, count);
        pass.end();
    };
}

// ─────────────────────────────────────────────────────────────
//  Main loop & entry point
// ─────────────────────────────────────────────────────────────
let avgFrameMs = 0, lastFrameTime = performance.now();
let _animFrameId = null;

function startLoop(encodeRender, fluidGPU) {
    function loop() {
        const aspect = c.width / c.height;
        const frameStart = performance.now();

        fluidGPU.updateHand(handState.pos, handState.vel,
            handState.radius, handState.strength, handState.active);
        handState.vel[0] = handState.vel[1] = handState.vel[2] = 0;

        // compute camera once — reused by both render and hand indicator
        const cv = cameraVectors(camera);
        const viewProj = mat4Multiply(
            mat4Perspective(camera.fovy, aspect, camera.near, camera.far),
            mat4LookAt(cv.eye, camera.target, [0, 1, 0])
        );

        // single submit: compute passes + render pass
        const cmd = fluidGPU.device.createCommandEncoder();
        fluidGPU.simFrame(cmd);
        encodeRender(cmd, fluidGPU.active_particle_num, cv, viewProj);
        fluidGPU.device.queue.submit([cmd.finish()]);

        const frameMs = frameStart - lastFrameTime;
        lastFrameTime = frameStart;
        avgFrameMs = avgFrameMs * 0.9 + frameMs * 0.1;

        octx.clearRect(0, 0, logicalW, logicalH);

        if (handState.active) {
            const sc = worldToScreen(handState.pos, viewProj, logicalW, logicalH);
            if (sc) {
                const ddx = handState.pos[0] - cv.eye[0];
                const ddy = handState.pos[1] - cv.eye[1];
                const ddz = handState.pos[2] - cv.eye[2];
                const dist = Math.hypot(ddx, ddy, ddz) || 1;
                const focal = 1 / Math.tan(camera.fovy / 2);
                const sr = Math.max(4, handState.radius / dist * focal * (logicalH / 2));
                octx.beginPath();
                octx.arc(sc[0], sc[1], sr, 0, Math.PI * 2);
                octx.strokeStyle = "rgba(255,255,255,0.7)";
                octx.lineWidth = 1.5;
                octx.stroke();
                octx.beginPath();
                octx.moveTo(sc[0] - 8, sc[1]); octx.lineTo(sc[0] + 8, sc[1]);
                octx.moveTo(sc[0], sc[1] - 8); octx.lineTo(sc[0], sc[1] + 8);
                octx.strokeStyle = "rgba(255,255,255,0.5)";
                octx.lineWidth = 1;
                octx.stroke();
            }
        }

        octx.font = "14px monospace";
        octx.fillStyle = "#aaa";
        octx.fillText(`frame: ${avgFrameMs.toFixed(2)} ms`, 16, 28);
        octx.fillText(`particles: ${fluidGPU.active_particle_num}`, 16, 46);
        octx.fillText(isMobile
            ? `drag in sim: push  drag outside: orbit`
            : `L-drag: push fluid  R-drag: orbit  wheel: zoom`, 16, 64);
        _animFrameId = requestAnimationFrame(loop);
    }
    loop();
}

function showError(err) {
    console.error(err);
    octx.clearRect(0, 0, logicalW, logicalH);
    octx.fillStyle = "#b00";
    octx.fillText("WebGPU error: " + err.message, 16, 28);
}

async function init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter.");
    const device = await adapter.requestDevice();

    device.lost.then(async (info) => {
        if (_animFrameId !== null) { cancelAnimationFrame(_animFrameId); _animFrameId = null; }
        if (info.reason !== "destroyed") {
            await new Promise(r => setTimeout(r, 1000));
            init().catch(showError);
        }
    });

    const fluidGPU = new FluidGPU(device, fluid);
    const renderFn = await initRenderer(device, fluidGPU.particlePosBuffer, fluidGPU.particleVelBuffer);
    startLoop(renderFn, fluidGPU);
}

init().catch(showError);