import { WG_SIZE, urlNum } from './config.js';
import {
    WGSL_CLEAR, WGSL_COMPUTE_WEIGHTS, WGSL_P2G_MASS, WGSL_P2G_MOM,
    WGSL_DECODE_MASS, WGSL_APPLY_HAND, WGSL_UPDATE_GRID, WGSL_G2P,
    WGSL_COMPACT, WGSL_RAYCAST,
} from './shaders.js';
import { lineBoxIntersect } from './math.js';

// ?simstage=N — gate simFrame()'s passes at stage N (1=CLEAR .. 8=diffuse.step),
// skipping everything after. Used to bisect mobile Vulkan driver crashes
// (VK_ERROR_OUT_OF_HOST_MEMORY triage — see CLAUDE.md). Default 99 = unchanged.
const SIM_STAGE = urlNum('simstage', 99);

export class FluidGPU {
    constructor(aspectX, aspectY, aspectZ, particleRadius, particleNum) {
        const normToGrid = 1 / (particleRadius * 4);
        this.grid_X_num = Math.ceil(aspectX * normToGrid);
        this.grid_Y_num = Math.ceil(aspectY * normToGrid);
        this.grid_Z_num = Math.ceil(aspectZ * normToGrid);
        this.grid_num   = this.grid_X_num * this.grid_Y_num * this.grid_Z_num;
        this.particle_num = particleNum;
        this.active_particle_num = 0;

        this.particle_pos    = new Float32Array(particleNum * 3);
        this.particle_vel    = new Float32Array(particleNum * 3);
        this.particle_affine = new Float32Array(particleNum * 9);
        this.particle_mass   = new Float32Array(particleNum);

        this.DT           = 0.3;
        this.REST_DENSITY = 8.0;
        this.STIFFNESS    = 100.0;
        this.EOS_POWER    = 1;
        this.VISCOSITY    = 0.01;
        this.GRAVITY      = -0.98;
        this.SUBSTEPS     = 1;

        // Splash方式の予測位置バネ補正 (ref/boundary-condition-splash-style.md ステップ3)。
        // 値はSplashの初期値をそのまま踏襲。
        this.WALL_MIN       = 3.0;
        this.WALL_STIFFNESS = 1.0;
        this.LOOKAHEAD_K    = 2.0;

        const HM = 2;
        this.HARD_MIN   = HM;
        this.HARD_MAX_X = this.grid_X_num - HM;
        this.HARD_MAX_Y = this.grid_Y_num - HM;
        this.HARD_MAX_Z = this.grid_Z_num - HM;

        // Optional DiffuseGPU instance (spray/foam/bubble tracer particles). When set,
        // simFrame() dispatches its generate+advect passes once per substep, right
        // after G2P — see diffuse-gpu.js.
        this.diffuse = null;
    }

    // x0..z1: normalized 0–1 per axis. Particles placed every 0.5 grid cells.
    // Multiple calls stack from the current active_particle_num.
    fillBlock(x0, y0, z0, x1, y1, z1) {
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

    // Real-time spawn: adds up to `count` particles in a jittered sphere around
    // (cx,cy,cz), clamped to the hard sim bounds so G2P's unchecked grid-index
    // math never goes negative. Clamped to remaining buffer capacity. Returns
    // the number actually spawned. Call only after initGPU().
    //
    // Particles get an outward radial kick (sprayVel) from the sphere center —
    // without it, held-down spawning keeps depositing new particles into the
    // same static volume faster than the fluid can relax, so local density
    // blows past rest_density and the EOS pressure term launches particles out
    // at high speed (visible as particles "bouncing" off the domain walls).
    // The radial kick disperses them like a hose nozzle instead of a static fill.
    //
    // Refuses (returns 0) while a deleteNear() compaction is still resolving —
    // its async readback overwrites active_particle_num with a value computed
    // before this spawn, which would otherwise silently drop these particles.
    spawnParticles(cx, cy, cz, count, radius, vel = [0, 0, 0], sprayVel = 3.0) {
        if (this._deletePending) return 0;
        const start = this.active_particle_num;
        const n = Math.min(count, this.particle_num - start);
        if (n <= 0) return 0;

        // Reused scratch arrays (sized to particle_num in _createBuffers, so any valid
        // n always fits) instead of a fresh allocation every call — held-down spawning
        // calls this every frame.
        const pos4 = this._spawnPos4, vel4 = this._spawnVel4;
        const hMin = this.HARD_MIN, hMaxX = this.HARD_MAX_X, hMaxY = this.HARD_MAX_Y, hMaxZ = this.HARD_MAX_Z;
        for (let i = 0; i < n; i++) {
            const r = radius * Math.cbrt(Math.random());
            const theta = Math.random() * Math.PI * 2;
            const cphi  = Math.random() * 2 - 1;
            const sphi  = Math.sqrt(1 - cphi * cphi);
            const dx = r * sphi * Math.cos(theta);
            const dy = r * sphi * Math.sin(theta);
            const dz = r * cphi;
            pos4[i * 4]     = Math.min(hMaxX, Math.max(hMin, cx + dx));
            pos4[i * 4 + 1] = Math.min(hMaxY, Math.max(hMin, cy + dy));
            pos4[i * 4 + 2] = Math.min(hMaxZ, Math.max(hMin, cz + dz));
            const rl = Math.hypot(dx, dy, dz) || 1;
            vel4[i * 4]     = vel[0] + dx / rl * sprayVel;
            vel4[i * 4 + 1] = vel[1] + dy / rl * sprayVel;
            vel4[i * 4 + 2] = vel[2] + dz / rl * sprayVel;
        }
        // mass is always 1.0 and _spawnMass is filled once at creation, so only pos/vel
        // need to be recomputed per call. Write via .buffer so the offset/size below
        // are unambiguously bytes (not TypedArray element counts).
        this.device.queue.writeBuffer(this.particlePosBuffer, start * 16, pos4.buffer, 0, n * 16);
        this.device.queue.writeBuffer(this.particleVelBuffer, start * 16, vel4.buffer, 0, n * 16);
        this.device.queue.writeBuffer(this.particleMassBuffer, start * 4, this._spawnMass.buffer, 0, n * 4);
        this.active_particle_num = start + n;
        return n;
    }

    // Call once after fillBlock(), before simFrame().
    initGPU(device) {
        this.device = device;
        this._createBuffers();
        this._createPipelines();
    }

    _pack3to4(arr3) {
        const n = arr3.length / 3;
        const arr4 = new Float32Array(n * 4);
        for (let i = 0; i < n; i++) {
            arr4[i * 4]     = arr3[i * 3];
            arr4[i * 4 + 1] = arr3[i * 3 + 1];
            arr4[i * 4 + 2] = arr3[i * 3 + 2];
        }
        return arr4;
    }

    _createBuffers() {
        const d = this.device;
        const N = this.particle_num;
        const G = this.grid_num;
        const SV = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST;
        const CS = GPUBufferUsage.COPY_SRC, VX = GPUBufferUsage.VERTEX;
        const UN = GPUBufferUsage.UNIFORM;

        this.particlePosBuffer    = d.createBuffer({ size: N * 16,    usage: SV | CD | CS | VX });
        this.particleVelBuffer    = d.createBuffer({ size: N * 16,    usage: SV | CD | CS | VX });
        this.particleAffineBuffer = d.createBuffer({ size: N * 9 * 4, usage: SV | CD | CS });
        this.particleMassBuffer    = d.createBuffer({ size: N * 4,     usage: SV | CD });
        this.particleDensityBuffer = d.createBuffer({ size: N * 4,     usage: SV });
        // BaseWeights: bx,by,bz(i32) + fx,fy,fz + wx0..wz2(9×f32) = 15×4 = 60 bytes/particle.
        // Computed once per substep (WGSL_COMPUTE_WEIGHTS) and shared by P2G_MASS/P2G_MOM/G2P.
        this.particleWeightsBuffer = d.createBuffer({ size: N * 60,    usage: SV });
        this.cellMvBuffer         = d.createBuffer({ size: G * 3 * 4, usage: SV | CD | CS });
        this.cellMassBuffer       = d.createBuffer({ size: G * 4,     usage: SV | CD | CS });
        this.cellMvF32Buffer      = d.createBuffer({ size: G * 3 * 4, usage: SV });
        this.cellMassF32Buffer    = d.createBuffer({ size: G * 4,     usage: SV });
        this.paramsBuffer         = d.createBuffer({ size: 80,        usage: UN | CD });
        // 80 bytes: the original 12 fields (48B) + bbox_min/bbox_max (6×i32) + 2×pad
        // (32B) — see updateHand()/WGSL_APPLY_HAND for why the bbox exists.
        this.handParamsBuffer     = d.createBuffer({ size: 80,        usage: UN | CD });

        // Delete (stream-compaction) scratch buffers — see spawnParticles()/deleteNear().
        // Affine is intentionally not double-buffered here (see WGSL_COMPACT comment).
        this.particlePosScratch  = d.createBuffer({ size: N * 16, usage: SV | CD | CS });
        this.particleVelScratch  = d.createBuffer({ size: N * 16, usage: SV | CS });
        this.particleMassScratch = d.createBuffer({ size: N * 4,  usage: SV | CS });
        this.compactCounterBuffer     = d.createBuffer({ size: 4, usage: SV | CD | CS });
        this.compactCounterReadBuffer = d.createBuffer({ size: 4, usage: CD | GPUBufferUsage.MAP_READ });
        // RayParams layout (both delete + raycast): ox,oy,oz,radius,dx,dy,dz,_pad — 32 bytes.
        this.deleteParamsBuffer = d.createBuffer({ size: 32, usage: UN | CD });
        this.rayParamsBuffer    = d.createBuffer({ size: 32, usage: UN | CD });
        this.rayHitBuffer       = d.createBuffer({ size: 4, usage: SV | CD | CS });
        this.rayHitReadBuffer   = d.createBuffer({ size: 4, usage: CD | GPUBufferUsage.MAP_READ });

        // Pre-allocated CPU buffers reused every frame to avoid GC pressure.
        this._paramsBuf = new ArrayBuffer(80);
        this._paramsU32 = new Uint32Array(this._paramsBuf);
        this._paramsF32 = new Float32Array(this._paramsBuf);
        // Sentinel-filled so the very first _writeParamsBuffer() call always uploads
        // (see _writeParamsBuffer's dedup check).
        this._paramsPrevU32 = new Uint32Array(20).fill(0xFFFFFFFF);
        this._handBuf   = new ArrayBuffer(80);
        this._handF32   = new Float32Array(this._handBuf);
        this._handU32   = new Uint32Array(this._handBuf);
        this._handI32   = new Int32Array(this._handBuf);
        this.handActive = false;
        this._handBBoxDimX = 0; this._handBBoxDimY = 0; this._handBBoxDimZ = 0;
        this._deleteBuf = new ArrayBuffer(32);
        this._deleteF32 = new Float32Array(this._deleteBuf);
        this._deletePending = false;
        this._rayBuf = new ArrayBuffer(32);
        this._rayF32 = new Float32Array(this._rayBuf);
        this._rayPending = false;
        this.lastRayHit = null;

        // Reused scratch arrays for spawnParticles() — sized to the full particle
        // capacity so any valid n never needs a fallback allocation. Mass is always
        // 1.0, so _spawnMass is filled once here rather than every call.
        this._spawnPos4 = new Float32Array(N * 4);
        this._spawnVel4 = new Float32Array(N * 4);
        this._spawnMass = new Float32Array(N).fill(1.0);

        d.queue.writeBuffer(this.particlePosBuffer,    0, this._pack3to4(this.particle_pos));
        d.queue.writeBuffer(this.particleVelBuffer,    0, this._pack3to4(this.particle_vel));
        d.queue.writeBuffer(this.particleAffineBuffer, 0, this.particle_affine);
        d.queue.writeBuffer(this.particleMassBuffer,   0, this.particle_mass);

        // particlePosScratch's never-written tail must hold a position inside the hard
        // bounds (not the buffer's zero default) so a stale post-compaction read during
        // the 1-frame async count readback can't send G2P's unchecked grid index negative.
        const centerPos = new Float32Array(N * 4);
        for (let i = 0; i < N; i++) {
            centerPos[i * 4]     = this.grid_X_num / 2;
            centerPos[i * 4 + 1] = this.grid_Y_num / 2;
            centerPos[i * 4 + 2] = this.grid_Z_num / 2;
        }
        d.queue.writeBuffer(this.particlePosScratch, 0, centerPos);
    }

    _writeParamsBuffer(dt) {
        // Layout must match the Params struct (u32 fields first, then f32 fields).
        this._paramsU32[0] = this.grid_X_num;
        this._paramsU32[1] = this.grid_Y_num;
        this._paramsU32[2] = this.grid_Z_num;
        this._paramsU32[3] = this.active_particle_num;
        this._paramsF32[4] = dt;
        this._paramsF32[5] = this.GRAVITY;
        this._paramsF32[6] = this.REST_DENSITY;
        this._paramsF32[7] = this.STIFFNESS;
        this._paramsF32[8] = this.EOS_POWER;
        this._paramsF32[9] = this.VISCOSITY;
        this._paramsF32[10] = this.HARD_MIN;
        this._paramsF32[11] = this.HARD_MAX_X;
        this._paramsF32[12] = this.HARD_MAX_Y;
        this._paramsF32[13] = this.HARD_MAX_Z;
        this._paramsF32[14] = this.WALL_MIN;
        this._paramsF32[15] = this.WALL_STIFFNESS;
        this._paramsF32[16] = this.LOOKAHEAD_K;

        // deleteNear()/raycastFluid()/simFrame() can all call this within the same
        // frame with byte-identical contents — dt is always DT/SUBSTEPS and every
        // other field is static; only active_particle_num can legitimately differ
        // between calls (if a spawn happened in between). Comparing the u32 view
        // covers the f32 fields too (equal floats have equal bit patterns), so skip
        // the redundant GPU upload when nothing actually changed since last time.
        let same = true;
        for (let i = 0; i < 20; i++) {
            if (this._paramsU32[i] !== this._paramsPrevU32[i]) { same = false; break; }
        }
        if (same) return;
        this._paramsPrevU32.set(this._paramsU32);
        this.device.queue.writeBuffer(this.paramsBuffer, 0, this._paramsBuf);
    }

    _createPipelines() {
        const d = this.device;
        const mkP = (code) => d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code }), entryPoint: 'main' },
        });
        this._clearPipeline       = mkP(WGSL_CLEAR);
        this._computeWeightsPipeline = mkP(WGSL_COMPUTE_WEIGHTS);
        this._p2gMassPipeline     = mkP(WGSL_P2G_MASS);
        this._decodeMassPipeline  = mkP(WGSL_DECODE_MASS);
        this._p2gMomPipeline      = mkP(WGSL_P2G_MOM);
        this._updateGridPipeline  = mkP(WGSL_UPDATE_GRID);
        this._g2pPipeline         = mkP(WGSL_G2P);
        this._applyHandPipeline   = mkP(WGSL_APPLY_HAND);
        this._compactPipeline     = mkP(WGSL_COMPACT);
        this._raycastPipeline     = mkP(WGSL_RAYCAST);

        const mkBG = (pipeline, entries) =>
            d.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
        const buf = (binding, buffer) => ({ binding, resource: { buffer } });

        this._clearBG = mkBG(this._clearPipeline, [
            buf(0, this.cellMvBuffer), buf(1, this.cellMassBuffer), buf(2, this.paramsBuffer),
        ]);
        this._computeWeightsBG = mkBG(this._computeWeightsPipeline, [
            buf(0, this.particlePosBuffer), buf(1, this.particleWeightsBuffer), buf(2, this.paramsBuffer),
        ]);
        this._p2gMassBG = mkBG(this._p2gMassPipeline, [
            buf(0, this.particleMassBuffer), buf(1, this.cellMassBuffer),
            buf(2, this.paramsBuffer),       buf(3, this.particleWeightsBuffer),
        ]);
        this._decodeMassBG = mkBG(this._decodeMassPipeline, [
            buf(0, this.cellMassBuffer), buf(1, this.cellMassF32Buffer), buf(2, this.paramsBuffer),
        ]);
        this._p2gMomBG = mkBG(this._p2gMomPipeline, [
            buf(0, this.particleVelBuffer),    buf(1, this.particleAffineBuffer),
            buf(2, this.particleMassBuffer),   buf(3, this.cellMassF32Buffer),
            buf(4, this.cellMvBuffer),         buf(5, this.paramsBuffer),
            buf(6, this.particleDensityBuffer),buf(7, this.particleWeightsBuffer),
        ]);
        this._updateGridBG = mkBG(this._updateGridPipeline, [
            buf(0, this.cellMvBuffer), buf(1, this.cellMassBuffer), buf(2, this.paramsBuffer),
            buf(3, this.cellMvF32Buffer),
        ]);
        this._g2pBG = mkBG(this._g2pPipeline, [
            buf(0, this.particlePosBuffer),    buf(1, this.particleVelBuffer),
            buf(2, this.particleAffineBuffer), buf(3, this.cellMvF32Buffer),
            buf(4, this.paramsBuffer),         buf(5, this.particleWeightsBuffer),
        ]);
        this._applyHandBG = mkBG(this._applyHandPipeline, [
            buf(0, this.cellMvF32Buffer), buf(1, this.paramsBuffer), buf(2, this.handParamsBuffer),
        ]);
        this._compactBG = mkBG(this._compactPipeline, [
            buf(0, this.particlePosBuffer),    buf(1, this.particleVelBuffer),    buf(2, this.particleMassBuffer),
            buf(3, this.particlePosScratch),   buf(4, this.particleVelScratch),   buf(5, this.particleMassScratch),
            buf(6, this.compactCounterBuffer), buf(7, this.paramsBuffer),         buf(8, this.deleteParamsBuffer),
        ]);
        this._raycastBG = mkBG(this._raycastPipeline, [
            buf(0, this.particlePosBuffer), buf(1, this.rayHitBuffer),
            buf(2, this.paramsBuffer),      buf(3, this.rayParamsBuffer),
        ]);
    }

    updateHand(pos, vel, radius, strength, active, eye) {
        this.handActive = active;
        this._handF32[0] = pos[0]; this._handF32[1] = pos[1]; this._handF32[2] = pos[2];
        this._handF32[3] = radius;
        this._handF32[4] = vel[0]; this._handF32[5] = vel[1]; this._handF32[6] = vel[2];
        this._handU32[7] = active ? 1 : 0;
        this._handF32[8] = strength;
        this._handF32[9] = eye[0]; this._handF32[10] = eye[1]; this._handF32[11] = eye[2];

        // Bounding box (grid cells) of (push cylinder ∩ grid), so APPLY_HAND only
        // dispatches over this box instead of the whole grid. The cylinder is infinite
        // along the camera ray through `pos` (depth-independent push, same as
        // deleteNear/raycastFluid) — NOT just a small sphere around `pos` — so a naive
        // "pos ± radius" box would silently drop cells the push is supposed to reach
        // further along that ray. Correct approach: intersect the infinite line
        // (pos, ray_dir) with the grid box inflated by `radius` on every side (any
        // point within `radius` of the line that also lies in the grid must have its
        // nearest on-line point inside that inflated box — see lineBoxIntersect's
        // doc comment) to get the relevant t-range, then expand the resulting segment
        // endpoints by `radius` again for the cylinder's actual cross-section.
        this._handBBoxDimX = 0; this._handBBoxDimY = 0; this._handBBoxDimZ = 0;
        if (active) {
            const dx = pos[0] - eye[0], dy = pos[1] - eye[1], dz = pos[2] - eye[2];
            const dl = Math.hypot(dx, dy, dz) || 1;
            const dir = [dx / dl, dy / dl, dz / dl];
            const R = radius;
            const seg = lineBoxIntersect(pos, dir,
                [-R, -R, -R],
                [this.grid_X_num + R, this.grid_Y_num + R, this.grid_Z_num + R]);
            if (seg) {
                const [t0, t1] = seg;
                const p0x = pos[0] + t0 * dir[0], p0y = pos[1] + t0 * dir[1], p0z = pos[2] + t0 * dir[2];
                const p1x = pos[0] + t1 * dir[0], p1y = pos[1] + t1 * dir[1], p1z = pos[2] + t1 * dir[2];
                const bx0 = Math.max(0, Math.floor(Math.min(p0x, p1x) - R));
                const by0 = Math.max(0, Math.floor(Math.min(p0y, p1y) - R));
                const bz0 = Math.max(0, Math.floor(Math.min(p0z, p1z) - R));
                const bx1 = Math.min(this.grid_X_num - 1, Math.ceil(Math.max(p0x, p1x) + R));
                const by1 = Math.min(this.grid_Y_num - 1, Math.ceil(Math.max(p0y, p1y) + R));
                const bz1 = Math.min(this.grid_Z_num - 1, Math.ceil(Math.max(p0z, p1z) + R));
                this._handI32[12] = bx0; this._handI32[13] = by0; this._handI32[14] = bz0;
                this._handI32[15] = bx1; this._handI32[16] = by1; this._handI32[17] = bz1;
                this._handBBoxDimX = Math.max(0, bx1 - bx0 + 1);
                this._handBBoxDimY = Math.max(0, by1 - by0 + 1);
                this._handBBoxDimZ = Math.max(0, bz1 - bz0 + 1);
            }
        }

        this.device.queue.writeBuffer(this.handParamsBuffer, 0, this._handBuf);
    }

    // Removes active particles within `radius` of the infinite line through `origin`
    // along `dir` (same cylinder test as APPLY_HAND) via GPU stream compaction. This
    // is depth-independent — unlike testing against a single point at a fixed plane
    // depth, it affects the fluid wherever it actually sits along the view ray.
    // Records into `cmd`; the compacted survivors are copied back into the main
    // buffers in this SAME frame (right after the compact dispatch, before this
    // frame's simFrame() runs) so the data simFrame advances is always current —
    // deferring this copy to pollDelete() previously caused a visible "rewind" each
    // cycle, since survivors would keep getting overwritten by an increasingly stale
    // snapshot from several simFrame() steps ago. The exact new active_particle_num
    // only becomes known after an async GPU readback, so call pollDelete() once per
    // frame after queue.submit(); until then active_particle_num still reports the
    // pre-delete count, so the buffer's unwritten tail (see particlePosScratch init
    // in _createBuffers) can be visible as stale "ghost" particles until that
    // readback resolves — normally one frame, but possibly more under GPU load,
    // since deleteNear() self-throttles (see _deletePending check below) and won't
    // re-dispatch until the outstanding mapAsync completes. A minor, self-correcting
    // cosmetic artifact, accepted as the lesser problem.
    // Skips (returns false) while a previous readback is still in flight to avoid
    // overlapping mapAsync calls on the same buffer.
    deleteNear(cmd, origin, dir, radius) {
        if (this._deletePending) return false;
        this._writeParamsBuffer(this.DT / this.SUBSTEPS);
        this._deleteF32[0] = origin[0]; this._deleteF32[1] = origin[1]; this._deleteF32[2] = origin[2];
        this._deleteF32[3] = radius;
        this._deleteF32[4] = dir[0];    this._deleteF32[5] = dir[1];    this._deleteF32[6] = dir[2];
        this.device.queue.writeBuffer(this.deleteParamsBuffer, 0, this._deleteBuf);
        this.device.queue.writeBuffer(this.compactCounterBuffer, 0, new Uint32Array([0]));

        const wgP = Math.ceil(this.active_particle_num / WG_SIZE);
        const pass = cmd.beginComputePass();
        pass.setPipeline(this._compactPipeline);
        pass.setBindGroup(0, this._compactBG);
        pass.dispatchWorkgroups(wgP);
        pass.end();

        const oldCount = this.active_particle_num;
        cmd.copyBufferToBuffer(this.particlePosScratch,  0, this.particlePosBuffer,  0, oldCount * 16);
        cmd.copyBufferToBuffer(this.particleVelScratch,  0, this.particleVelBuffer,  0, oldCount * 16);
        cmd.copyBufferToBuffer(this.particleMassScratch, 0, this.particleMassBuffer, 0, oldCount * 4);
        cmd.copyBufferToBuffer(this.compactCounterBuffer, 0, this.compactCounterReadBuffer, 0, 4);

        this._deletePending = true;
        return true;
    }

    // Call once per frame after queue.submit(), whether or not deleteNear() ran.
    pollDelete() {
        if (!this._deletePending || this._deleteMapping) return;
        this._deleteMapping = true;
        this.compactCounterReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const newCount = new Uint32Array(this.compactCounterReadBuffer.getMappedRange())[0];
            this.compactCounterReadBuffer.unmap();
            this.active_particle_num = newCount;
            this._deletePending = false;
            this._deleteMapping = false;
        });
    }

    // Finds where the view ray (origin, dir) first meets the existing fluid, within
    // `radius` of the ray axis (same cylinder test as deleteNear/APPLY_HAND). Records
    // into `cmd`; result lands in this.lastRayHit ([x,y,z] world pos, or null if the
    // ray doesn't currently pass near any particle) after an async readback — call
    // pollRaycast() once per frame after queue.submit(). Used to aim spawnParticles()
    // at the actual fluid surface instead of a fixed-depth plane.
    raycastFluid(cmd, origin, dir, radius) {
        if (this._rayPending) return false;
        this._writeParamsBuffer(this.DT / this.SUBSTEPS);
        this._rayF32[0] = origin[0]; this._rayF32[1] = origin[1]; this._rayF32[2] = origin[2];
        this._rayF32[3] = radius;
        this._rayF32[4] = dir[0];    this._rayF32[5] = dir[1];    this._rayF32[6] = dir[2];
        this.device.queue.writeBuffer(this.rayParamsBuffer, 0, this._rayBuf);
        this.device.queue.writeBuffer(this.rayHitBuffer, 0, new Uint32Array([0xFFFFFFFF]));

        const wgP = Math.ceil(this.active_particle_num / WG_SIZE);
        const pass = cmd.beginComputePass();
        pass.setPipeline(this._raycastPipeline);
        pass.setBindGroup(0, this._raycastBG);
        pass.dispatchWorkgroups(wgP);
        pass.end();

        cmd.copyBufferToBuffer(this.rayHitBuffer, 0, this.rayHitReadBuffer, 0, 4);

        this._rayOrigin = origin;
        this._rayDir = dir;
        this._rayPending = true;
        return true;
    }

    // Call once per frame after queue.submit(), whether or not raycastFluid() ran.
    pollRaycast() {
        if (!this._rayPending || this._rayMapping) return;
        this._rayMapping = true;
        this.rayHitReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const bits = new Uint32Array(this.rayHitReadBuffer.getMappedRange())[0];
            this.rayHitReadBuffer.unmap();
            if (bits === 0xFFFFFFFF) {
                this.lastRayHit = null;
            } else {
                const t = new Float32Array(new Uint32Array([bits]).buffer)[0];
                this.lastRayHit = [
                    this._rayOrigin[0] + t * this._rayDir[0],
                    this._rayOrigin[1] + t * this._rayDir[1],
                    this._rayOrigin[2] + t * this._rayDir[2],
                ];
            }
            this._rayPending = false;
            this._rayMapping = false;
        });
    }

    simFrame(cmd, tsQuery = null) {
        const sub_dt = this.DT / this.SUBSTEPS;
        const wgP = Math.ceil(this.active_particle_num / WG_SIZE);
        const wgG = Math.ceil(this.grid_num / WG_SIZE);
        this._writeParamsBuffer(sub_dt);

        // `wg` is either a workgroup count (1D) or a [x,y,z] array (3D, used by
        // APPLY_HAND's bbox-sized dispatch — see updateHand()).
        const run = (pip, bg, wg, beginTs, endTs) => {
            const desc = {};
            if (tsQuery !== null && (beginTs !== undefined || endTs !== undefined)) {
                desc.timestampWrites = { querySet: tsQuery.querySet };
                if (beginTs !== undefined) desc.timestampWrites.beginningOfPassWriteIndex = beginTs;
                if (endTs   !== undefined) desc.timestampWrites.endOfPassWriteIndex       = endTs;
            }
            const pass = cmd.beginComputePass(desc);
            pass.setPipeline(pip); pass.setBindGroup(0, bg);
            if (Array.isArray(wg)) pass.dispatchWorkgroups(wg[0], wg[1], wg[2]);
            else pass.dispatchWorkgroups(wg);
            pass.end();
        };
        for (let step = 0; step < this.SUBSTEPS; step++) {
            const first = step === 0, last = step === this.SUBSTEPS - 1;
            if (SIM_STAGE >= 1) run(this._clearPipeline,      this._clearBG,       wgG, first ? tsQuery?.beginIndex : undefined);
            if (SIM_STAGE >= 2) run(this._computeWeightsPipeline, this._computeWeightsBG, wgP);
            if (SIM_STAGE >= 3) run(this._p2gMassPipeline,   this._p2gMassBG,    wgP);
            if (SIM_STAGE >= 4) run(this._decodeMassPipeline, this._decodeMassBG, wgG);
            if (SIM_STAGE >= 5) run(this._p2gMomPipeline,     this._p2gMomBG,     wgP);
            if (SIM_STAGE >= 6) run(this._updateGridPipeline, this._updateGridBG, wgG);
            if (SIM_STAGE >= 6 && this.handActive) {
                // Dispatched only over the (push cylinder ∩ grid) bbox computed in
                // updateHand() — 4×4×4 matches WGSL_APPLY_HAND's workgroup_size.
                const wgHX = Math.ceil(this._handBBoxDimX / 4);
                const wgHY = Math.ceil(this._handBBoxDimY / 4);
                const wgHZ = Math.ceil(this._handBBoxDimZ / 4);
                if (wgHX > 0 && wgHY > 0 && wgHZ > 0) {
                    run(this._applyHandPipeline, this._applyHandBG, [wgHX, wgHY, wgHZ]);
                }
            }
            if (SIM_STAGE >= 7) run(this._g2pPipeline,        this._g2pBG,        wgP, undefined, last ? tsQuery?.endIndex : undefined);
            if (SIM_STAGE >= 8 && this.diffuse) this.diffuse.step(cmd, this.active_particle_num);
        }
    }
}
