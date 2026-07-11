import { WG_SIZE } from './config.js';
import { WGSL_DIFFUSE_GENERATE, WGSL_DIFFUSE_ADVECT } from './shaders.js';

// DiffuseParticle layout: pos(vec4) + vel(vec4) + ptype(u32) + alive(u32) + lifetime(f32) + _pad(f32)
const PARTICLE_BYTES = 48;

// GPU port of the CPU `DiffuseParticles` class (see
// ref/diffuse-particles-gpu-migration.md). Spray/foam/bubble tracer particles that
// sample the main fluid's grid but never feed back into it (one-way coupling).
//
// Unlike the CPU version's `count` + swap-remove array, the GPU pool is a fixed-size
// array of `maxCount` slots plus a free-list (stack of free slot indices + an atomic
// top). Spawn = pop a slot, kill = push a slot back — both O(1) and race-free across
// threads, unlike swap-remove which would need serialized compaction.
export class DiffuseGPU {
    constructor(maxCount) {
        this.maxCount = maxCount;

        // Tunable parameters — defaults mirror the CPU DiffuseParticles class.
        this.KE_THRESHOLD = 0.5;
        this.PRESSURE_THRESHOLD = 1.0;
        this.CREST_DOT_THRESHOLD = 0.3;
        this.SPAWN_RATE_K = 3.0;
        this.LIFETIME_FOAM = 3.0;
        this.BUOYANCY = 1.2;
        this.DRAG = 4.0;
        this.DENSITY_SPRAY_MAX = 1.5;
        this.DENSITY_BUBBLE_MIN = 6.0;

        this.aliveCount = 0;
        this._seed = 1;
        this._debugPending = false;
        this._debugMapping = false;
    }

    // Call once after fluid.initGPU(device) — binds directly to the main fluid's
    // existing buffers (position/velocity/mass/density/cell arrays), so `fluid`
    // must already have created them.
    initGPU(device, fluid) {
        this.device = device;
        this.fluid = fluid;
        this._createBuffers();
        this._createPipelines();
    }

    _createBuffers() {
        const d = this.device;
        const N = this.maxCount;
        const SV = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, VX = GPUBufferUsage.VERTEX;
        const CS = GPUBufferUsage.COPY_SRC, UN = GPUBufferUsage.UNIFORM;

        this.particleBuffer     = d.createBuffer({ size: N * PARTICLE_BYTES, usage: SV | CD | VX });
        this.freeListBuffer     = d.createBuffer({ size: N * 4, usage: SV | CD });
        // COPY_SRC: freeListTop is copied out for the debug alive-count readback (see pollDebug()).
        this.freeListTopBuffer  = d.createBuffer({ size: 4, usage: SV | CD | CS });
        this.genParamsBuffer    = d.createBuffer({ size: 32, usage: UN | CD });
        this.advectParamsBuffer = d.createBuffer({ size: 16, usage: UN | CD });
        this._debugReadBuffer   = d.createBuffer({ size: 4, usage: CD | GPUBufferUsage.MAP_READ });

        // Every slot starts "free": freeList = [0, 1, ..., N-1], top = N.
        const initFreeList = new Uint32Array(N);
        for (let i = 0; i < N; i++) initFreeList[i] = i;
        d.queue.writeBuffer(this.freeListBuffer, 0, initFreeList);
        d.queue.writeBuffer(this.freeListTopBuffer, 0, new Int32Array([N]));

        // alive=0 (dead) for every slot; zero-filling the whole struct is enough since
        // u32 0 is a valid "dead" bit pattern for both ptype and alive.
        d.queue.writeBuffer(this.particleBuffer, 0, new Float32Array(N * (PARTICLE_BYTES / 4)));

        this._advectBuf = new ArrayBuffer(16);
        this._advectU32 = new Uint32Array(this._advectBuf);
        this._advectF32 = new Float32Array(this._advectBuf);
        this._advectU32[0] = N;
        this._advectF32[1] = this.BUOYANCY;
        this._advectF32[2] = this.DRAG;
        d.queue.writeBuffer(this.advectParamsBuffer, 0, this._advectBuf);

        this._genBuf = new ArrayBuffer(32);
        this._genF32 = new Float32Array(this._genBuf);
        this._genU32 = new Uint32Array(this._genBuf);
        this._writeGenParams();
    }

    _writeGenParams() {
        this._genF32[0] = this.KE_THRESHOLD;
        this._genF32[1] = this.SPAWN_RATE_K;
        this._genF32[2] = this.DENSITY_SPRAY_MAX;
        this._genF32[3] = this.DENSITY_BUBBLE_MIN;
        this._genF32[4] = this.LIFETIME_FOAM;
        this._genF32[5] = this.PRESSURE_THRESHOLD;
        this._genF32[6] = this.CREST_DOT_THRESHOLD;
        this._genU32[7] = this._seed;
        this.device.queue.writeBuffer(this.genParamsBuffer, 0, this._genBuf);
    }

    _createPipelines() {
        const d = this.device;
        const mkP = (code) => d.createComputePipeline({
            layout: 'auto',
            compute: { module: d.createShaderModule({ code }), entryPoint: 'main' },
        });
        this._generatePipeline = mkP(WGSL_DIFFUSE_GENERATE);
        this._advectPipeline   = mkP(WGSL_DIFFUSE_ADVECT);

        const buf = (binding, buffer) => ({ binding, resource: { buffer } });
        const f = this.fluid;

        this._generateBG = d.createBindGroup({
            layout: this._generatePipeline.getBindGroupLayout(0),
            entries: [
                buf(0, f.paramsBuffer),
                buf(1, this.genParamsBuffer),
                buf(2, f.particlePosBuffer),
                buf(3, f.particleVelBuffer),
                buf(4, f.particleMassBuffer),
                buf(5, f.particleDensityBuffer),
                buf(6, f.cellMassF32Buffer),
                buf(7, this.particleBuffer),
                buf(8, this.freeListBuffer),
                buf(9, this.freeListTopBuffer),
            ],
        });
        this._advectBG = d.createBindGroup({
            layout: this._advectPipeline.getBindGroupLayout(0),
            entries: [
                buf(0, f.paramsBuffer),
                buf(1, this.advectParamsBuffer),
                buf(2, this.particleBuffer),
                buf(3, f.cellMvF32Buffer),
                buf(4, this.freeListBuffer),
                buf(5, this.freeListTopBuffer),
            ],
        });
    }

    // Called once per substep, right after G2P — mirrors the CPU's
    // `diffuse.generate(fluid, sub_dt); diffuse.advect(fluid, sub_dt);`. Generate and
    // advect must be separate dispatches in that order (not one fused pass): a
    // particle spawned this substep should get exactly one advection step, matching
    // the CPU call order, and running them concurrently would race a freshly-spawned
    // slot against advect's alive-check for the same index.
    step(cmd, activeParticleNum) {
        this._seed = (this._seed + 1) >>> 0;
        this._genU32[7] = this._seed;
        this.device.queue.writeBuffer(this.genParamsBuffer, 28, this._genBuf, 28, 4);

        const wgGen = Math.ceil(activeParticleNum / WG_SIZE);
        const wgAdv = Math.ceil(this.maxCount / WG_SIZE);

        const genPass = cmd.beginComputePass();
        genPass.setPipeline(this._generatePipeline);
        genPass.setBindGroup(0, this._generateBG);
        genPass.dispatchWorkgroups(wgGen);
        genPass.end();

        const advPass = cmd.beginComputePass();
        advPass.setPipeline(this._advectPipeline);
        advPass.setBindGroup(0, this._advectBG);
        advPass.dispatchWorkgroups(wgAdv);
        advPass.end();
    }

    // Optional debug readback of "how many diffuse particles are currently alive"
    // (maxCount - freeListTop). Call requestDebugReadback(cmd) before queue.submit(),
    // then pollDebug() after — same self-throttling async pattern as FluidGPU's
    // deleteNear()/pollDelete().
    requestDebugReadback(cmd) {
        if (this._debugPending) return;
        cmd.copyBufferToBuffer(this.freeListTopBuffer, 0, this._debugReadBuffer, 0, 4);
        this._debugPending = true;
    }

    pollDebug() {
        if (!this._debugPending || this._debugMapping) return;
        this._debugMapping = true;
        this._debugReadBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const top = new Int32Array(this._debugReadBuffer.getMappedRange())[0];
            this._debugReadBuffer.unmap();
            this.aliveCount = this.maxCount - top;
            this._debugPending = false;
            this._debugMapping = false;
        });
    }
}
