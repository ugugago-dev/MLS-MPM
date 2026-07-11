// fluid-renderer.js — NRF depth-smoothed fluid renderer + Beer-Lambert thickness
// Pass 1  : sphere impostor → r32float linear depth (rawTex)          [FLUID_RES_SCALE res]
// Pass T1 : thickness accumulation → r16float (thickRawTex, additive) [FLUID_RES_SCALE res]
// Pass 2…  : NRF 1D H+V × NRF_ITERATIONS, MRT depth+thickness, ping-pong [FLUID_RES_SCALE res]
//            (filtA,thickA)↔(filtB,thickB); ends at (filtBTex, thickBTex)
// Pass ThB: thickness-only fixed-radius Gaussian blur H+V (grid-frequency thickness
//           mottling cleanup) → thickBTex→thickATex→thickBTex (thickATex reused as the
//           scratch buffer; it's free once the NRF chain above is done) [FLUID_RES_SCALE res]
// Pass Nf : foam/bubble thickness accumulation → r16float (foamThickRawTex, FOAM_RES_SCALE res, additive)
// Pass N  : normal reconstruct + shade → sceneColorTex (RENDER_SCALE res; upsamples FLUID_RES_SCALE filtB/thickB; 9×9 NRF cleanup inlined)
// Pass Sp : spray billboards → sceneColorTex (load; manual occlusion vs filtBTex)
// Pass Cf : foam composite → sceneColorTex (load; bilinear upsample, density→opacity sigmoid; discards outside the filtBTex fluid silhouette)
// Pass Bl : blit → swapchain (bilinear upscale of sceneColorTex + optional FXAA)
//
// Two independent resolution knobs:
//   FLUID_RES_SCALE — the fluid screen-space chain (pass1/T1/NRF) renders at this
//     fraction of the swapchain res; the shade pass upsamples those targets.
//   RENDER_SCALE    — the FINAL composited scene (shade/spray/foam-composite) renders
//     into an offscreen sceneColorTex at this fraction; the blit pass upscales it
//     (bilinear + optional FXAA) to the swapchain.
// They are orthogonal: shade both consumes FLUID_RES_SCALE targets AND writes at
// RENDER_SCALE. When RENDER_SCALE == FLUID_RES_SCALE the shade depth "upsample" is
// effectively 1:1, but the structure is kept so each can be tuned independently
// (e.g. RENDER_SCALE=1.0 native + FLUID_RES_SCALE=0.25 cheap fluid, or vice versa).
//
// NRF now MRT-filters depth AND thickness with a single (depth-derived) kernel, so
// the old dedicated thickness blur (T2/T3) is gone. NRF_ITERATIONS=1 → 2 filter
// passes total (1×H + 1×V), each writing both depth and thickness.

// DEBUG_PASS_TIMING: per-render-pass GPU timestamp-query profiling, fully
// self-contained in this file. Flip to false to strip it out completely — every
// piece of debug code below is gated on this flag (or on `dbgQuerySet` being
// non-null, which is itself gated on this flag at creation time). main.js is
// untouched; this uses its own querySet/buffers, independent of the tsRes/renTsq
// mechanism in main.js.
const DEBUG_PASS_TIMING = false;

const PARTICLE_RADIUS  = 0.45;          // grid units
const NRF_SIGMA        = 1.5  * PARTICLE_RADIUS;
const NRF_DELTA        = 10.0 * PARTICLE_RADIUS;
const NRF_MU           = 1  * PARTICLE_RADIUS;
const NRF_ITERATIONS   = 2;             // NRF H+V iterations (each iter = 1×H + 1×V, MRT depth+thickness)
                                        // 1 was tried for perf but graininess persisted (user-confirmed); filter
                                        // passes are not the render bottleneck anyway (6→2 passes bought ~0.1ms).
const BG_DEPTH         = -1.0;          // sentinel for background pixels
// Thickness-only fixed-radius Gaussian blur, run once after the NRF chain finishes
// (separate from the NRF-MRT thickness blur above, which is tied to NRF_SIGMA and
// too weak to smooth out the simulation's grid-frequency thickness mottling on its
// own). sigma/radius are in FLUID_RES_SCALE texel units.
const THICK_SMOOTH_SIGMA  = 3.0;
const THICK_SMOOTH_RADIUS = Math.ceil(2 * THICK_SMOOTH_SIGMA);
// Fluid screen-space resolution scale. 0.5 = half-res for pass1/T1/NRF chain
// (shade upsamples back to full res). Set to 1.0 to roll back to full-res.
const FLUID_RES_SCALE  = 0.5;
// Foam thickness resolution scale (unchanged: foam has always been half-res).
// Kept as a named constant so the foam→fluid texel remapping (foam f_att) can be
// derived from the FLUID_RES_SCALE / FOAM_RES_SCALE ratio instead of a hardcoded ×2.
const FOAM_RES_SCALE   = 0.5;
// Final scene render scale. shade/spray/foam-composite render into an offscreen
// scene texture at this scale of the swapchain, then one blit pass upscales it
// bilinearly (+FXAA) to the display. 1.0 = render at native res (blit becomes AA-only).
const RENDER_SCALE = 0.5;
const FXAA_ENABLED = true;   // cheap luma-based AA applied in the upscale blit
// FXAA tuning (baked into the blit shader; only used when FXAA_ENABLED). EDGE_*
// gate the early-out (skip AA on low-contrast texels); the rest is the classic
// FXAA3-console edge blend.
const EDGE_THRESHOLD     = 1.0 / 8.0;
const EDGE_THRESHOLD_MIN = 1.0 / 16.0;
const FXAA_SPAN_MAX      = 8.0;
const FXAA_REDUCE_MUL    = 1.0 / 8.0;
const FXAA_REDUCE_MIN    = 1.0 / 128.0;
const THICKNESS_ABSORPTION = [0.05, 0.02, 0.005]; // per world-unit RGB absorption (tune to taste)
const STRETCH_SENSITIVITY = 0.15;        // velocity → stretch sensitivity (larger = stretches more easily)
const STRETCH_MAX         = 1.5;        // max additional elongation (total = 1 + STRETCH_MAX)
// Floor spring-correction zone (must match FluidGPU.HARD_MIN / WALL_MIN in fluid-gpu.js):
// below HARD_MIN+WALL_MIN the G2P floor spring inflates particle vy, so the render
// stretch fades its vertical component to zero toward the floor.
const STRETCH_FLOOR_MIN  = 2.0;  // = FluidGPU.HARD_MIN
const STRETCH_FLOOR_FADE = 3.0;  // = FluidGPU.WALL_MIN (fade band height)

// Diffuse particle look, indexed by ptype (0=spray, 1=foam, 2=bubble). Sizes are
// relative to PARTICLE_RADIUS; no NRF/thickness treatment — these are drawn as
// simple alpha-blended billboards on top of the shaded fluid.
const DIFFUSE_SIZE_SCALE = [0.35, 0.55, 0.3];
const DIFFUSE_COLORS     = [[1.0, 1.0, 1.0], [0.9, 0.95, 1.0], [0.75, 0.88, 1.0]];
const DIFFUSE_ALPHA      = [0.9, 0.55, 0.35];
// Foam fades as its lifetime runs out (spray/bubble have lifetime pinned to 999
// in WGSL_DIFFUSE_GENERATE, so their ratio always clamps to 1.0 — no effect on them).
// The normalisation constant is read from diffuse.LIFETIME_FOAM at pipeline-build
// time (see initFluidRenderer) so it can never drift out of sync with the sim side.

// Screen-space foam rendering (Akinci et al., WSCG 2013). Foam/bubble diffuse
// particles are accumulated into a separate thickness buffer (foamThickRawTex),
// then composited over the shaded fluid with a density→opacity sigmoid. Spray
// stays on the alpha-blended billboard path (it reads as discrete droplets, not
// a continuous foam sheet). Constants are baked into WGSL via template literals,
// same flavour as the NRF_*/THICKNESS_ABSORPTION block above.
const FOAM_THICK_SCALE = 0.5;   // per-particle thickness contribution
const FOAM_RHO_MOD     = 3.0;   // intensity sigmoid denominator (paper ρ_mod)
const FOAM_RHO_EXP     = 1.25;  // intensity sigmoid exponent   (paper ρ_exp)
const FOAM_ETA_MAX     = 8.0;   // depth diff (grid units) at which foam behind the fluid surface fully vanishes
const FOAM_ETA_N       = 1.0;
const FOAM_ETA_M       = 1.0;
const FOAM_COLOR       = [1.0, 1.0, 1.0];

// Fullscreen-triangle vertex shader shared by every screen-space pass (NRF H/V,
// thickness adaptive blur, shade, foam composite) — spliced into each shader
// module via template literal, same pattern as WGSL_COMMON in shaders.js.
const WGSL_FULLSCREEN_VS = /* wgsl */`
    @vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
        let p = array<vec2<f32>,3>(vec2<f32>(-1,-1), vec2<f32>(3,-1), vec2<f32>(-1,3));
        return vec4<f32>(p[vi], 0.0, 1.0);
    }
`;

export async function initFluidRenderer(device, canvas, particlePosBuffer, particleVelBuffer, particleDensityBuffer, restDensity, diffuse = null) {
    const context = canvas.getContext("webgpu");
    const format  = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    // ── DEBUG_PASS_TIMING: dedicated timestamp-query set (independent of the
    // main.js tsRes/renTsq querySet) ──────────────────────────────────────
    // One (begin,end) pair per labeled pass. NRF spans several actual render
    // passes (H+V × NRF_ITERATIONS) but is measured as a single "nrf" span from
    // the first H pass's begin to the last V pass's end (see dbgTWBegin/dbgTWEnd
    // usage in render()). Passes that don't run in a given build (diffuse=null →
    // no foamNf/spray/foamCf) simply never get their query indices written;
    // resolveQuerySet reports unwritten slots as 0, so they read as 0ms — no
    // special-casing needed at readback time.
    const DBG_PASS_LABELS = ['pass1', 'thickT1', 'nrf', 'foamNf', 'shade', 'spray', 'foamCf', 'blit'];
    let dbgQuerySet = null, dbgResolveBuf = null, dbgReadBuf = null, dbgPending = false;
    const dbgEmaMs = new Float64Array(DBG_PASS_LABELS.length);
    let dbgFrameCounter = 0;
    if (DEBUG_PASS_TIMING && device.features.has('timestamp-query')) {
        const n = DBG_PASS_LABELS.length * 2;
        dbgQuerySet   = device.createQuerySet({ type: 'timestamp', count: n });
        dbgResolveBuf = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        dbgReadBuf    = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    // timestampWrites helpers — all return undefined (no-op) when disabled.
    const dbgTW = (label) => {
        if (!dbgQuerySet) return undefined;
        const i = DBG_PASS_LABELS.indexOf(label);
        return { querySet: dbgQuerySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 };
    };
    const dbgTWBegin = (label) => {
        if (!dbgQuerySet) return undefined;
        const i = DBG_PASS_LABELS.indexOf(label);
        return { querySet: dbgQuerySet, beginningOfPassWriteIndex: i * 2 };
    };
    const dbgTWEnd = (label) => {
        if (!dbgQuerySet) return undefined;
        const i = DBG_PASS_LABELS.indexOf(label);
        return { querySet: dbgQuerySet, endOfPassWriteIndex: i * 2 + 1 };
    };

    // ── Particle pass uniforms ────────────────────────────────────────────
    // viewProj(64) + view(64) + proj(64) + camRight+halfSize(16) + camUp+pad(16)
    //   + tanHalfFovY+aspect+near+far(16) + screenRes+pad(16) = 256 bytes.
    // screenRes (= the RENDER_SCALE scene-texture res, the shade pass's own render
    // target) is only read by the shade pass to map its fragment coords (and NDC)
    // onto the FLUID_RES_SCALE depth/thickness textures it upsamples from.
    const particleUniBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const particleUni    = new Float32Array(64);

    // ── Filter pass uniforms ──────────────────────────────────────────────
    // proj(64) + screenRes(8) + sigma(4) + delta(4) + mu(4) + _pad(4) = 88 → alloc 96
    const filterUniBuf = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const filterUni    = new Float32Array(24);
    filterUni[18] = NRF_SIGMA;
    filterUni[19] = NRF_DELTA;
    filterUni[20] = NRF_MU;

    // ── Pass 1: sphere impostor → linear depth ────────────────────────────
    const particleShader = device.createShaderModule({ code: /* wgsl */`
        struct U {
            viewProj : mat4x4<f32>, view : mat4x4<f32>, proj : mat4x4<f32>,
            camRight : vec3<f32>, halfSize : f32,
            camUp    : vec3<f32>, _pad     : f32,
            tanHalfFovY : f32, aspect : f32, zNear : f32, zFar : f32,
        };
        @group(0) @binding(0) var<uniform>      u           : U;
        @group(0) @binding(1) var<storage, read> density_arr : array<f32>;

        struct VsOut {
            @builtin(position) pos    : vec4<f32>,
            @location(0)       uv     : vec2<f32>,
            @location(1)       center : vec3<f32>,
            @location(2)       size   : f32,
            @location(3)       ex     : vec2<f32>,
            @location(4)       ey     : vec2<f32>,
        };

        @vertex fn vs(
            @location(0) pos: vec3<f32>, @location(1) vel: vec3<f32>,
            @builtin(vertex_index) vi: u32,
            @builtin(instance_index) iid: u32
        ) -> VsOut {
            // Single circumscribing triangle (3 verts) instead of a 6-vert quad:
            // its uv spans [-2,2], and the FS's dd=dot(uv,uv)>1 discard carves the
            // unit circle out of it exactly as before — half the vertex work.
            var C = array<vec2<f32>,3>(
                vec2<f32>(0, 2), vec2<f32>(-1.7320508, -1), vec2<f32>(1.7320508, -1)
            );
            let cn = C[vi % 3u];
            // density-based size: 0.6× when isolated, 1.0× at rest density
            let density_norm = clamp(density_arr[iid] / ${restDensity}, 0.0, 1.0);
            let sizeScale    = mix(0.6, 1.0, density_norm);
            let effSize      = u.halfSize * sizeScale;
            // Floor-zone stretch suppression: G2P's floor spring correction (shaders.js)
            // injects vertical velocity into particles resting near the floor, which the
            // velocity stretch would otherwise render as vertical elongation. Fade only
            // the vertical component so lateral splash stretch survives.
            var sv = vel;
            sv.y *= clamp((pos.y - ${STRETCH_FLOOR_MIN}) / ${STRETCH_FLOOR_FADE}, 0.0, 1.0);
            let vv = (u.view * vec4<f32>(sv, 0.0)).xy;
            let vl = length(vv);
            var ex = vec2<f32>(1, 0);
            if (vl > 1e-4) { ex = vv / vl; }
            let ey      = vec2<f32>(-ex.y, ex.x);
            // area-preserving stretch: scale in local coords before rotating
            let stretch = 1.0 + clamp(vl / effSize * ${STRETCH_SENSITIVITY}, 0.0, ${STRETCH_MAX});
            var lc = cn; lc.x *= stretch; lc.y /= stretch;
            lc = ex * lc.x + ey * lc.y;
            let world = pos + (lc.x * u.camRight + lc.y * u.camUp) * effSize;
            var o: VsOut;
            o.pos    = u.viewProj * vec4<f32>(world, 1.0);
            o.uv     = cn;
            o.center = (u.view * vec4<f32>(pos, 1.0)).xyz;
            o.size   = effSize;
            o.ex = ex; o.ey = ey;
            return o;
        }

        struct FsOut {
            @builtin(frag_depth) depth : f32,
            @location(0)         dep   : f32,
        };

        @fragment fn fs(in: VsOut) -> FsOut {
            let dd = dot(in.uv, in.uv);
            if (dd > 1.0) { discard; }
            let z  = sqrt(1.0 - dd);
            let xy = in.ex * in.uv.x + in.ey * in.uv.y;
            let vp = in.center + vec3<f32>(xy * in.size, z * in.size);
            let cl = u.proj * vec4<f32>(vp, 1.0);
            var o: FsOut;
            o.depth = cl.z / cl.w;
            o.dep   = -vp.z;   // positive linear view-space depth
            return o;
        }

        // Thickness pass: accumulate 2*sqrt(1-dd)*size per particle (additive blend, r16float)
        @fragment fn fsThick(in: VsOut) -> @location(0) f32 {
            let dd = dot(in.uv, in.uv);
            if (dd > 1.0) { discard; }
            return 2.0 * sqrt(1.0 - dd) * in.size;
        }
    ` });

    const particlePipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: particleShader, entryPoint: "vs",
            buffers: [
                { arrayStride: 16, stepMode: "instance", attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
                { arrayStride: 16, stepMode: "instance", attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
            ],
        },
        fragment: { module: particleShader, entryPoint: "fs", targets: [{ format: "r32float" }] },
        primitive: { topology: "triangle-list" },
        depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
    });

    const particleBG = device.createBindGroup({
        layout: particlePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleUniBuf } },
            { binding: 1, resource: { buffer: particleDensityBuffer } },
        ],
    });

    // Thickness accumulation pipeline: r16float additive blend (4× less bandwidth than rgba16float)
    // r16float is filterable → supports blending; single-channel avoids wasted g/b/a writes
    const thicknessPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: particleShader, entryPoint: "vs",
            buffers: [
                { arrayStride: 16, stepMode: "instance", attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }] },
                { arrayStride: 16, stepMode: "instance", attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }] },
            ],
        },
        fragment: {
            module: particleShader, entryPoint: "fsThick",
            targets: [{
                format: "r16float",
                blend: {
                    color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                    alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
                },
            }],
        },
        primitive: { topology: "triangle-list" },
    });
    const thickParticleBG = device.createBindGroup({
        layout: thicknessPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleUniBuf } },
            { binding: 1, resource: { buffer: particleDensityBuffer } },
        ],
    });

    // ── Filter uniform BGL (NRF 1D, MRT depth+thickness) ──────────────────
    // binding 1: depth (r32float), binding 2: thickness (r16float). Both read via
    // textureLoad, so both bound as unfilterable-float.
    const filterBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        ],
    });
    const filterPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [filterBGL] });

    // ── Pass 2 & 3: NRF 1D (horizontal and vertical), MRT depth + thickness ──
    // Direction is hardcoded per shader to avoid mid-frame uniform conflicts.
    // The NRF kernel weights are computed from depth ONLY (bilateral); the same
    // weights are reused to blur thickness in lockstep (target1). This replaces the
    // former standalone thickness blur (T2/T3): thickness blur amount is now tied to
    // NRF_SIGMA rather than an independent THICK_SIGMA.
    const mkNRF1DShader = (horiz) => {
        // For H: radius uses proj[0][0] (f/aspect) and the target width
        // For V: radius uses proj[1][1] (f)        and the target height
        // screenDim is taken from textureDimensions (= the render-target / FLUID_RES_SCALE
        // resolution), NOT the shared filter uniform's screenRes: that uniform still
        // carries the FULL-res dimensions for the foam composite pass. Reading the
        // target's own size here keeps the kernel radius resolution-independent.
        const projElem  = horiz ? 'u.proj[0][0]' : 'u.proj[1][1]';
        const screenDim = horiz ? 'f32(sz.x)' : 'f32(sz.y)';
        const stepDir   = horiz ? 'vec2<i32>(i, 0)' : 'vec2<i32>(0, i)';
        return device.createShaderModule({ code: /* wgsl */`
            struct FU {
                proj      : mat4x4<f32>,
                screenRes : vec2<f32>,
                sigma     : f32, delta : f32, mu : f32, _pad : f32,
            };
            @group(0) @binding(0) var<uniform> u : FU;
            @group(0) @binding(1) var depthTex   : texture_2d<f32>;
            @group(0) @binding(2) var thickTex   : texture_2d<f32>;

            ${WGSL_FULLSCREEN_VS}

            fn gaussian(dist: f32, isigma2: f32) -> f32 {
                return exp(-0.5 * dist * dist * isigma2);
            }

            struct FsOut {
                @location(0) depth : f32,   // r32float, NRF-smoothed
                @location(1) thick : f32,   // r16float, blurred with the same kernel
            };

            @fragment fn fs(@builtin(position) fragPos: vec4<f32>) -> FsOut {
                var o: FsOut;
                let sz    = vec2<i32>(textureDimensions(depthTex));
                let coord = vec2<i32>(fragPos.xy);
                let depth = textureLoad(depthTex, coord, 0).r;
                if (depth < 0.0) { o.depth = ${BG_DEPTH}; o.thick = 0.0; return o; }

                let thickC = textureLoad(thickTex, coord, 0).r;

                // Eq.5: screen-space kernel radius
                let rad = min(i32(ceil(u.sigma * ${projElem} * ${screenDim} / (2.0 * depth))), 100);
                if (rad == 0) { o.depth = depth; o.thick = thickC; return o; }

                let sigma2   = f32(rad) * f32(rad) / 9.0;
                let isigma2  = 1.0 / sigma2;

                // Dynamic Range initial values (Eq.7-9)
                var dLow  = u.delta;
                var dHigh = u.delta;

                var sum    = depth;
                var tsum   = thickC;
                var wsum   = 1.0;

                for (var i: i32 = 1; i <= rad; i++) {
                    let ncJ = clamp(coord - ${stepDir}, vec2<i32>(0), sz - vec2<i32>(1));
                    let ncK = clamp(coord + ${stepDir}, vec2<i32>(0), sz - vec2<i32>(1));
                    let zj  = textureLoad(depthTex, ncJ, 0).r;
                    let zk  = textureLoad(depthTex, ncK, 0).r;

                    let jBg = zj < 0.0;
                    let kBg = zk < 0.0;

                    // Dynamic Range update: foreground only
                    let jOk = !jBg && zj > depth - dLow  && zj < depth + dHigh;
                    let kOk = !kBg && zk > depth - dLow  && zk < depth + dHigh;
                    if (jOk) { dLow  = max(dLow,  depth - zj + u.delta);
                               dHigh = max(dHigh, zj - depth + u.delta); }
                    if (kOk) { dLow  = max(dLow,  depth - zk + u.delta);
                               dHigh = max(dHigh, zk - depth + u.delta); }

                    // Bias Correction (Eq.6): foreground outlier → skip pair
                    let outlier = (!jBg && zj > depth + dHigh) || (!kBg && zk > depth + dHigh);
                    let w = select(gaussian(f32(i), isigma2), 0.0, outlier);

                    // Clamp function (Eq.2): too-far-back foreground → depth - mu
                    let fj = select(zj, depth - u.mu, zj < depth - dLow);
                    let fk = select(zk, depth - u.mu, zk < depth - dLow);

                    // If either side is background, skip the whole pair.
                    // Per-pixel zeroing would asymmetrically pull silhouette depth
                    // toward the interior, inverting the depth gradient and creating
                    // wrong (downward) normals at the fluid edge.
                    let pairBg = jBg || kBg;
                    let wj = select(w, 0.0, pairBg);
                    let wk = select(w, 0.0, pairBg);
                    sum  += fj * wj + fk * wk;
                    wsum += wj + wk;

                    // Thickness: same weights (wj/wk already fold in outlier / background
                    // rejection), raw neighbour thickness values (no depth-clamp analog).
                    let tj = textureLoad(thickTex, ncJ, 0).r;
                    let tk = textureLoad(thickTex, ncK, 0).r;
                    tsum += tj * wj + tk * wk;
                }

                o.depth = sum / wsum;
                o.thick = tsum / wsum;
                return o;
            }
        ` });
    };

    const mkFilterPipeline = (mod) => device.createRenderPipeline({
        layout: filterPipelineLayout,
        vertex:   { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: "r32float" }, { format: "r16float" }] },
        primitive: { topology: "triangle-list" },
    });

    const filterHPipeline = mkFilterPipeline(mkNRF1DShader(true));
    const filterVPipeline = mkFilterPipeline(mkNRF1DShader(false));

    // ── Thickness-only Gaussian blur (post-NRF) ───────────────────────────
    // Fixed-radius separable blur, independent of depth/silhouette — runs once
    // after the NRF chain finishes, H then V, reusing thickATex as the scratch
    // buffer (thickB → thickA → thickB). Background pixels carry thickness 0 from
    // the NRF stage already, so no depth-masking is needed here (shade only colors
    // depth>0 pixels anyway, per the comment in the task).
    const thickSmoothBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        ],
    });
    const thickSmoothLayout = device.createPipelineLayout({ bindGroupLayouts: [thickSmoothBGL] });

    const mkThickSmoothShader = (horiz) => {
        const stepDir = horiz ? 'vec2<i32>(i, 0)' : 'vec2<i32>(0, i)';
        return device.createShaderModule({ code: /* wgsl */`
            @group(0) @binding(0) var srcTex : texture_2d<f32>;

            ${WGSL_FULLSCREEN_VS}

            @fragment fn fs(@builtin(position) fragPos: vec4<f32>) -> @location(0) f32 {
                let sz    = vec2<i32>(textureDimensions(srcTex));
                let coord = vec2<i32>(fragPos.xy);
                var sum  = textureLoad(srcTex, coord, 0).r;
                var wsum = 1.0;
                let isigma2 = 1.0 / (${THICK_SMOOTH_SIGMA} * ${THICK_SMOOTH_SIGMA});
                for (var i: i32 = 1; i <= ${THICK_SMOOTH_RADIUS}; i++) {
                    let ncJ = clamp(coord - ${stepDir}, vec2<i32>(0), sz - vec2<i32>(1));
                    let ncK = clamp(coord + ${stepDir}, vec2<i32>(0), sz - vec2<i32>(1));
                    let w = exp(-0.5 * f32(i) * f32(i) * isigma2);
                    sum  += (textureLoad(srcTex, ncJ, 0).r + textureLoad(srcTex, ncK, 0).r) * w;
                    wsum += 2.0 * w;
                }
                return sum / wsum;
            }
        ` });
    };

    const mkThickSmoothPipeline = (mod) => device.createRenderPipeline({
        layout: thickSmoothLayout,
        vertex:   { module: mod, entryPoint: "vs" },
        fragment: { module: mod, entryPoint: "fs", targets: [{ format: "r16float" }] },
        primitive: { topology: "triangle-list" },
    });
    const thickSmoothHPipeline = mkThickSmoothPipeline(mkThickSmoothShader(true));
    const thickSmoothVPipeline = mkThickSmoothPipeline(mkThickSmoothShader(false));

    // ── Pass 4: normal reconstruct + shade (cleanup inlined) ─────────────
    // binding 1: depth (r32float, unfilterable — read via textureLoad + manual
    //            bilateral bilinear so the -1 background sentinel never blends).
    // binding 2: thickness (r16float, filterable — plain bilinear via sampler).
    // binding 3: filtering sampler for the thickness bilinear upsample.
    const shadeBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        ],
    });

    const shadeShader = device.createShaderModule({ code: /* wgsl */`
        struct U {
            viewProj : mat4x4<f32>, view : mat4x4<f32>, proj : mat4x4<f32>,
            camRight : vec3<f32>, halfSize : f32,
            camUp    : vec3<f32>, _pad     : f32,
            tanHalfFovY : f32, aspect : f32, zNear : f32, zFar : f32,
            screenRes : vec2<f32>, _pad2 : vec2<f32>,   // RENDER_SCALE scene res (this pass's target; upsamples FLUID_RES_SCALE depth/thick)
        };
        @group(0) @binding(0) var<uniform> u        : U;
        @group(0) @binding(1) var          depthTex : texture_2d<f32>;
        @group(0) @binding(2) var          thickTex : texture_2d<f32>;
        @group(0) @binding(3) var          thickSamp : sampler;

        ${WGSL_FULLSCREEN_VS}

        // Bilateral bilinear upsample of the half-res depth. Background samples
        // (depth < 0) are excluded from the weighted sum so silhouette edges never
        // blend fluid depth with the -1 sentinel. Returns BG_DEPTH if all 4 taps are bg.
        fn sampleDepthBilinear(uv: vec2<f32>) -> f32 {
            let dims = vec2<f32>(textureDimensions(depthTex));
            let c    = uv * dims - 0.5;
            let i0   = vec2<i32>(floor(c));
            let f    = c - floor(c);
            var sum = 0.0; var wsum = 0.0;
            for (var j = 0; j < 2; j++) {
                for (var i = 0; i < 2; i++) {
                    let t = clamp(i0 + vec2<i32>(i, j), vec2<i32>(0), vec2<i32>(dims) - 1);
                    let d = textureLoad(depthTex, t, 0).r;
                    if (d < 0.0) { continue; }
                    let w = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
                    sum += w * d; wsum += w;
                }
            }
            if (wsum <= 0.0) { return ${BG_DEPTH}; }
            return sum / wsum;
        }

        @fragment fn fs(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
            // The fluid depth/thickness are at FLUID_RES_SCALE; this pass renders at
            // RENDER_SCALE (u.screenRes = its own target size). Map this pass's fragment
            // to the fluid texel via UV so it is resolution-independent (no hardcoded
            // ratio). All neighbour taps below then operate in fluid-texel space.
            let sz    = vec2<f32>(textureDimensions(depthTex));   // FLUID_RES_SCALE dims
            let isz   = vec2<i32>(sz);
            let uv    = fragPos.xy / u.screenRes;                 // scene-res → [0,1)
            let coord = vec2<i32>(uv * sz);                       // → fluid texel
            let rawDepth = textureLoad(depthTex, coord, 0).r;
            if (rawDepth < 0.0) { return vec4<f32>(0.08, 0.08, 0.08, 1.0); }

            // Inline cleanup: 9×9 symmetric-pair 2D NRF (was a separate pass)
            var csum  = rawDepth;
            var cwsum = 1.0;
            for (var ci: i32 = 0; ci <= 4; ci++) {
                let jStart = select(-4, 1, ci == 0);
                for (var cj: i32 = jStart; cj <= 4; cj++) {
                    let ncA = clamp(coord - vec2<i32>(ci, cj), vec2<i32>(0), isz - vec2<i32>(1));
                    let ncB = clamp(coord + vec2<i32>(ci, cj), vec2<i32>(0), isz - vec2<i32>(1));
                    let zA  = textureLoad(depthTex, ncA, 0).r;
                    let zB  = textureLoad(depthTex, ncB, 0).r;
                    let aBg = zA < 0.0;
                    let bBg = zB < 0.0;
                    let outlier = (!aBg && zA > rawDepth + ${NRF_DELTA}) ||
                                  (!bBg && zB > rawDepth + ${NRF_DELTA});
                    let dist = length(vec2<f32>(f32(ci), f32(cj)));
                    let w    = select(exp(-0.125 * dist * dist), 0.0, outlier);
                    let fA   = select(zA, rawDepth - ${NRF_MU}, zA < rawDepth - ${NRF_DELTA});
                    let fB   = select(zB, rawDepth - ${NRF_MU}, zB < rawDepth - ${NRF_DELTA});
                    let w2   = select(w, 0.0, aBg || bBg);
                    csum  += (fA + fB) * w2;
                    cwsum += 2.0 * w2;
                }
            }
            let cleanDepth = csum / cwsum;

            // Bilateral bilinear upsample of the half-res depth so the shaded depth
            // (and the reconstructed normal) varies per full-res pixel instead of
            // being constant across each 2×2 half-res block (the source of the block
            // artifact). The 25-tap cleanup above is a denoise; carry its correction
            // as an offset onto the smooth bilinear depth so we keep the denoise
            // without re-running the 25-tap cleanup at 4 bilinear corners.
            let depth = sampleDepthBilinear(uv) + (cleanDepth - rawDepth);

            // Normal-reconstruction neighbours: bilinear-sampled at ±1 half-res texel
            // in UV so the depth gradient (hence the normal) is smooth per full-res px.
            let texel = 1.0 / sz;
            let dR = sampleDepthBilinear(uv + vec2<f32>( texel.x, 0.0));
            let dL = sampleDepthBilinear(uv + vec2<f32>(-texel.x, 0.0));
            let dU = sampleDepthBilinear(uv + vec2<f32>(0.0, -texel.y));
            let dD = sampleDepthBilinear(uv + vec2<f32>(0.0,  texel.y));

            // World-space size of one pixel at this depth.
            // Using worldPerPixel instead of full reconstruct() avoids perspective
            // contamination: reconstruct(neighbor) - reconstruct(center) can flip
            // tangent.y sign in the lower screen half (ndcY < 0) when depth differs.
            let wpX = 2.0 * depth * u.tanHalfFovY * u.aspect / sz.x;
            let wpY = 2.0 * depth * u.tanHalfFovY / sz.y;
            let GRAD_SCALE = 50.0;

            let validR = dR > 0.0 && abs(dR - depth) < wpX * GRAD_SCALE;
            let validL = dL > 0.0 && abs(dL - depth) < wpX * GRAD_SCALE;
            let validU = dU > 0.0 && abs(dU - depth) < wpY * GRAD_SCALE;
            let validD = dD > 0.0 && abs(dD - depth) < wpY * GRAD_SCALE;

            // dzdx > 0 = right is farther; dzdy > 0 = upper (dU) is farther
            var dzdx: f32;
            if (validR && validL) { dzdx = (dR - dL) * 0.5; }
            else if (validR)      { dzdx = dR - depth; }
            else if (validL)      { dzdx = depth - dL; }
            else                  { dzdx = 0.0; }

            var dzdy: f32;
            if (validU && validD) { dzdy = (dU - dD) * 0.5; }
            else if (validU)      { dzdy = dU - depth; }
            else if (validD)      { dzdy = depth - dD; }
            else                  { dzdy = 0.0; }

            // cross((wpX,0,-dzdx), (0,wpY,-dzdy)) = (dzdx*wpY, dzdy*wpX, wpX*wpY)
            // n.z = wpX*wpY > 0 always → always front-facing, no flip needed
            let n_raw = vec3<f32>(dzdx * wpY, dzdy * wpX, wpX * wpY);
            let n_len = length(n_raw);
            var n = select(vec3<f32>(0.0, 0.0, 1.0), n_raw / n_len, n_len > 1e-8);

            // View-space position reconstructed from depth. NDC comes from this pass's
            // own fragment (u.screenRes = scene res), independent of the fluid depth texture.
            let ndcX =  2.0 * fragPos.x / u.screenRes.x - 1.0;
            let ndcY = -(2.0 * fragPos.y / u.screenRes.y - 1.0);
            let pos = vec3<f32>(ndcX * u.aspect * u.tanHalfFovY * depth,
                                ndcY * u.tanHalfFovY * depth, -depth);

            // ここでシェーダ処理
            // Schlick Fresnel (F0 = 0.02, water)
            let viewDir = normalize(-pos);
            let NdotV   = max(dot(n, viewDir), 0.0);
            let fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);
            let skyColor = vec3<f32>(0.75, 0.88, 1.0);

            // Beer-Lambert: thick water absorbs more light (R absorbed fastest → blue tint)
            let thickness     = textureSampleLevel(thickTex, thickSamp, uv, 0.0).r;
            let absorption    = vec3<f32>(${THICKNESS_ABSORPTION.join(", ")});
            let transmittance = exp(-absorption * thickness);
            // Thin edges: light sky-blue; thick centre: deep water blue
            let deepWater  = vec3<f32>(0.04, 0.25, 0.60);
            let shallowWater = vec3<f32>(0.45, 0.72, 0.90);
            let bodyColor  = mix(shallowWater, deepWater, 1.0 - transmittance.g);

            return vec4<f32>(bodyColor + fresnel * skyColor, 1.0);
        }
    ` });

    const shadePipeline = device.createRenderPipeline({
        layout:    device.createPipelineLayout({ bindGroupLayouts: [shadeBGL] }),
        vertex:   { module: shadeShader, entryPoint: "vs" },
        fragment: { module: shadeShader, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
    });

    // Filtering sampler for the half-res thickness bilinear upsample in the shade pass
    // (depth is upsampled manually via sampleDepthBilinear; thickness has no sentinel
    // to protect so a plain bilinear sampler is fine — same as the foam composite).
    const shadeSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

    // ── Diffuse particles (spray/foam/bubble): simple alpha-blended billboards,
    // drawn on top of the shaded fluid. No NRF/thickness treatment — occlusion
    // against the fluid surface is done manually in the fragment shader by
    // comparing against filtBTex (the final smoothed fluid depth), since the
    // shade pass has no hardware depth buffer of its own to test against.
    let diffusePipeline = null;
    if (diffuse) {
        const diffuseShader = device.createShaderModule({ code: /* wgsl */`
            struct U {
                viewProj : mat4x4<f32>, view : mat4x4<f32>, proj : mat4x4<f32>,
                camRight : vec3<f32>, halfSize : f32,
                camUp    : vec3<f32>, _pad     : f32,
                tanHalfFovY : f32, aspect : f32, zNear : f32, zFar : f32,
            };
            @group(0) @binding(0) var<uniform> u        : U;
            @group(0) @binding(1) var          depthTex : texture_2d<f32>;

            struct VsOut {
                @builtin(position) pos    : vec4<f32>,
                @location(0)       uv     : vec2<f32>,
                @location(1)       center : vec3<f32>,
                @location(2)       size   : f32,
                @location(3)       color  : vec3<f32>,
                @location(4)       alpha  : f32,
                @location(5)       ex     : vec2<f32>,
                @location(6)       ey     : vec2<f32>,
                @location(7) @interpolate(flat) ptype : u32,
            };

            @vertex fn vs(
                @location(0) pos: vec4<f32>,
                @location(1) vel: vec4<f32>,
                @location(2) ptype: u32,
                @location(3) alive: u32,
                @location(4) lifetime: f32,
                @builtin(vertex_index) vi: u32,
            ) -> VsOut {
                var o: VsOut;
                if (alive == 0u) {
                    // NaN the clip position so the rasterizer drops this instance entirely
                    // (dead pool slots stay resident in the buffer; see diffuse-gpu.js).
                    // Both a literal 0.0/0.0 and bitcast<f32>(0x7fc00000u) are rejected at
                    // shader-creation time — WGSL const-evaluates them and disallows a NaN
                    // constant. Dividing by a runtime-derived zero (from the alive vertex
                    // attribute, guaranteed 0 here but not const-foldable) defers the divide
                    // to actual IEEE754 execution, which does produce NaN.
                    let z = f32(alive);
                    let nan = z / z;
                    o.pos = vec4<f32>(nan, nan, nan, nan);
                    o.uv = vec2<f32>(0.0); o.center = vec3<f32>(0.0); o.size = 0.0;
                    o.color = vec3<f32>(0.0); o.alpha = 0.0;
                    o.ex = vec2<f32>(1.0, 0.0); o.ey = vec2<f32>(0.0, 1.0);
                    o.ptype = ptype;
                    return o;
                }

                var C = array<vec2<f32>,6>(
                    vec2<f32>(-1,-1), vec2<f32>(1,-1), vec2<f32>(-1,1),
                    vec2<f32>(-1,1),  vec2<f32>(1,-1), vec2<f32>(1,1)
                );
                let cn = C[vi % 6u];

                var SIZE_SCALE = array<f32,3>(${DIFFUSE_SIZE_SCALE.join(', ')});
                var COLOR = array<vec3<f32>,3>(
                    ${DIFFUSE_COLORS.map(c => `vec3<f32>(${c.join(', ')})`).join(',\n                    ')}
                );
                var ALPHA = array<f32,3>(${DIFFUSE_ALPHA.join(', ')});

                let shrink = clamp(lifetime / ${diffuse.LIFETIME_FOAM}, 0.0, 1.0);
                let effSize = u.halfSize * SIZE_SCALE[ptype] * shrink;

                // Velocity-based stretch — same area-preserving technique as the main
                // fluid particles (see the particleShader vs() above): scale in local
                // (ex,ey) coords before rotating into the camera-facing basis so the
                // billboard's screen-space area stays constant regardless of speed.
                let vv = (u.view * vec4<f32>(vel.xyz, 0.0)).xy;
                let vl = length(vv);
                var ex = vec2<f32>(1.0, 0.0);
                if (vl > 1e-4) { ex = vv / vl; }
                let ey = vec2<f32>(-ex.y, ex.x);
                let stretch = 1.0 + clamp(vl / effSize * ${STRETCH_SENSITIVITY}, 0.0, ${STRETCH_MAX});
                var lc = cn; lc.x *= stretch; lc.y /= stretch;
                lc = ex * lc.x + ey * lc.y;

                let world = pos.xyz + (lc.x * u.camRight + lc.y * u.camUp) * effSize;

                o.pos    = u.viewProj * vec4<f32>(world, 1.0);
                o.uv     = cn;
                o.center = (u.view * vec4<f32>(pos.xyz, 1.0)).xyz;
                o.size   = effSize;
                o.color  = COLOR[ptype];
                o.alpha  = ALPHA[ptype];
                o.ex = ex; o.ey = ey;
                o.ptype = ptype;
                return o;
            }

            @fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
                // Spray-only: foam/bubble are rendered by the screen-space foam
                // accumulate + composite passes (foamThickPipeline/foamCompositePipeline).
                if (in.ptype != 0u) { discard; }
                let dd = dot(in.uv, in.uv);
                if (dd > 1.0) { discard; }
                let z  = sqrt(1.0 - dd);
                let xy = in.ex * in.uv.x + in.ey * in.uv.y;
                let vp = in.center + vec3<f32>(xy * in.size, z * in.size);
                let dep = -vp.z;   // positive linear view-space depth, same convention as the fluid pass

                // Occlude against the fluid surface: if the (already-filtered) fluid depth
                // at this pixel is nearer than this particle, hide it. This pass renders at
                // RENDER_SCALE but filtBTex is at FLUID_RES_SCALE, so remap the fragment
                // coord by the resolution ratio (no hardcoded factor). fragPos(scene) /
                // RENDER_SCALE → full-res px, × FLUID_RES_SCALE → fluid texel.
                let coord      = vec2<i32>(in.pos.xy * (${FLUID_RES_SCALE} / ${RENDER_SCALE}));
                let fluidDepth = textureLoad(depthTex, coord, 0).r;
                if (fluidDepth > 0.0 && fluidDepth < dep - 0.05) { discard; }

                return vec4<f32>(in.color, in.alpha);
            }
        ` });

        diffusePipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: diffuseShader, entryPoint: "vs",
                buffers: [{
                    arrayStride: 48, stepMode: "instance",
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: "float32x4" }, // pos
                        { shaderLocation: 1, offset: 16, format: "float32x4" }, // vel
                        { shaderLocation: 2, offset: 32, format: "uint32" },    // ptype
                        { shaderLocation: 3, offset: 36, format: "uint32" },    // alive
                        { shaderLocation: 4, offset: 40, format: "float32" },   // lifetime
                    ],
                }],
            },
            fragment: {
                module: diffuseShader, entryPoint: "fs",
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                        alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
                    },
                }],
            },
            primitive: { topology: "triangle-list" },
        });
    }

    // ── Screen-space foam (Akinci et al., WSCG 2013) ──────────────────────
    // Pass Nf accumulates foam/bubble thickness into foamThickRawTex (additive,
    // r16float — mirrors the fluid thicknessPipeline). Pass Cf composites it over
    // the shaded fluid with a density→opacity sigmoid. Only built when diffuse
    // particles exist, since both passes are driven by the diffuse pool.
    let foamThickPipeline = null, foamCompositePipeline = null;
    let foamThickBGL = null, foamCompositeBGL = null;
    let foamSampler = null;
    if (diffuse) {
        // binding 0: particle uniforms (VS only — sizes/stretch/basis)
        // binding 1: filtBTex (fluid depth, unfilterable-float, textureLoad)
        foamThickBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX,   buffer: {} },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
            ],
        });

        const foamThickShader = device.createShaderModule({ code: /* wgsl */`
            struct U {
                viewProj : mat4x4<f32>, view : mat4x4<f32>, proj : mat4x4<f32>,
                camRight : vec3<f32>, halfSize : f32,
                camUp    : vec3<f32>, _pad     : f32,
                tanHalfFovY : f32, aspect : f32, zNear : f32, zFar : f32,
            };
            @group(0) @binding(0) var<uniform> u        : U;
            @group(0) @binding(1) var          depthTex : texture_2d<f32>;

            // Shared radial/temporal falloff (paper Eq. shape term). Returns
            // (1 - (x/b)^n)^m for x < b, 0 beyond b.
            fn falloff(x: f32, b: f32, n: f32, m: f32) -> f32 {
                let r = x / b;
                if (r >= 1.0) { return 0.0; }
                return pow(1.0 - pow(r, n), m);
            }

            struct VsOut {
                @builtin(position) pos      : vec4<f32>,
                @location(0)       uv       : vec2<f32>,
                @location(1)       center   : vec3<f32>,
                @location(2)       size     : f32,
                @location(3)       ex       : vec2<f32>,
                @location(4)       ey       : vec2<f32>,
                @location(5) @interpolate(flat) ptype : u32,
                @location(6)       lifetime : f32,
            };

            @vertex fn vs(
                @location(0) pos: vec4<f32>,
                @location(1) vel: vec4<f32>,
                @location(2) ptype: u32,
                @location(3) alive: u32,
                @location(4) lifetime: f32,
                @builtin(vertex_index) vi: u32,
            ) -> VsOut {
                var o: VsOut;
                if (alive == 0u) {
                    // Same runtime-divide NaN trick as the diffuse billboard VS: drop
                    // dead pool slots by NaN-ing the clip position (see diffuse-gpu.js).
                    let z = f32(alive);
                    let nan = z / z;
                    o.pos = vec4<f32>(nan, nan, nan, nan);
                    o.uv = vec2<f32>(0.0); o.center = vec3<f32>(0.0); o.size = 0.0;
                    o.ex = vec2<f32>(1.0, 0.0); o.ey = vec2<f32>(0.0, 1.0);
                    o.ptype = ptype; o.lifetime = lifetime;
                    return o;
                }

                var C = array<vec2<f32>,6>(
                    vec2<f32>(-1,-1), vec2<f32>(1,-1), vec2<f32>(-1,1),
                    vec2<f32>(-1,1),  vec2<f32>(1,-1), vec2<f32>(1,1)
                );
                let cn = C[vi % 6u];

                var SIZE_SCALE = array<f32,3>(${DIFFUSE_SIZE_SCALE.join(', ')});
                // No lifetime shrink here — the temporal fade is done once in the FS
                // (f_lifetime), to avoid double-fading the foam.
                let effSize = u.halfSize * SIZE_SCALE[ptype];

                // Velocity-based area-preserving stretch (same as diffuse billboard VS).
                let vv = (u.view * vec4<f32>(vel.xyz, 0.0)).xy;
                let vl = length(vv);
                var ex = vec2<f32>(1.0, 0.0);
                if (vl > 1e-4) { ex = vv / vl; }
                let ey = vec2<f32>(-ex.y, ex.x);
                let stretch = 1.0 + clamp(vl / effSize * ${STRETCH_SENSITIVITY}, 0.0, ${STRETCH_MAX});
                var lc = cn; lc.x *= stretch; lc.y /= stretch;
                lc = ex * lc.x + ey * lc.y;

                let world = pos.xyz + (lc.x * u.camRight + lc.y * u.camUp) * effSize;

                o.pos      = u.viewProj * vec4<f32>(world, 1.0);
                o.uv       = cn;
                o.center   = (u.view * vec4<f32>(pos.xyz, 1.0)).xyz;
                o.size     = effSize;
                o.ex = ex; o.ey = ey;
                o.ptype    = ptype;
                o.lifetime = lifetime;
                return o;
            }

            // Additive thickness accumulation → r16float (mirrors fluid fsThick).
            @fragment fn fs(in: VsOut) -> @location(0) f32 {
                if (in.ptype == 0u) { discard; }   // spray handled by the billboard pass
                let x = length(in.uv);
                if (x > 1.0) { discard; }
                var shape: f32;
                if (in.ptype == 1u) { shape = falloff(x, 1.0, 2.25, 1.0); }        // foam: filled disc
                else                { shape = 1.0 - falloff(x, 1.0, 2.0, 1.0); }    // bubble: hollow ring
                let age = 1.0 - clamp(in.lifetime / ${diffuse.LIFETIME_FOAM}, 0.0, 1.0);
                let fl  = falloff(age, 1.0, 2.0, 0.4);

                // Sphere depth reconstruction (same as diffuse billboard FS).
                let dd = dot(in.uv, in.uv);
                let z  = sqrt(1.0 - dd);
                let xy = in.ex * in.uv.x + in.ey * in.uv.y;
                let vp = in.center + vec3<f32>(xy * in.size, z * in.size);
                let dep = -vp.z;   // positive linear view-space depth

                // f_att: continuous attenuation when foam sits behind the fluid surface
                // (replaces the billboard's hard occlusion discard).
                // This pass renders at FOAM_RES_SCALE and filtBTex is at FLUID_RES_SCALE, so
                // remap the fragment coord by the resolution ratio (no hardcoded ×2). With
                // both = 0.5 the ratio is 1.0 (foam and fluid depth are the same size).
                let ef = textureLoad(depthTex, vec2<i32>(in.pos.xy * (${FLUID_RES_SCALE} / ${FOAM_RES_SCALE})), 0).r;  // filtBTex, background = -1.0
                var fatt = 1.0;
                if (ef > 0.0 && dep > ef) { fatt = falloff(dep - ef, ${FOAM_ETA_MAX}, ${FOAM_ETA_N}, ${FOAM_ETA_M}); }

                return ${FOAM_THICK_SCALE} * shape * fl * fatt;
            }
        ` });

        foamThickPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [foamThickBGL] }),
            vertex: {
                module: foamThickShader, entryPoint: "vs",
                buffers: [{
                    arrayStride: 48, stepMode: "instance",
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: "float32x4" }, // pos
                        { shaderLocation: 1, offset: 16, format: "float32x4" }, // vel
                        { shaderLocation: 2, offset: 32, format: "uint32" },    // ptype
                        { shaderLocation: 3, offset: 36, format: "uint32" },    // alive
                        { shaderLocation: 4, offset: 40, format: "float32" },   // lifetime
                    ],
                }],
            },
            fragment: {
                module: foamThickShader, entryPoint: "fs",
                targets: [{
                    format: "r16float",
                    blend: {
                        color: { srcFactor: "one", dstFactor: "one", operation: "add" },
                        alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
                    },
                }],
            },
            primitive: { topology: "triangle-list" },
        });

        // Composite: foam thickness → opacity sigmoid, alpha-blended over the swapchain.
        // foamThickRawTex is half-resolution, so it is bilinearly upsampled here via a
        // filtering sampler (r16float is filterable) — nearest would show blocky edges.
        // binding 0: screenRes uniform (reuses filterUniBuf), binding 1: foam thickness,
        // binding 2: filtering sampler, binding 3: filtBTex (fluid depth, for silhouette
        // clipping — background pixels behind/outside the fluid must not show foam that
        // the bilinear upsample smeared past the fluid's edge).
        foamCompositeBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
            ],
        });

        const foamCompositeShader = device.createShaderModule({ code: /* wgsl */`
            struct FU {
                proj      : mat4x4<f32>,
                screenRes : vec2<f32>,
                sigma     : f32, delta : f32, mu : f32, _pad : f32,
            };
            @group(0) @binding(0) var<uniform> u        : FU;
            @group(0) @binding(1) var          foamTex  : texture_2d<f32>;
            @group(0) @binding(2) var          foamSamp : sampler;
            @group(0) @binding(3) var          depthTex : texture_2d<f32>;

            ${WGSL_FULLSCREEN_VS}

            @fragment fn fs(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
                // UV in scene-resolution screen space (this pass renders into sceneColorTex
                // at RENDER_SCALE, so u.screenRes carries the scene res); the filtering
                // sampler upsamples the FOAM_RES_SCALE foam thickness bilinearly.
                let uv  = fragPos.xy / u.screenRes;
                let rho = textureSampleLevel(foamTex, foamSamp, uv, 0.0).r;
                if (rho <= 0.0) { discard; }

                // Clip against the fluid silhouette: filtBTex is at FLUID_RES_SCALE, this
                // pass renders at RENDER_SCALE, so remap the fragment coord by the
                // resolution ratio (same convention as the spray-occlusion FS above).
                let coord = vec2<i32>(fragPos.xy * (${FLUID_RES_SCALE} / ${RENDER_SCALE}));
                let fluidDepth = textureLoad(depthTex, coord, 0).r;
                if (fluidDepth < 0.0) { discard; }

                let re   = pow(rho, ${FOAM_RHO_EXP});
                let iota = re / (${FOAM_RHO_MOD} + re);
                return vec4<f32>(${FOAM_COLOR.join(', ')}, iota);
            }
        ` });

        foamCompositePipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [foamCompositeBGL] }),
            vertex:   { module: foamCompositeShader, entryPoint: "vs" },
            fragment: {
                module: foamCompositeShader, entryPoint: "fs",
                targets: [{
                    format,
                    blend: {
                        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                        alpha: { srcFactor: "one",       dstFactor: "one-minus-src-alpha", operation: "add" },
                    },
                }],
            },
            primitive: { topology: "triangle-list" },
        });

        // Filtering sampler for the half-res foam thickness bilinear upsample.
        foamSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
    }

    // ── Final blit: sceneColorTex (RENDER_SCALE) → swapchain, bilinear + FXAA ──
    // The whole composited scene (shade + spray + foam composite) is rendered at
    // RENDER_SCALE into sceneColorTex; this pass upscales it to native swapchain
    // resolution with a filtering sampler. When FXAA_ENABLED, a light luma-based
    // FXAA (FXAA3-console flavour, ~11 taps) runs in scene-texel space first; the
    // branch is baked at build time (no runtime uniform toggle).
    // binding 0: fullRes uniform (swapchain size, for UV), binding 1: sceneColorTex,
    // binding 2: filtering sampler.
    const blitUniBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const blitUni    = new Float32Array(4);
    const blitBGL = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
            { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        ],
    });
    const blitSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

    // FXAA body vs. plain bilinear — chosen at build time so the unused branch is
    // never emitted into the shader.
    const blitSampleBody = FXAA_ENABLED ? /* wgsl */`
                // FXAA3-console: detect the local luma edge, blend along it.
                let sceneDims = vec2<f32>(textureDimensions(sceneTex));
                let inv = 1.0 / sceneDims;
                let rgbM = textureSampleLevel(sceneTex, samp, uv, 0.0).rgb;
                let lM  = luma(rgbM);
                let lNW = luma(textureSampleLevel(sceneTex, samp, uv + vec2<f32>(-1.0,-1.0) * inv, 0.0).rgb);
                let lNE = luma(textureSampleLevel(sceneTex, samp, uv + vec2<f32>( 1.0,-1.0) * inv, 0.0).rgb);
                let lSW = luma(textureSampleLevel(sceneTex, samp, uv + vec2<f32>(-1.0, 1.0) * inv, 0.0).rgb);
                let lSE = luma(textureSampleLevel(sceneTex, samp, uv + vec2<f32>( 1.0, 1.0) * inv, 0.0).rgb);

                let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
                let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
                let range = lMax - lMin;
                // Low-contrast texel → no edge, keep the plain bilinear sample.
                if (range < max(${EDGE_THRESHOLD_MIN}, lMax * ${EDGE_THRESHOLD})) {
                    return vec4<f32>(rgbM, 1.0);
                }

                var dir = vec2<f32>(
                    -((lNW + lNE) - (lSW + lSE)),
                     ((lNW + lSW) - (lNE + lSE))
                );
                let dirReduce = max((lNW + lNE + lSW + lSE) * 0.25 * ${FXAA_REDUCE_MUL}, ${FXAA_REDUCE_MIN});
                let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
                dir = clamp(dir * rcpDirMin, vec2<f32>(-${FXAA_SPAN_MAX}), vec2<f32>(${FXAA_SPAN_MAX})) * inv;

                let rgbA = 0.5 * (
                    textureSampleLevel(sceneTex, samp, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
                    textureSampleLevel(sceneTex, samp, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
                let rgbB = rgbA * 0.5 + 0.25 * (
                    textureSampleLevel(sceneTex, samp, uv + dir * -0.5, 0.0).rgb +
                    textureSampleLevel(sceneTex, samp, uv + dir *  0.5, 0.0).rgb);
                let lB = luma(rgbB);
                if (lB < lMin || lB > lMax) { return vec4<f32>(rgbA, 1.0); }
                return vec4<f32>(rgbB, 1.0);
    ` : /* wgsl */`
                return textureSampleLevel(sceneTex, samp, uv, 0.0);
    `;

    const blitShader = device.createShaderModule({ code: /* wgsl */`
        struct BU { fullRes : vec2<f32>, _pad : vec2<f32> };
        @group(0) @binding(0) var<uniform> u        : BU;
        @group(0) @binding(1) var          sceneTex : texture_2d<f32>;
        @group(0) @binding(2) var          samp     : sampler;

        ${WGSL_FULLSCREEN_VS}

        fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.299, 0.587, 0.114)); }

        @fragment fn fs(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
            let uv = fragPos.xy / u.fullRes;   // swapchain-res fragment → [0,1)
            ${blitSampleBody}
        }
    ` });

    const blitPipeline = device.createRenderPipeline({
        layout:    device.createPipelineLayout({ bindGroupLayouts: [blitBGL] }),
        vertex:   { module: blitShader, entryPoint: "vs" },
        fragment: { module: blitShader, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
    });

    // ── Textures + bind groups (rebuilt on resize) ────────────────────────
    let tw = 0, th = 0;
    let hwDepthTex, rawTex, filtATex, filtBTex;
    let hwDV, rawV, filtAV, filtBV;
    let thickRawTex, thickATex, thickBTex;      // thickA/thickB: NRF ping-pong (mirror filtA/filtB)
    let thickRawV, thickAV, thickBV;
    let foamThickRawTex, foamThickRawV;
    let sceneColorTex, sceneColorV;   // final composited scene at RENDER_SCALE (blit source)
    // Filter bind groups: each reads a (depth, thickness) texture pair for the MRT NRF
    let filterBG_raw, filterBG_filtA, filterBG_filtB;
    let thickSmoothBG_B, thickSmoothBG_A;   // post-NRF thickness blur: B→A (H), A→B (V)
    let shadeBG;
    let diffuseBG;
    let foamThickBG, foamCompositeBG;
    let blitBG;

    function rebuildTextures(w, h) {
        if (tw === w && th === h) return;
        tw = w; th = h;
        hwDepthTex?.destroy(); rawTex?.destroy();
        filtATex?.destroy();   filtBTex?.destroy();
        thickRawTex?.destroy(); thickATex?.destroy(); thickBTex?.destroy();
        foamThickRawTex?.destroy();
        sceneColorTex?.destroy();

        const ra = GPUTextureUsage.RENDER_ATTACHMENT;
        const tb = GPUTextureUsage.TEXTURE_BINDING;

        // Fluid screen-space chain renders at FLUID_RES_SCALE of the swapchain res.
        // Render passes derive their viewport from the target size automatically, so
        // no explicit viewport call is needed — just size these textures accordingly.
        const dw = Math.ceil(w * FLUID_RES_SCALE), dh = Math.ceil(h * FLUID_RES_SCALE);

        hwDepthTex   = device.createTexture({ size: [dw, dh], format: "depth32float", usage: ra });
        rawTex       = device.createTexture({ size: [dw, dh], format: "r32float",     usage: ra | tb });
        filtATex     = device.createTexture({ size: [dw, dh], format: "r32float",     usage: ra | tb });
        filtBTex     = device.createTexture({ size: [dw, dh], format: "r32float",     usage: ra | tb });
        // r16float: filterable → supports additive blending; 2 bytes/px vs 8 bytes for rgba16float.
        // thickA/thickB are the NRF thickness ping-pong pair (MRT target1), mirroring filtA/filtB.
        thickRawTex = device.createTexture({ size: [dw, dh], format: "r16float", usage: ra | tb });
        thickATex   = device.createTexture({ size: [dw, dh], format: "r16float", usage: ra | tb });
        thickBTex   = device.createTexture({ size: [dw, dh], format: "r16float", usage: ra | tb });

        hwDV   = hwDepthTex.createView();
        rawV   = rawTex.createView();
        filtAV = filtATex.createView();
        filtBV = filtBTex.createView();
        thickRawV = thickRawTex.createView();
        thickAV   = thickATex.createView();
        thickBV   = thickBTex.createView();

        if (foamThickPipeline) {
            // Half-resolution foam thickness (r16float additive). Low-frequency signal,
            // so half-res + bilinear upsample in the composite is visually cheap.
            const fw = Math.ceil(w * FOAM_RES_SCALE), fh = Math.ceil(h * FOAM_RES_SCALE);
            foamThickRawTex = device.createTexture({ size: [fw, fh], format: "r16float", usage: ra | tb });
            foamThickRawV   = foamThickRawTex.createView();
        }

        // Offscreen scene color at RENDER_SCALE (swapchain format so the blit is a
        // straight resample). shade clears it, spray/foam-composite load over it, the
        // blit pass reads it via a filtering sampler.
        const sw = Math.ceil(w * RENDER_SCALE), sh = Math.ceil(h * RENDER_SCALE);
        sceneColorTex = device.createTexture({ size: [sw, sh], format, usage: ra | tb });
        sceneColorV   = sceneColorTex.createView();

        // Each NRF filter BG reads a (depth, thickness) pair.
        const mkFBG = (depthView, thickView) => device.createBindGroup({
            layout: filterBGL,
            entries: [
                { binding: 0, resource: { buffer: filterUniBuf } },
                { binding: 1, resource: depthView },
                { binding: 2, resource: thickView },
            ],
        });
        filterBG_raw   = mkFBG(rawV,   thickRawV);
        filterBG_filtA = mkFBG(filtAV, thickAV);
        filterBG_filtB = mkFBG(filtBV, thickBV);

        // Post-NRF thickness-only blur bind groups (reuse thickATex as scratch).
        thickSmoothBG_B = device.createBindGroup({ layout: thickSmoothBGL, entries: [{ binding: 0, resource: thickBV }] });
        thickSmoothBG_A = device.createBindGroup({ layout: thickSmoothBGL, entries: [{ binding: 0, resource: thickAV }] });

        // Shade reads filtBTex (NRF depth output) + thickBTex (NRF thickness output)
        shadeBG = device.createBindGroup({
            layout: shadeBGL,
            entries: [
                { binding: 0, resource: { buffer: particleUniBuf } },
                { binding: 1, resource: filtBV },
                { binding: 2, resource: thickBV },
                { binding: 3, resource: shadeSampler },
            ],
        });

        if (diffusePipeline) {
            diffuseBG = device.createBindGroup({
                layout: diffusePipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: particleUniBuf } },
                    { binding: 1, resource: filtBV },
                ],
            });
        }

        if (foamThickPipeline) {
            // Accumulate bind group depends on filtBV (rebuilt on resize).
            foamThickBG = device.createBindGroup({
                layout: foamThickBGL,
                entries: [
                    { binding: 0, resource: { buffer: particleUniBuf } },
                    { binding: 1, resource: filtBV },
                ],
            });
            foamCompositeBG = device.createBindGroup({
                layout: foamCompositeBGL,
                entries: [
                    { binding: 0, resource: { buffer: filterUniBuf } },  // screenRes for UV
                    { binding: 1, resource: foamThickRawV },
                    { binding: 2, resource: foamSampler },
                    { binding: 3, resource: filtBV },                    // fluid silhouette clip
                ],
            });
        }

        // Blit reads the RENDER_SCALE scene color (rebuilt on resize).
        blitBG = device.createBindGroup({
            layout: blitBGL,
            entries: [
                { binding: 0, resource: { buffer: blitUniBuf } },
                { binding: 1, resource: sceneColorV },
                { binding: 2, resource: blitSampler },
            ],
        });
    }

    // ── Render ────────────────────────────────────────────────────────────
    return function render(cmd, count, cv, view, proj, viewProj, tsQuery = null) {
        const colorTex = context.getCurrentTexture();
        rebuildTextures(colorTex.width, colorTex.height);

        // Particle uniforms
        const tanHalfFovY = 1.0 / proj[5];
        const aspect      = proj[5] / proj[0];
        particleUni.set(viewProj, 0); particleUni.set(view, 16); particleUni.set(proj, 32);
        particleUni[48] = cv.right[0]; particleUni[49] = cv.right[1]; particleUni[50] = cv.right[2];
        particleUni[51] = PARTICLE_RADIUS;
        particleUni[52] = cv.up[0];    particleUni[53] = cv.up[1];    particleUni[54] = cv.up[2];
        particleUni[55] = 0;
        particleUni[56] = tanHalfFovY; particleUni[57] = aspect; particleUni[58] = 0.1; particleUni[59] = 10000;
        // Scene (RENDER_SCALE) res — read by the shade pass, whose render target is
        // sceneColorTex, to map its fragment coords / NDC onto the FLUID_RES_SCALE
        // depth/thickness textures. [62,63] = _pad2.
        const sceneW = Math.ceil(colorTex.width  * RENDER_SCALE);
        const sceneH = Math.ceil(colorTex.height * RENDER_SCALE);
        particleUni[60] = sceneW; particleUni[61] = sceneH;
        device.queue.writeBuffer(particleUniBuf, 0, particleUni);

        // Filter uniforms (proj + screenRes; sigma/delta/mu are pre-written at init).
        // screenRes = scene (RENDER_SCALE) res: the NRF passes derive their kernel radius
        // from textureDimensions instead, but the foam COMPOSITE pass now renders into
        // sceneColorTex, so its UV screenRes must be the scene res.
        filterUni.set(proj, 0);
        filterUni[16] = sceneW; filterUni[17] = sceneH;
        device.queue.writeBuffer(filterUniBuf, 0, filterUni);

        // Blit uniforms: full swapchain res for the fullscreen-triangle UV.
        blitUni[0] = colorTex.width; blitUni[1] = colorTex.height;
        device.queue.writeBuffer(blitUniBuf, 0, blitUni);

        const colorV = colorTex.createView();
        const bgColor = { r: BG_DEPTH, g: 0, b: 0, a: 1 };

        // Helper: run a fullscreen MRT filter pass (target0 = depth, target1 = thickness).
        // tsWrites (DEBUG_PASS_TIMING): optional { querySet, beginningOfPassWriteIndex? ,
        // endOfPassWriteIndex? } to bracket the whole NRF H+V×iterations span as one "nrf" label.
        const runFilter = (pipeline, bg, depthDest, thickDest, tsWrites) => {
            const desc = {
                colorAttachments: [
                    { view: depthDest, clearValue: bgColor,                 loadOp: "clear", storeOp: "store" },
                    { view: thickDest, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
                ],
            };
            if (tsWrites) desc.timestampWrites = tsWrites; // DEBUG_PASS_TIMING
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.draw(3); pass.end();
        };

        // Pass 1: sphere impostor → raw linear depth
        {
            const desc = {
                colorAttachments: [{ view: rawV, clearValue: bgColor, loadOp: "clear", storeOp: "store" }],
                depthStencilAttachment: { view: hwDV, depthClearValue: 1.0, depthLoadOp: "clear", depthStoreOp: "store" },
            };
            // DEBUG_PASS_TIMING: our own querySet takes priority over the external
            // tsQuery on this pass — a pass can only have one timestampWrites target,
            // and the two querySets can't share a slot. Falls back to tsQuery (the
            // main.js render-total timer) when debug timing is off.
            if (dbgQuerySet) desc.timestampWrites = dbgTW('pass1');
            else if (tsQuery) desc.timestampWrites = { querySet: tsQuery.querySet, beginningOfPassWriteIndex: tsQuery.beginIndex };
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(particlePipeline);
            pass.setBindGroup(0, particleBG);
            pass.setVertexBuffer(0, particlePosBuffer);
            pass.setVertexBuffer(1, particleVelBuffer);
            pass.draw(3, count);   // 3-vert circumscribing triangle (see particleShader vs)
            pass.end();
        }

        // Pass T1: thickness accumulation (all particles, no depth test, additive blend)
        {
            const desc = {
                colorAttachments: [{ view: thickRawV, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
            };
            if (dbgQuerySet) desc.timestampWrites = dbgTW('thickT1'); // DEBUG_PASS_TIMING
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(thicknessPipeline);
            pass.setBindGroup(0, thickParticleBG);
            pass.setVertexBuffer(0, particlePosBuffer);
            pass.setVertexBuffer(1, particleVelBuffer);
            pass.draw(3, count);   // 3-vert circumscribing triangle (shared particleShader vs)
            pass.end();
        }

        // NRF iterations (MRT depth+thickness): each is H (raw/prevB → filtA/thickA)
        // + V (filtA/thickA → filtB/thickB). Ends at (filtBTex, thickBTex).
        // Iteration 1: source is raw
        // DEBUG_PASS_TIMING: bracket the entire NRF H+V×iterations span as one "nrf"
        // pass — begin on the very first H pass, end on the very last V pass.
        runFilter(filterHPipeline, filterBG_raw,   filtAV, thickAV, dbgTWBegin('nrf'));
        runFilter(filterVPipeline, filterBG_filtA, filtBV, thickBV, NRF_ITERATIONS === 1 ? dbgTWEnd('nrf') : undefined);
        // Additional iterations ping-pong between filtB → filtA → filtB
        for (let i = 1; i < NRF_ITERATIONS; i++) {
            runFilter(filterHPipeline, filterBG_filtB, filtAV, thickAV);
            runFilter(filterVPipeline, filterBG_filtA, filtBV, thickBV, i === NRF_ITERATIONS - 1 ? dbgTWEnd('nrf') : undefined);
        }

        // Pass ThB: thickness-only fixed-radius Gaussian blur (grid-frequency thickness
        // mottling cleanup, independent of NRF_SIGMA). thickB → thickA (H) → thickB (V);
        // thickATex is free scratch space here since the NRF chain above has finished.
        {
            const passH = cmd.beginRenderPass({
                colorAttachments: [{ view: thickAV, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
            });
            passH.setPipeline(thickSmoothHPipeline); passH.setBindGroup(0, thickSmoothBG_B); passH.draw(3); passH.end();

            const passV = cmd.beginRenderPass({
                colorAttachments: [{ view: thickBV, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
            });
            passV.setPipeline(thickSmoothVPipeline); passV.setBindGroup(0, thickSmoothBG_A); passV.draw(3); passV.end();
        }

        // Pass Nf: foam/bubble thickness accumulation (additive, reads filtBTex for
        // depth attenuation). Must run before shade so it can sample the fluid depth.
        if (foamThickPipeline) {
            const desc = {
                colorAttachments: [{ view: foamThickRawV, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
            };
            if (dbgQuerySet) desc.timestampWrites = dbgTW('foamNf'); // DEBUG_PASS_TIMING
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(foamThickPipeline);
            pass.setBindGroup(0, foamThickBG);
            pass.setVertexBuffer(0, diffuse.particleBuffer);
            pass.draw(6, diffuse.maxCount);
            pass.end();
        }

        // Shade pass: normal reconstruct → sceneColorTex (RENDER_SCALE). The final
        // upscale to the swapchain happens in the blit pass below.
        {
            const desc = {
                colorAttachments: [{ view: sceneColorV, clearValue: { r:0.08,g:0.08,b:0.08,a:1 }, loadOp: "clear", storeOp: "store" }],
            };
            if (dbgQuerySet) desc.timestampWrites = dbgTW('shade'); // DEBUG_PASS_TIMING
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(shadePipeline); pass.setBindGroup(0, shadeBG); pass.draw(3); pass.end();
        }

        // Diffuse particles: composited on top of the shaded fluid. Always draws
        // diffuse.maxCount instances — dead pool slots are NaN'd out in the vertex
        // shader (see diffuse-gpu.js for why the pool never shrinks the draw count).
        if (diffusePipeline) {
            const desc = {
                colorAttachments: [{ view: sceneColorV, loadOp: "load", storeOp: "store" }],
            };
            if (dbgQuerySet) desc.timestampWrites = dbgTW('spray'); // DEBUG_PASS_TIMING
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(diffusePipeline);
            pass.setBindGroup(0, diffuseBG);
            pass.setVertexBuffer(0, diffuse.particleBuffer);
            pass.draw(6, diffuse.maxCount);
            pass.end();
        }

        // Pass Cf: composite accumulated foam over the shaded fluid + spray.
        if (foamCompositePipeline) {
            const desc = {
                colorAttachments: [{ view: sceneColorV, loadOp: "load", storeOp: "store" }],
            };
            if (dbgQuerySet) desc.timestampWrites = dbgTW('foamCf'); // DEBUG_PASS_TIMING
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(foamCompositePipeline);
            pass.setBindGroup(0, foamCompositeBG);
            pass.draw(3);
            pass.end();
        }

        // Pass Bl: upscale the RENDER_SCALE scene color → swapchain (bilinear + FXAA).
        // This is the only pass that writes the swapchain, so the external tsQuery's
        // end-of-render marker lands here (was on shade before the offscreen scene tex).
        {
            const desc = {
                colorAttachments: [{ view: colorV, clearValue: { r:0,g:0,b:0,a:1 }, loadOp: "clear", storeOp: "store" }],
            };
            if (dbgQuerySet) desc.timestampWrites = dbgTW('blit');
            else if (tsQuery) desc.timestampWrites = { querySet: tsQuery.querySet, endOfPassWriteIndex: tsQuery.endIndex };
            const pass = cmd.beginRenderPass(desc);
            pass.setPipeline(blitPipeline); pass.setBindGroup(0, blitBG); pass.draw(3); pass.end();
        }

        // DEBUG_PASS_TIMING: resolve this frame's queries + self-throttled async
        // readback (skip if the previous frame's mapAsync hasn't resolved yet — same
        // "self-throttling" pattern as main.js's tsRes/tsPending). Fully self-contained
        // here: resolveQuerySet/copyBufferToBuffer are just encoded onto `cmd`, which
        // main.js finishes and submits right after this function returns; mapAsync
        // itself is deferred to a microtask so it only runs after that submit.
        if (dbgQuerySet && !dbgPending) {
            const n = DBG_PASS_LABELS.length * 2;
            cmd.resolveQuerySet(dbgQuerySet, 0, n, dbgResolveBuf, 0);
            cmd.copyBufferToBuffer(dbgResolveBuf, 0, dbgReadBuf, 0, n * 8);
            dbgPending = true;
            queueMicrotask(() => {
                dbgReadBuf.mapAsync(GPUMapMode.READ).then(() => {
                    const ts = new BigUint64Array(dbgReadBuf.getMappedRange());
                    for (let i = 0; i < DBG_PASS_LABELS.length; i++) {
                        const b = ts[i * 2], e = ts[i * 2 + 1];
                        const ms = e > b ? Number(e - b) / 1e6 : 0;
                        dbgEmaMs[i] = dbgEmaMs[i] * 0.9 + ms * 0.1;
                    }
                    dbgReadBuf.unmap();
                    dbgPending = false;

                    dbgFrameCounter++;
                    if (dbgFrameCounter >= 60) {
                        dbgFrameCounter = 0;
                        const rows = {};
                        let total = 0;
                        for (let i = 0; i < DBG_PASS_LABELS.length; i++) {
                            rows[DBG_PASS_LABELS[i]] = { ms: dbgEmaMs[i].toFixed(3) };
                            total += dbgEmaMs[i];
                        }
                        rows['TOTAL'] = { ms: total.toFixed(3) };
                        if (console.table) console.table(rows); else console.log(rows);
                    }
                }).catch(() => { dbgPending = false; });
            });
        }
    };
}
