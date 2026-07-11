import { isMobile, dpr, DIFFUSE_MAX_COUNT } from './config.js';
import { FluidGPU } from './fluid-gpu.js';
import { DiffuseGPU } from './diffuse-gpu.js';
import { initFluidRenderer } from './fluid-renderer.js';
import {
    mat4Perspective, mat4LookAt, mat4Multiply,
    cameraVectors, screenToWorld, worldToScreen, simBoundsOnScreen, rayBoxIntersect,
} from './math.js';

// ─────────────────────────────────────────────────────────────
//  Particle initialisation
// ─────────────────────────────────────────────────────────────
// Fixed 1.8:2:1.8 domain — independent of screen orientation so portrait phones get the same grid size as landscape
// Capacity is sized above the initial fillBlock() count (~100034 desktop / ~30213 mobile)
// so there is headroom left for real-time spawnParticles() additions.
const fluid = new FluidGPU(2, 2, 3, isMobile ? 0.02 : 0.0125, isMobile ? 80000 : 200000);
fluid.fillBlock(0.05, 0.05, 0.05, 0.95, 0.95, 0.45);
const diffuse = new DiffuseGPU(DIFFUSE_MAX_COUNT);

// ─────────────────────────────────────────────────────────────
//  Canvas / overlay setup
// ─────────────────────────────────────────────────────────────
const c       = document.querySelector("#gpu");
const overlay = document.querySelector("#overlay");
const octx    = overlay.getContext("2d");
overlay.style.touchAction = "none";

let logicalW = window.innerWidth, logicalH = window.innerHeight;

function resizeCanvases() {
    logicalW = window.innerWidth;
    logicalH = window.innerHeight;
    c.width = overlay.width  = Math.round(logicalW * dpr);
    c.height = overlay.height = Math.round(logicalH * dpr);
    // Scale the 2D context so all overlay drawing uses CSS-pixel coordinates.
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
resizeCanvases();
window.addEventListener("resize", resizeCanvases);

// ─────────────────────────────────────────────────────────────
//  Orbit camera + interaction state
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

const handState = {
    pos: [0, 0, 0], vel: [0, 0, 0],
    radius: 6.0, strength: 1, active: false,
};

// Shared camera-frame computation — used by both the pointerdown handler (to
// project the sim bounds for the mobile orbit-vs-push heuristic) and the main loop
// (for rendering/aiming). Writes into persistent scratch matrices/objects instead of
// allocating fresh ones each call (mat4*/cameraVectors default to a fresh allocation
// only when no `out` buffer is passed — see math.js). Safe to share across call sites
// since every caller only reads the result synchronously within the same call, never
// retaining the returned object/arrays past that point.
const _camView = new Float32Array(16), _camProj = new Float32Array(16), _camViewProj = new Float32Array(16);
const _camCv = { eye: new Float32Array(3), right: new Float32Array(3), up: new Float32Array(3) };
const _camFrame = { cv: _camCv, fovY: 0, view: _camView, proj: _camProj, viewProj: _camViewProj };
const WORLD_UP = [0, 1, 0];
function computeCameraFrame(aspect) {
    cameraVectors(camera, _camCv);
    _camFrame.fovY = 2 * Math.atan(Math.tan(camera.fovy / 2) / Math.min(aspect, 1));
    mat4LookAt(_camCv.eye, camera.target, WORLD_UP, _camView);
    mat4Perspective(_camFrame.fovY, aspect, camera.near, camera.far, _camProj);
    mat4Multiply(_camProj, _camView, _camViewProj);
    return _camFrame;
}

// Real-time particle add/remove: held while the E / R keys are down, applied
// at the current pointer position (unprojected onto the cam.target plane every
// frame, independent of whether a drag is active).
const spawnState  = { active: false, radius: 2.5, rate: 4000, accum: 0 }; // rate = particles/sec
const deleteState = { active: false, radius: 5.0 };
let lastPointerX = window.innerWidth / 2, lastPointerY = window.innerHeight / 2;

// Tracks raw physical key state; spawnState.active/deleteState.active are derived
// from this every frame (see loop()) rather than set directly here. Deriving fresh
// each frame — instead of latching on keydown — means releasing whichever key was
// blocking the other immediately re-arms it, with no "stuck until release+re-press"
// gap (keydown alone can't do this because held-key repeats are suppressed below,
// so a key blocked at press time would never get another chance to activate).
let eKeyDown = false, rKeyDown = false;
window.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    if (e.key === "e" || e.key === "E") eKeyDown = true;
    if (e.key === "r" || e.key === "R") rKeyDown = true;
});
window.addEventListener("keyup", (e) => {
    if (e.key === "e" || e.key === "E") eKeyDown = false;
    if (e.key === "r" || e.key === "R") rKeyDown = false;
});
// Dragging elsewhere (e.g. orbit) can eat the keyup; drop both modes on blur too.
window.addEventListener("blur", () => { eKeyDown = false; rKeyDown = false; });

// ─────────────────────────────────────────────────────────────
//  Pointer events
// ─────────────────────────────────────────────────────────────
{
    const orbit = { active: false, pointerId: null, lastX: 0, lastY: 0 };
    const iact  = { active: false, pointerId: null };

    overlay.addEventListener("contextmenu", (e) => e.preventDefault());
    // iOS Safari fires gesturestart/gesturechange for pinch-zoom regardless of touch-action;
    // block them so the fixed-position canvas can't get scaled by an OS-level page zoom.
    document.addEventListener("gesturestart", (e) => e.preventDefault());
    document.addEventListener("gesturechange", (e) => e.preventDefault());

    overlay.addEventListener("pointerdown", (e) => {
        const aspect  = c.width / c.height;
        const { cv, viewProj: vp } = computeCameraFrame(aspect);

        // On mobile, tapping outside the projected simulation box triggers orbit instead of push.
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
            orbit.active = true; orbit.pointerId = e.pointerId;
            orbit.lastX = e.clientX; orbit.lastY = e.clientY;
            overlay.setPointerCapture(e.pointerId);
            e.preventDefault();
        } else {
            iact.active = true; iact.pointerId = e.pointerId;
            const p0 = screenToWorld(e.offsetX, e.offsetY, logicalW, logicalH, camera, cv);
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
        lastPointerX = e.offsetX; lastPointerY = e.offsetY;
        if (orbit.active && e.pointerId === orbit.pointerId) {
            const dx = e.clientX - orbit.lastX, dy = e.clientY - orbit.lastY;
            orbit.lastX = e.clientX; orbit.lastY = e.clientY;
            camera.theta -= dx * 0.01;
            camera.phi = Math.max(-(Math.PI / 2 - 0.01), Math.min(Math.PI / 2 - 0.01, camera.phi + dy * 0.01));
        }
        if (iact.active && e.pointerId === iact.pointerId) {
            // Recompute absolute 3D position — never accumulate drift.
            const cv = cameraVectors(camera);
            const np = screenToWorld(e.offsetX, e.offsetY, logicalW, logicalH, camera, cv);
            const vs = 2.0;
            handState.vel[0] = (np[0] - handState.pos[0]) * vs;
            handState.vel[1] = (np[1] - handState.pos[1]) * vs;
            handState.vel[2] = (np[2] - handState.pos[2]) * vs;
            handState.pos = np;
        }
    });

    const endPointer = (e) => {
        if (e.pointerId === iact.pointerId) {
            iact.active = false; iact.pointerId = null;
            handState.active = false; handState.vel[0] = handState.vel[1] = handState.vel[2] = 0;
        }
        if (e.pointerId === orbit.pointerId) { orbit.active = false; orbit.pointerId = null; }
    };
    overlay.addEventListener("pointerup",     endPointer);
    overlay.addEventListener("pointercancel", endPointer);

    overlay.addEventListener("wheel", (e) => {
        e.preventDefault();
        camera.radius = Math.max(1, Math.min(100000, camera.radius * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
}

// ─────────────────────────────────────────────────────────────
//  Main loop
// ─────────────────────────────────────────────────────────────
let avgFrameMs = 0, lastFrameTime = performance.now();
let _animFrameId = null;
let gpuSimMs = 0, gpuRenderMs = 0, gpuTimersActive = false;
// Reused each frame — read into ddx/ddy/ddz immediately below, never retained.
const _cursorPos = [0, 0, 0];
// diffuse.aliveCount only feeds the on-screen debug text, so its GPU readback
// (copyBufferToBuffer + mapAsync) doesn't need to run every single frame.
const DEBUG_READBACK_INTERVAL = 15;
let debugFrameCounter = 0;

function drawOverlay(cv, viewProj, frameMs) {
    octx.clearRect(0, 0, logicalW, logicalH);

    if (handState.active) {
        const sc = worldToScreen(handState.pos, viewProj, logicalW, logicalH);
        if (sc) {
            const dx = handState.pos[0] - cv.eye[0];
            const dy = handState.pos[1] - cv.eye[1];
            const dz = handState.pos[2] - cv.eye[2];
            const dist  = Math.hypot(dx, dy, dz) || 1;
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

    if (spawnState.active || deleteState.active) {
        // For add, show the actual resolved aim point (fluid surface hit) when known —
        // it can differ from the cam.target-plane projection since raycastFluid()
        // follows the whole view ray. Delete has no single hit point (it's a
        // cylinder), so it always shows the plane projection as an aim approximation.
        const markerPos = (spawnState.active && fluid.lastRayHit)
            ? fluid.lastRayHit
            : screenToWorld(lastPointerX, lastPointerY, logicalW, logicalH, camera, cv);
        const sc = worldToScreen(markerPos, viewProj, logicalW, logicalH);
        if (sc) {
            const dx = markerPos[0] - cv.eye[0], dy = markerPos[1] - cv.eye[1], dz = markerPos[2] - cv.eye[2];
            const dist  = Math.hypot(dx, dy, dz) || 1;
            const focal = 1 / Math.tan(camera.fovy / 2);
            const radius = spawnState.active ? spawnState.radius : deleteState.radius;
            const sr = Math.max(4, radius / dist * focal * (logicalH / 2));
            octx.beginPath();
            octx.arc(sc[0], sc[1], sr, 0, Math.PI * 2);
            octx.strokeStyle = spawnState.active ? "rgba(120,255,150,0.8)" : "rgba(255,100,100,0.8)";
            octx.lineWidth = 2;
            octx.stroke();
        }
    }

    octx.font = "14px monospace";
    octx.fillStyle = "#aaa";
    octx.fillText(`frame: ${frameMs.toFixed(2)} ms`, 16, 28);
    octx.fillText(`particles: ${fluid.active_particle_num}  diffuse: ${diffuse.aliveCount}`, 16, 46);
    if (gpuTimersActive) {
        octx.fillText(`GPU sim: ${gpuSimMs.toFixed(2)} ms  render: ${gpuRenderMs.toFixed(2)} ms`, 16, 64);
        octx.fillText(isMobile
            ? `drag in sim: push  drag outside: orbit`
            : `L-drag: push fluid  R-drag: orbit  wheel: zoom  E: add  R: remove`, 16, 82);
    } else {
        octx.fillText(isMobile
            ? `drag in sim: push  drag outside: orbit`
            : `L-drag: push fluid  R-drag: orbit  wheel: zoom  E: add  R: remove`, 16, 64);
    }
}

function startLoop(encodeRender, tsRes) {
    const simTsq = tsRes ? { querySet: tsRes.querySet, beginIndex: 0, endIndex: 1 } : null;
    const renTsq = tsRes ? { querySet: tsRes.querySet, beginIndex: 2, endIndex: 3 } : null;
    let tsPending = false;
    gpuTimersActive = !!tsRes;

    // Fixed-timestep accumulator: run exactly enough simFrames per real second.
    // SIM_STEP_S = wall-clock interval for one simFrame (1/60 → 60 steps/sec at 60fps).
    // MAX_SIM_STEPS caps runaway when frames are slow (spiral-of-death prevention).
    const SIM_STEP_S   = 1 / 60;
    const MAX_SIM_STEPS = 3;
    let simAccum = 0;

    function loop(timestamp) {
        const aspect     = c.width / c.height;
        const frameStart = timestamp;

        const deltaS = Math.min((frameStart - lastFrameTime) / 1000, 0.1);
        simAccum += deltaS;

        // Mutually exclusive, re-derived every frame from physical key state so
        // releasing whichever key was blocking the other re-arms it immediately.
        spawnState.active  = eKeyDown && !rKeyDown;
        deleteState.active = rKeyDown && !eKeyDown;

        // Use the shorter screen dimension as the FOV reference so the
        // simulation never clips horizontally on portrait screens.
        const { cv, view, proj, viewProj } = computeCameraFrame(aspect);
        fluid.updateHand(handState.pos, handState.vel, handState.radius, handState.strength, handState.active, cv.eye);
        handState.vel[0] = handState.vel[1] = handState.vel[2] = 0;

        const cmd = fluid.device.createCommandEncoder();

        // E/R aim along the same view ray the cursor projects to (origin=eye, through
        // the cam.target-plane point) rather than a single point at a fixed depth —
        // this is depth-independent, like APPLY_HAND's push, and stays correct
        // regardless of where the fluid surface actually sits along that ray.
        if (spawnState.active || deleteState.active) {
            const cursorPos = screenToWorld(lastPointerX, lastPointerY, logicalW, logicalH, camera, cv, _cursorPos);
            const ddx = cursorPos[0] - cv.eye[0], ddy = cursorPos[1] - cv.eye[1], ddz = cursorPos[2] - cv.eye[2];
            const dl  = Math.hypot(ddx, ddy, ddz) || 1;
            const dir = [ddx / dl, ddy / dl, ddz / dl];

            if (deleteState.active) {
                fluid.deleteNear(cmd, cv.eye, dir, deleteState.radius);
            }
            if (spawnState.active) {
                fluid.raycastFluid(cmd, cv.eye, dir, spawnState.radius);
                spawnState.accum = Math.min(spawnState.accum + spawnState.rate * deltaS, spawnState.rate);
                const n = Math.floor(spawnState.accum);
                if (n > 0) {
                    // Aim at the existing fluid surface (last raycast hit); if the ray
                    // currently passes through empty space, fall back to where it
                    // enters the domain so a new blob can still be seeded there.
                    let target = fluid.lastRayHit;
                    if (!target) {
                        const t = rayBoxIntersect(cv.eye, dir,
                            [fluid.HARD_MIN, fluid.HARD_MIN, fluid.HARD_MIN],
                            [fluid.HARD_MAX_X, fluid.HARD_MAX_Y, fluid.HARD_MAX_Z]);
                        target = t !== null
                            ? [cv.eye[0] + t * dir[0], cv.eye[1] + t * dir[1], cv.eye[2] + t * dir[2]]
                            : null;
                    }
                    if (target) {
                        const spawned = fluid.spawnParticles(target[0], target[1], target[2], n, spawnState.radius);
                        spawnState.accum -= spawned;
                    }
                }
            }
        }

        let simSteps = 0;
        while (simAccum >= SIM_STEP_S && simSteps < MAX_SIM_STEPS) {
            fluid.simFrame(cmd, simSteps === 0 ? simTsq : null);
            simAccum -= SIM_STEP_S;
            simSteps++;
        }
        // Safety valve: if real frame cost stays above SIM_STEP_S*MAX_SIM_STEPS for a
        // while (e.g. a diffuse-particle-heavy transient making every frame expensive),
        // the loop above can never fully drain simAccum, and it would otherwise keep
        // growing every frame without bound. Drop the un-simulated backlog instead of
        // queuing it up forever — once real frame cost recovers, this avoids also
        // paying for a pile of extra catch-up steps on top of the recovery.
        const MAX_SIM_ACCUM = SIM_STEP_S * MAX_SIM_STEPS;
        if (simAccum > MAX_SIM_ACCUM) simAccum = MAX_SIM_ACCUM;
        encodeRender(cmd, fluid.active_particle_num, cv, view, proj, viewProj, renTsq);

        if (tsRes && !tsPending) {
            cmd.resolveQuerySet(tsRes.querySet, 0, 4, tsRes.resolveBuffer, 0);
            cmd.copyBufferToBuffer(tsRes.resolveBuffer, 0, tsRes.readBuffer, 0, 32);
        }

        debugFrameCounter++;
        if (debugFrameCounter >= DEBUG_READBACK_INTERVAL) {
            debugFrameCounter = 0;
            diffuse.requestDebugReadback(cmd);
        }

        fluid.device.queue.submit([cmd.finish()]);
        fluid.pollDelete();
        fluid.pollRaycast();
        diffuse.pollDebug();

        if (tsRes && !tsPending) {
            tsPending = true;
            tsRes.readBuffer.mapAsync(GPUMapMode.READ).then(() => {
                const ts = new BigUint64Array(tsRes.readBuffer.getMappedRange());
                const simNs = ts[1] > ts[0] ? Number(ts[1] - ts[0]) : 0;
                const renNs = ts[3] > ts[2] ? Number(ts[3] - ts[2]) : 0;
                gpuSimMs    = gpuSimMs    * 0.9 + simNs / 1e6 * 0.1;
                gpuRenderMs = gpuRenderMs * 0.9 + renNs / 1e6 * 0.1;
                tsRes.readBuffer.unmap();
                tsPending = false;
            });
        }

        const frameMs = frameStart - lastFrameTime;
        lastFrameTime = frameStart;
        avgFrameMs = avgFrameMs * 0.9 + frameMs * 0.1;

        drawOverlay(cv, viewProj, avgFrameMs);
        _animFrameId = requestAnimationFrame(loop);
    }
    lastFrameTime = performance.now();
    _animFrameId = requestAnimationFrame(loop);
}

// ─────────────────────────────────────────────────────────────
//  Entry point
// ─────────────────────────────────────────────────────────────
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
    const supportsTs = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
        requiredFeatures: supportsTs ? ['timestamp-query'] : [],
    });

    device.lost.then(async (info) => {
        if (_animFrameId !== null) { cancelAnimationFrame(_animFrameId); _animFrameId = null; }
        if (info.reason !== "destroyed") {
            await new Promise(r => setTimeout(r, 1000));
            init().catch(showError);
        }
    });

    fluid.initGPU(device);
    diffuse.initGPU(device, fluid);
    fluid.diffuse = diffuse;

    const tsRes = supportsTs ? {
        querySet:      device.createQuerySet({ type: 'timestamp', count: 4 }),
        resolveBuffer: device.createBuffer({ size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }),
        readBuffer:    device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST    | GPUBufferUsage.MAP_READ }),
    } : null;

    const fr = await initFluidRenderer(device, c, fluid.particlePosBuffer, fluid.particleVelBuffer, fluid.particleDensityBuffer, fluid.REST_DENSITY, diffuse);
    startLoop((cmd, count, cv, view, proj, viewProj, renTsq) => fr(cmd, count, cv, view, proj, viewProj, renTsq), tsRes);
}

init().catch(showError);
