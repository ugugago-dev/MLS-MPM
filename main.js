import { isMobile, dpr, DIFFUSE_MAX_COUNT, urlNum } from './config.js';
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
// Diagnostic override (device-lost/TDR triage on mobile — see CLAUDE.md):
// ?p=<count> forces the particle count regardless of isMobile.
const particleCount = isMobile ? 80000 : 400000;

const fluid = new FluidGPU(2, 2, 3, isMobile ? 0.0175 : 0.0125, particleCount);
const spacing = 0.03;
fluid.fillBlock(spacing, spacing, spacing, 1.0 - spacing, 1.0 - spacing, 0.4 - spacing);
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

// ─────────────────────────────────────────────────────────────
//  Movable wall (wall 0 only — W toggles it, Shift+L-drag moves it)
// ─────────────────────────────────────────────────────────────
// setWall()/removeWall() write straight to a GPU buffer (no command encoder involved,
// same pattern as fluid.updateHand() below), so they're safe to call directly from
// input handlers rather than deferring to the sim loop. Guard on fluid.device so a
// stray keypress/drag before init() finishes (GPU not ready yet, fluid.walls may not
// even be populated) is a no-op instead of a crash.
const wall0Min = new Float32Array(3), wall0Max = new Float32Array(3);
function toggleWall0() {
    if (!fluid.device) return;
    if (fluid.walls[0].active) {
        fluid.removeWall(0);
        return;
    }
    // Initial shape: a slab perpendicular to Z, 3 cells thick, straddling the grid
    // center; spans the full X depth and most of the height (floor margin to 85%).
    // Z-perpendicular because the default camera (theta=π/2) looks down the X axis:
    // screen-horizontal = Z, so Shift+drag sweeps the wall sideways through the tank
    // (an X-perpendicular slab would face the camera full-frame and its only visible
    // travel direction, Z, is already spanned — the clamp would pin it in place).
    const czc = fluid.grid_Z_num / 2;
    wall0Min[0] = fluid.HARD_MIN;   wall0Max[0] = fluid.HARD_MAX_X;
    wall0Min[1] = fluid.HARD_MIN;   wall0Max[1] = fluid.grid_Y_num * 0.85;
    wall0Min[2] = czc - 1.5;        wall0Max[2] = czc + 1.5;
    fluid.setWall(0, wall0Min, wall0Max);
}

// Drag state for moving wall 0: the plane is fixed at drag start (view-direction normal
// through the wall's center at that moment), and every subsequent pointer move
// re-intersects the pointer ray with that same plane to get a world-space delta —
// added to the wall's start AABB (position only; size never changes during a drag).
// All scratch arrays below are pre-allocated and reused every pointerdown/move, per the
// project's no-per-frame/per-event-allocation convention (see computeCameraFrame above).
const wallDrag = {
    active: false, pointerId: null,
    forward: new Float32Array(3), planePoint: new Float32Array(3),
    startHit: new Float32Array(3), startMin: new Float32Array(3), startMax: new Float32Array(3),
};
const _wallRayPt  = [0, 0, 0];       // scratch for screenToWorld() during wall drag
const _wallRayDir = new Float32Array(3);
const _wallHit    = new Float32Array(3);
const _wallNewMin = new Float32Array(3), _wallNewMax = new Float32Array(3);
const _wallHardMax = new Float32Array(3); // [HARD_MAX_X, HARD_MAX_Y, HARD_MAX_Z], filled once below
_wallHardMax[0] = fluid.HARD_MAX_X; _wallHardMax[1] = fluid.HARD_MAX_Y; _wallHardMax[2] = fluid.HARD_MAX_Z;

// Ray/plane intersection: eye + t*dir where the plane passes through `point` with
// normal `forward`. Writes into `out`; returns false (out untouched) if the ray runs
// parallel to the plane (near-impossible for a view-aligned plane facing the camera).
function rayPlaneIntersect(eye, dir, point, forward, out) {
    const denom = dir[0] * forward[0] + dir[1] * forward[1] + dir[2] * forward[2];
    if (Math.abs(denom) < 1e-6) return false;
    const t = ((point[0] - eye[0]) * forward[0] + (point[1] - eye[1]) * forward[1] + (point[2] - eye[2]) * forward[2]) / denom;
    out[0] = eye[0] + t * dir[0]; out[1] = eye[1] + t * dir[1]; out[2] = eye[2] + t * dir[2];
    return true;
}

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
    if (e.key === "w" || e.key === "W") toggleWall0();
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

        // Shift+left-drag moves wall 0 when it's active — checked first so this branch
        // preempts both the push (iact) and orbit handling below; falls through to the
        // normal push if the wall isn't active (per spec: no active wall = normal drag).
        if (e.button === 0 && e.shiftKey && fluid.walls[0].active) {
            const fx = camera.target[0] - cv.eye[0], fy = camera.target[1] - cv.eye[1], fz = camera.target[2] - cv.eye[2];
            const fl = Math.hypot(fx, fy, fz) || 1;
            wallDrag.forward[0] = fx / fl; wallDrag.forward[1] = fy / fl; wallDrag.forward[2] = fz / fl;

            const w = fluid.walls[0];
            wallDrag.planePoint[0] = (w.min[0] + w.max[0]) * 0.5;
            wallDrag.planePoint[1] = (w.min[1] + w.max[1]) * 0.5;
            wallDrag.planePoint[2] = (w.min[2] + w.max[2]) * 0.5;

            const p0 = screenToWorld(e.offsetX, e.offsetY, logicalW, logicalH, camera, cv, _wallRayPt);
            let dx = p0[0] - cv.eye[0], dy = p0[1] - cv.eye[1], dz = p0[2] - cv.eye[2];
            const dl = Math.hypot(dx, dy, dz) || 1;
            _wallRayDir[0] = dx / dl; _wallRayDir[1] = dy / dl; _wallRayDir[2] = dz / dl;

            if (rayPlaneIntersect(cv.eye, _wallRayDir, wallDrag.planePoint, wallDrag.forward, wallDrag.startHit)) {
                wallDrag.startMin.set(w.min); wallDrag.startMax.set(w.max);
                wallDrag.active = true; wallDrag.pointerId = e.pointerId;
                overlay.setPointerCapture(e.pointerId);
                e.preventDefault();
                return;
            }
            // Degenerate ray (parallel to the drag plane) — bail out to normal handling
            // below rather than leaving the pointer captured with nothing to drag.
        }

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
        if (wallDrag.active && e.pointerId === wallDrag.pointerId) {
            const cv = cameraVectors(camera, _camCv);
            const p1 = screenToWorld(e.offsetX, e.offsetY, logicalW, logicalH, camera, cv, _wallRayPt);
            let dx = p1[0] - cv.eye[0], dy = p1[1] - cv.eye[1], dz = p1[2] - cv.eye[2];
            const dl = Math.hypot(dx, dy, dz) || 1;
            _wallRayDir[0] = dx / dl; _wallRayDir[1] = dy / dl; _wallRayDir[2] = dz / dl;

            if (rayPlaneIntersect(cv.eye, _wallRayDir, wallDrag.planePoint, wallDrag.forward, _wallHit)) {
                // World-space offset since drag start, clamped per axis so the moved
                // AABB stays within the hard sim bounds (size fixed — position only).
                for (let i = 0; i < 3; i++) {
                    let delta = _wallHit[i] - wallDrag.startHit[i];
                    const lo = fluid.HARD_MIN - wallDrag.startMin[i];
                    const hi = _wallHardMax[i] - wallDrag.startMax[i];
                    if (delta < lo) delta = lo; else if (delta > hi) delta = hi;
                    _wallNewMin[i] = wallDrag.startMin[i] + delta;
                    _wallNewMax[i] = wallDrag.startMax[i] + delta;
                }
                fluid.setWall(0, _wallNewMin, _wallNewMax);
            }
            return;
        }
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
        if (e.pointerId === wallDrag.pointerId) { wallDrag.active = false; wallDrag.pointerId = null; }
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
// Reused each frame — read into ddx/ddy/ddz immediately below, never retained.
const _cursorPos = [0, 0, 0];

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
    let ln = 0;
    const nextY = () => 28 + 18 * ln++;
    octx.fillText(`frame: ${frameMs.toFixed(2)} ms`, 16, nextY());
    octx.fillText(isMobile
        ? `drag in sim: push  drag outside: orbit  W: toggle wall`
        : `L-drag: push fluid  R-drag: orbit  wheel: zoom  E: add  R: remove  W: toggle wall  Shift+L-drag: move wall`, 16, nextY());
}

function startLoop(encodeRender) {
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
            fluid.simFrame(cmd);
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
        encodeRender(cmd, fluid.active_particle_num, cv, view, proj, viewProj);

        fluid.device.queue.submit([cmd.finish()]);
        fluid.pollDelete();
        fluid.pollRaycast();

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
// On-screen error overlay. Phones give us no console, so surface everything
// (thrown exceptions, rejected promises, WebGPU uncaptured errors) here as
// word-wrapped, full-screen readable text. `sticky` errors latch so a later
// cascade of secondary failures can't scroll the root cause off screen; a
// non-sticky message (e.g. "device lost, retrying") can still be overwritten.
let _errorSticky = false;
function _formatError(err) {
    if (err == null) return "Unknown error (null)";
    if (typeof err === "string") return err;
    if (err instanceof Error) {
        return (err.name ? err.name + ": " : "") + err.message +
               (err.stack ? "\n\n" + err.stack : "");
    }
    if (err.message) return String(err.message);
    try { return JSON.stringify(err); } catch { return String(err); }
}
function showError(err, sticky = true) {
    console.error(err);
    if (_errorSticky) return;          // root cause already latched
    if (sticky) _errorSticky = true;
    if (sticky && _animFrameId !== null) { cancelAnimationFrame(_animFrameId); _animFrameId = null; }

    const msg = "WebGPU / runtime error\n\n" + _formatError(err);
    const pad = 12, lineH = 18, maxW = Math.max(40, logicalW - pad * 2);
    octx.setTransform(dpr, 0, 0, dpr, 0, 0);   // restore CSS-pixel space (loop may have changed nothing, but be safe)
    octx.clearRect(0, 0, logicalW, logicalH);
    octx.fillStyle = "rgba(25,0,0,0.94)";
    octx.fillRect(0, 0, logicalW, logicalH);
    octx.font = "14px monospace";
    octx.textBaseline = "top";
    octx.fillStyle = "#ff8080";

    // Wrap each source line to the canvas width, hard-breaking overlong tokens.
    const lines = [];
    for (const raw of msg.split("\n")) {
        if (raw === "") { lines.push(""); continue; }
        let cur = "";
        for (const tok of raw.split(/(\s+)/)) {
            const test = cur + tok;
            if (cur !== "" && octx.measureText(test).width > maxW) { lines.push(cur); cur = tok.replace(/^\s+/, ""); }
            else cur = test;
        }
        while (octx.measureText(cur).width > maxW) {
            let n = cur.length;
            while (n > 1 && octx.measureText(cur.slice(0, n)).width > maxW) n--;
            lines.push(cur.slice(0, n)); cur = cur.slice(n);
        }
        lines.push(cur);
    }
    let y = pad;
    for (const ln of lines) {
        if (y > logicalH - lineH) break;
        octx.fillText(ln, pad, y);
        y += lineH;
    }
}

// Catch anything that escapes the async init()/loop() call chain (which .catch
// only covers synchronously-thrown / awaited errors, not stray rejections or
// errors thrown inside requestAnimationFrame callbacks).
window.addEventListener("error", (e) => showError(e.error || e.message || "Uncaught error"));
window.addEventListener("unhandledrejection", (e) => showError(e.reason || "Unhandled promise rejection"));

// Device-lost auto-retry counter. Mobile GPUs can enter a lost/re-init loop
// (TDR-style thrashing under too much load) that never recovers and just keeps
// the device hot; cap retries so it gives up and tells the user to back off the
// load via the diagnostic URL overrides above instead of looping forever.
const MAX_DEVICE_LOST_RETRIES = 3;
let deviceLostRetries = 0;

async function init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No GPU adapter.");
    const device = await adapter.requestDevice();

    // Surface WebGPU validation / out-of-memory errors that aren't thrown but
    // reported asynchronously (the usual reason a phone just shows a white frame:
    // a shader/pipeline/allocation failed and nothing on screen ever said so).
    device.addEventListener?.("uncapturederror", (e) => showError(e.error));
    if (!device.addEventListener) device.onuncapturederror = (e) => showError(e.error);

    device.lost.then(async (info) => {
        if (_animFrameId !== null) { cancelAnimationFrame(_animFrameId); _animFrameId = null; }
        if (info.reason !== "destroyed") {
            deviceLostRetries++;
            if (deviceLostRetries > MAX_DEVICE_LOST_RETRIES) {
                showError(`GPU device lost x${deviceLostRetries} — giving up. Try ?p=20000, ?frs=0.25, ?rs=0.25, ?nrf=1 to reduce load.`);
                return;
            }
            // Non-sticky so a real init error on the retry can still replace it.
            showError(`GPU device lost (${info.reason}): ${info.message} — retrying… (${deviceLostRetries}/${MAX_DEVICE_LOST_RETRIES})`, false);
            await new Promise(r => setTimeout(r, 1000));
            init().catch(showError);
        }
    });

    // Bracket all pipeline / buffer / texture creation in error scopes so a
    // validation or OOM failure during setup is reported precisely (and doesn't
    // just silently produce a non-functional device).
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");

    fluid.initGPU(device);
    diffuse.initGPU(device, fluid);
    fluid.diffuse = diffuse;

    const fr = await initFluidRenderer(device, c, fluid.particlePosBuffer, fluid.particleVelBuffer, fluid.particleDensityBuffer, fluid.REST_DENSITY, diffuse);

    const oomErr = await device.popErrorScope();       // innermost first
    const valErr = await device.popErrorScope();
    if (valErr) { showError("Validation error during init: " + valErr.message); return; }
    if (oomErr) { showError("Out-of-memory during init: " + oomErr.message); return; }

    startLoop((cmd, count, cv, view, proj, viewProj) => {
        // Push the current wall AABBs to the renderer right before drawing — cheap
        // per-frame sync, no allocation on either side (renderer reads fluid.walls in place).
        fr.setWalls(fluid.walls);
        fr(cmd, count, cv, view, proj, viewProj);
    });
}

init().catch(showError);
