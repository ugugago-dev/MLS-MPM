// 4×4 matrix utilities (column-major, WebGPU depth [0,1])

// `out` is an optional caller-owned Float32Array(16) to write into instead of
// allocating a new one — pass a reused scratch buffer from a per-frame hot path
// (see main.js) to avoid GC pressure. Defaults to a fresh allocation so existing
// call sites that rely on the return value staying stable keep working.
export function mat4Perspective(fovy, aspect, near, far, out = new Float32Array(16)) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    out[0] = f / aspect; out[1] = 0; out[2] = 0;        out[3] = 0;
    out[4] = 0;           out[5] = f; out[6] = 0;        out[7] = 0;
    out[8] = 0;           out[9] = 0; out[10] = far * nf; out[11] = -1;
    out[12] = 0;          out[13] = 0; out[14] = far * near * nf; out[15] = 0;
    return out;
}

export function mat4LookAt(eye, center, up, out = new Float32Array(16)) {
    let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let zl = Math.hypot(zx, zy, zz) || 1; zx /= zl; zy /= zl; zz /= zl;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    let xl = Math.hypot(xx, xy, xz) || 1; xx /= xl; xy /= xl; xz /= xl;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
    out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
    out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
    out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    out[15] = 1;
    return out;
}

// NOTE: `out` must not alias `a` or `b` — each column read of `a` (a[0], a[4], a[8],
// a[12], etc.) happens fresh on every loop iteration, so writing into `a` (or `b`)
// mid-loop would corrupt later columns' inputs. Always pass a third, distinct buffer.
export function mat4Multiply(a, b, out = new Float32Array(16)) {
    for (let col = 0; col < 4; col++) {
        const b0 = b[col*4], b1 = b[col*4+1], b2 = b[col*4+2], b3 = b[col*4+3];
        out[col*4]   = a[0]*b0 + a[4]*b1 + a[8]*b2  + a[12]*b3;
        out[col*4+1] = a[1]*b0 + a[5]*b1 + a[9]*b2  + a[13]*b3;
        out[col*4+2] = a[2]*b0 + a[6]*b1 + a[10]*b2 + a[14]*b3;
        out[col*4+3] = a[3]*b0 + a[7]*b1 + a[11]*b2 + a[15]*b3;
    }
    return out;
}

// `out` is an optional caller-owned { eye:[x,y,z], right:[x,y,z], up:[x,y,z] } to
// write into instead of allocating fresh arrays/object — same reuse pattern as the
// mat4* functions above. Defaults to a fresh allocation for existing call sites.
export function cameraVectors(cam, out = { eye: [0, 0, 0], right: [0, 0, 0], up: [0, 0, 0] }) {
    const cp = Math.cos(cam.phi), sp = Math.sin(cam.phi), ct = Math.cos(cam.theta), st = Math.sin(cam.theta);
    const dx = cp * st, dy = sp, dz = cp * ct;
    out.eye[0] = cam.target[0] + cam.radius * dx;
    out.eye[1] = cam.target[1] + cam.radius * dy;
    out.eye[2] = cam.target[2] + cam.radius * dz;
    const fx = -dx, fy = -dy, fz = -dz;
    // wup = (0,1,0), so cross(f, wup) simplifies to (-fz, 0, fx).
    let rx = -fz, ry = 0, rz = fx;
    const rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; ry /= rl; rz /= rl;
    const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
    out.right[0] = rx; out.right[1] = ry; out.right[2] = rz;
    out.up[0] = ux; out.up[1] = uy; out.up[2] = uz;
    return out;
}

// Unproject screen pixel (CSS pixels) onto the plane through cam.target. `out` is an
// optional caller-owned [x,y,z] to write into instead of allocating — pass a reused
// scratch array from a per-frame hot path. Defaults to a fresh allocation; callers that
// retain the returned array across frames (e.g. assigning it into persistent state)
// must keep using the default rather than passing a shared scratch buffer.
export function screenToWorld(sx, sy, logicalW, logicalH, cam, cv, out = [0, 0, 0]) {
    const ndcX = (sx / logicalW) * 2 - 1;
    const ndcY = 1 - (sy / logicalH) * 2;
    const aspect = logicalW / logicalH;
    const thV = Math.tan(cam.fovy / 2) / Math.min(aspect, 1);
    const thH = thV * aspect;
    const fx = cam.target[0] - cv.eye[0], fy = cam.target[1] - cv.eye[1], fz = cam.target[2] - cv.eye[2];
    const fl = Math.hypot(fx, fy, fz) || 1;
    const fdx = fx / fl, fdy = fy / fl, fdz = fz / fl;
    let rdx = fdx + ndcX * thH * cv.right[0] + ndcY * thV * cv.up[0];
    let rdy = fdy + ndcX * thH * cv.right[1] + ndcY * thV * cv.up[1];
    let rdz = fdz + ndcX * thH * cv.right[2] + ndcY * thV * cv.up[2];
    const rl = Math.hypot(rdx, rdy, rdz) || 1;
    rdx /= rl; rdy /= rl; rdz /= rl;
    const denom = rdx * fdx + rdy * fdy + rdz * fdz;
    if (Math.abs(denom) < 1e-6) { out[0] = cam.target[0]; out[1] = cam.target[1]; out[2] = cam.target[2]; return out; }
    const ttx = cam.target[0] - cv.eye[0], tty = cam.target[1] - cv.eye[1], ttz = cam.target[2] - cv.eye[2];
    const t = (ttx * fdx + tty * fdy + ttz * fdz) / denom;
    out[0] = cv.eye[0] + t * rdx; out[1] = cv.eye[1] + t * rdy; out[2] = cv.eye[2] + t * rdz;
    return out;
}

export function worldToScreen(pos, viewProj, w, h) {
    const x = pos[0], y = pos[1], z = pos[2];
    const cx = viewProj[0]*x + viewProj[4]*y + viewProj[8]*z  + viewProj[12];
    const cy = viewProj[1]*x + viewProj[5]*y + viewProj[9]*z  + viewProj[13];
    const cw = viewProj[3]*x + viewProj[7]*y + viewProj[11]*z + viewProj[15];
    if (cw <= 0) return null;
    return [(cx / cw + 1) / 2 * w, (1 - cy / cw) / 2 * h];
}

// Ray/AABB slab intersection. Returns the nearest positive t (world units along `dir`),
// or null if the ray points away from / misses the box entirely.
export function rayBoxIntersect(origin, dir, bmin, bmax) {
    let tmin = -Infinity, tmax = Infinity;
    for (let i = 0; i < 3; i++) {
        if (Math.abs(dir[i]) < 1e-9) {
            if (origin[i] < bmin[i] || origin[i] > bmax[i]) return null;
            continue;
        }
        let t1 = (bmin[i] - origin[i]) / dir[i];
        let t2 = (bmax[i] - origin[i]) / dir[i];
        if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
        tmin = Math.max(tmin, t1);
        tmax = Math.min(tmax, t2);
        if (tmin > tmax) return null;
    }
    if (tmax < 0) return null;
    return tmin > 0 ? tmin : tmax;
}

// Same slab test as rayBoxIntersect, but for an infinite LINE rather than a ray —
// negative t is a valid hit too. Returns [tmin, tmax] (the entry/exit parameters) or
// null if the line misses the box entirely. Used by FluidGPU.updateHand() to bound
// APPLY_HAND's dispatch to the segment of the (depth-independent, camera-ray-aligned)
// push cylinder that can actually touch the grid, instead of dispatching the whole grid.
export function lineBoxIntersect(origin, dir, bmin, bmax) {
    let tmin = -Infinity, tmax = Infinity;
    for (let i = 0; i < 3; i++) {
        if (Math.abs(dir[i]) < 1e-9) {
            if (origin[i] < bmin[i] || origin[i] > bmax[i]) return null;
            continue;
        }
        let t1 = (bmin[i] - origin[i]) / dir[i];
        let t2 = (bmax[i] - origin[i]) / dir[i];
        if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
        tmin = Math.max(tmin, t1);
        tmax = Math.min(tmax, t2);
        if (tmin > tmax) return null;
    }
    return [tmin, tmax];
}

// Returns [minX, minY, maxX, maxY] of the simulation box in screen space, or null if any corner is behind the camera.
export function simBoundsOnScreen(gridDims, viewProj, w, h) {
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
