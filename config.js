export const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.innerWidth < 768;
export const WG_SIZE = isMobile ? 32 : 64;
// Mobile WebGPU implementations may not handle DPR-scaled canvases reliably; use 1x there.
export const dpr = isMobile ? 1 : Math.min(window.devicePixelRatio || 1, 2);
// Diffuse (spray/foam/bubble) particle pool size — fixed-size free-list pool,
// sized ~2-3x the expected simultaneous live count (see ref/diffuse-particles-gpu-migration.md §7-3).
export const DIFFUSE_MAX_COUNT = isMobile ? 12000 : 30000;

// ─────────────────────────────────────────────────────────────
//  Diagnostic URL overrides (e.g. ?p=20000&frs=0.25) — lets low-end/mobile
//  devices be load-tested (device-lost / TDR triage) without a code push.
//  Absent params fall back to the default (call-site-supplied), so passing
//  no query params at all reproduces current behaviour exactly.
// ─────────────────────────────────────────────────────────────
const _q = new URLSearchParams(location.search);
export function urlNum(key, def)  { const v = parseFloat(_q.get(key)); return Number.isFinite(v) ? v : def; }
export function urlFlag(key)      { return _q.has(key); }
