export const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.innerWidth < 768;
export const WG_SIZE = isMobile ? 32 : 64;
// Mobile WebGPU implementations may not handle DPR-scaled canvases reliably; use 1x there.
export const dpr = isMobile ? 1 : Math.min(window.devicePixelRatio || 1, 2);
// Diffuse (spray/foam/bubble) particle pool size — fixed-size free-list pool,
// sized ~2-3x the expected simultaneous live count (see ref/diffuse-particles-gpu-migration.md §7-3).
export const DIFFUSE_MAX_COUNT = isMobile ? 8000 : 30000;
