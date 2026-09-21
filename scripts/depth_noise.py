#!/usr/bin/env python3
"""
depth_noise.py
--------------
Make MuJoCo's perfect depth image look like an Orbbec Femto Bolt's. MuJoCo
returns the exact ray-hit distance for every pixel; a real ToF camera does not.

    python3 scripts/depth_noise.py --check     synthetic proofs, no ROS

Applied by mujoco_ros_cell.py after render and before the uint16-mm
conversion, when the ROS param ~depth_noise is true (default false).

WHAT IS MODELLED, AND WHERE THE NUMBERS COME FROM
=================================================
1. Random error (per-pixel Gaussian, sigma grows with distance).
   Orbbec's datasheet only gives a CEILING: "standard deviation of random
   error <= 17 mm" (850 nm, 15-95 % reflectivity, excluding multipath).
   Independent measurements of the same sensor family (Azure Kinect, which
   the Femto Bolt is a drop-in for) are far tighter, NFOV binned, white wall,
   Tolgyessy et al., Sensors 2021, 21(2), 413, Table 2:
       0.8 m -> 0.50 mm    1.5 m -> 0.58 mm    3.0 m -> 0.98 mm
   SIGMA_MM_BY_DIST below is that measured curve. Use sigma_scale > 1 to move
   toward the 17 mm datasheet ceiling (scale ~17-30 reaches it) for
   worst-case testing.
2. Incidence angle. The same paper found surfaces at grazing angles are much
   noisier. Approximated as sigma / cos(theta), with theta taken from the
   depth image's own local slope (floored so it cannot blow up).
3. Flying pixels. Depth at object edges is "inaccurately placed outside the
   actual object" (same paper): edge pixels mix foreground and background
   returns. Modelled as edge pixels (depth jump > EDGE_JUMP_M to a neighbour)
   either dropping out or landing at a random depth between the two surfaces.
4. Operating range. NFOV unbinned is specified 0.5-3.86 m; outside that the
   sensor returns 0 (invalid), so we do too.
5. Systematic bias (optional, default 0): spec is "< 11 mm + 0.1 % of
   distance". bias_mm adds a constant offset; +-11 mm is the datasheet bound.

NOT MODELLED (no honest numbers for them from these sources): multipath,
the ~2 mm warm-up drift over the first ~60 min, reflectivity-dependent noise
(the sim has no per-surface IR reflectivity), lens distortion.

Sources: Orbbec Femto Bolt Hardware Specifications
(doc.orbbec.com/.../Femto Bolt Hardware Specifications); Tolgyessy et al.,
"Evaluation of the Azure Kinect and Its Comparison to Kinect V1 and V2",
Sensors 2021 (PMC7827245).
"""

import sys
from dataclasses import dataclass

import numpy as np

# Measured Azure Kinect NFOV-binned noise (metres -> mm std dev).
SIGMA_MM_BY_DIST = ([0.8, 1.5, 3.0], [0.5019, 0.5800, 0.9776])

RANGE_M = (0.5, 3.86)       # Femto Bolt NFOV unbinned operating range
EDGE_JUMP_M = 0.05          # neighbour depth jump that marks an object edge
COS_FLOOR = 0.25            # caps the grazing-angle noise gain at 4x


@dataclass
class DepthNoiseConfig:
    sigma_scale: float = 1.0        # 1 = measured; ~17-30 = datasheet ceiling
    bias_mm: float = 0.0            # constant systematic offset
    incidence: bool = True
    flying_pixels: bool = True
    flying_prob: float = 0.5        # fraction of edge pixels affected
    flying_dropout: float = 0.5     # of those, fraction that drop to invalid
    clip_range: bool = True


def _neighbour_extrema(z, invalid):
    """Per-pixel min/max depth over the 3x3 neighbourhood, ignoring invalid."""
    f = np.where(invalid, np.nan, z)
    p = np.pad(f, 1, mode="edge")
    stack = np.stack([p[i:i + z.shape[0], j:j + z.shape[1]]
                      for i in range(3) for j in range(3)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmin(stack, axis=0), np.nanmax(stack, axis=0)


def apply_depth_noise(depth_m, invalid, fx, rng, cfg=None):
    """Return (noisy depth in metres float32, invalid mask). Inputs untouched."""
    cfg = cfg or DepthNoiseConfig()
    z = depth_m.astype(np.float64).copy()
    invalid = invalid.copy()
    valid = ~invalid

    sigma_mm = np.interp(z, *SIGMA_MM_BY_DIST) * cfg.sigma_scale
    if cfg.incidence:
        zi = np.where(valid, z, np.nan)
        gy, gx = np.gradient(zi)
        px = z / fx                                   # metres per pixel at depth z
        cos_t = 1.0 / np.sqrt(1.0 + (gx / px) ** 2 + (gy / px) ** 2)
        cos_t = np.nan_to_num(cos_t, nan=1.0)
        sigma_mm = sigma_mm / np.clip(cos_t, COS_FLOOR, 1.0)

    z = z + rng.standard_normal(z.shape) * sigma_mm * 1e-3 + cfg.bias_mm * 1e-3

    if cfg.flying_pixels:
        lo, hi = _neighbour_extrema(depth_m, invalid)
        edge = valid & ((hi - lo) > EDGE_JUMP_M)
        hit = edge & (rng.random(z.shape) < cfg.flying_prob)
        drop = hit & (rng.random(z.shape) < cfg.flying_dropout)
        smear = hit & ~drop
        z = np.where(smear, lo + rng.random(z.shape) * (hi - lo), z)
        invalid |= drop

    if cfg.clip_range:
        invalid |= (z < RANGE_M[0]) | (z > RANGE_M[1])

    return z.astype(np.float32), invalid


# ----------------------------------------------------------------------

def _check():
    rng = np.random.default_rng(0)
    fx = 504.0
    print("depth_noise --check")

    for dist in (0.8, 1.5, 3.0):
        wall = np.full((100, 100), dist)
        cfg = DepthNoiseConfig(incidence=False, flying_pixels=False)
        z, inv = apply_depth_noise(wall, np.zeros_like(wall, bool), fx, rng, cfg)
        want = np.interp(dist, *SIGMA_MM_BY_DIST)
        got = (z - dist).std() * 1e3
        print(f"  flat wall {dist:.1f} m: sigma {got:.3f} mm (target {want:.3f})")
        assert abs(got - want) / want < 0.1 and not inv.any()

    wall = np.full((100, 100), 1.0)
    z, _ = apply_depth_noise(wall, np.zeros_like(wall, bool), fx, rng,
                             DepthNoiseConfig(sigma_scale=30, incidence=False, flying_pixels=False))
    print(f"  sigma_scale=30 at 1 m: {(z - 1).std() * 1e3:.1f} mm (datasheet ceiling 17 mm)")

    ramp = 1.0 + np.tile(np.linspace(0, 1.0, 100), (100, 1))       # steep slope
    flat = np.full((100, 100), 1.0)
    cfg = DepthNoiseConfig(flying_pixels=False)
    s_flat = (apply_depth_noise(flat, np.zeros_like(flat, bool), fx, rng, cfg)[0] - flat).std()
    s_ramp = (apply_depth_noise(ramp, np.zeros_like(ramp, bool), fx, rng, cfg)[0] - ramp)[:, 5:-5].std()
    print(f"  grazing surface noisier than head-on: {s_ramp * 1e3:.2f} vs {s_flat * 1e3:.2f} mm")
    assert s_ramp > 1.5 * s_flat

    step = np.where(np.arange(100)[None, :] < 50, 1.0, 2.0) * np.ones((100, 1))
    z, inv = apply_depth_noise(step, np.zeros_like(step, bool), fx, rng)
    edge_cols = slice(49, 51)
    between = ((z > 1.1) & (z < 1.9) & ~inv)[:, edge_cols].sum()
    print(f"  step edge: {inv[:, edge_cols].sum()} dropouts, {between} flying pixels "
          f"(of 200 edge px); away from edge: {inv[:, :45].sum() + inv[:, 55:].sum()} invalid")
    assert inv[:, edge_cols].sum() > 0 and between > 0 and inv[:, :45].sum() == 0

    rows = np.stack([np.full(30, d) for d in (0.3, 4.5, 1.0)])      # three flat bands
    _, inv = apply_depth_noise(rows, np.zeros_like(rows, bool), fx, rng,
                               DepthNoiseConfig(flying_pixels=False))
    got = inv[:, 10].tolist()
    print(f"  range clip 0.5-3.86 m: 0.3 m, 4.5 m, 1.0 m -> invalid {got}")
    assert got == [True, True, False]
    print("OK.")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _check()
