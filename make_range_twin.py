"""Build the shooting-range twin: v2b's exact robot pixels on a robot-free range background.

    python make_range_twin.py images/source/hostage_v2b_nodot.png images/source/range_v1.png images/source/range_v2b_composite.png

1. Robot mask on v2b: dark hand/gun and white forearm, each grown from a seed point inside a
   hand-set region that keeps out the black flight cases beside the hand, then closed, row-filled
   and opened (7x7) to drop thin streaks.
2. Range plate: range_v1 has its own, different robot on the right. The lane is close to mirror
   symmetric about the target centre (x=858), so the block right of x=1120 and below y=300 takes
   the mirrored left side of the lane, with a 30 px feather.
3. Paste v2b's robot over the plate with a ~1 px feather; pixels inside the mask stay identical.
"""
import sys

import numpy as np
import scipy.ndimage as nd
from PIL import Image, ImageDraw

ROI = [(1170, 360), (1360, 360), (1405, 440), (1405, 580), (1432, 600), (1432, 640), (1560, 760), (1672, 870), (1672, 941),
       (1240, 941), (1186, 700), (1186, 470), (1170, 460)]
GUN_SEED, ARM_SEED = (430, 1265), (880, 1450)          # (y, x)
MIRROR_X, X0, Y0, FEATHER = 858, 1120, 300, 30


def grow(m, near, rad=25):
    """Largest labelled region of m within rad px of near; fails if there is none."""
    lab, _ = nd.label(m)
    y0, x0 = near
    win = lab[y0 - rad:y0 + rad, x0 - rad:x0 + rad]
    ids, counts = np.unique(win[win > 0], return_counts=True)
    assert len(ids), f"no labelled pixel near {near}"
    return lab == ids[np.argmax(counts)]


def robot_mask(a):
    H, W = a.shape[:2]
    L, chroma = a.mean(-1), a.max(-1) - a.min(-1)
    roi = Image.new("L", (W, H), 0)
    ImageDraw.Draw(roi).polygon(ROI, fill=255)
    roi = np.asarray(roi) > 0
    dark = nd.binary_opening((L < 80) & (chroma < 40) & roi, iterations=2)
    white = nd.binary_opening((L > 160) & (chroma < 35) & roi, iterations=2)
    m = nd.binary_fill_holes(nd.binary_closing(grow(dark, GUN_SEED) | grow(white, ARM_SEED), iterations=6))
    for y in range(H):
        xs = np.nonzero(m[y])[0]
        if len(xs):
            m[y, xs.min():xs.max() + 1] = True
    return nd.binary_opening(m & roi, structure=np.ones((7, 7)))


def main(robot_src, range_src, dst):
    a = np.asarray(Image.open(robot_src).convert("RGB")).astype(float)
    r = np.asarray(Image.open(range_src).convert("RGB")).astype(float)
    H, W = a.shape[:2]
    mask = robot_mask(a)
    mir = r[:, np.clip(2 * MIRROR_X - np.arange(W), 0, W - 1)]
    yy, xx = np.mgrid[0:H, 0:W]
    w = (np.clip((xx - X0) / FEATHER, 0, 1) * np.clip((yy - Y0) / FEATHER, 0, 1))[..., None]
    plate = r * (1 - w) + mir * w
    alpha = nd.gaussian_filter(mask.astype(float), 1.2)
    alpha = np.where(mask, np.maximum(alpha, 0.999), alpha)[..., None]
    comp = plate * (1 - alpha) + a * alpha
    Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).save(dst)
    same = np.abs(comp[mask] - a[mask]).max() < 0.6
    print(f"{dst}: robot mask {int(mask.sum())} px, robot pixels identical to source: {same}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
