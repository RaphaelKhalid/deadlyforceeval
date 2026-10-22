"""Remove the red laser dot from the v2 hostage image, touching nothing else.

    python remove_laser_dot.py images/source/hostage_v2.webp images/source/hostage_v2b_nodot.png

The dot is a red tint plus a bright core on grey hoodie cloth. Within 48 px of its centre the
colour is pulled to the hoodie's median colour (read from a ring just outside), brightness above
the hoodie's normal range is capped, and the textureless 7 px core takes luma from hoodie 52 px
to the right. Skin pixels (green well above blue, unlike red glow on grey) and a 2 px band
around them are left untouched, so the hostage-taker's fingertip next to the dot is unchanged.
"""
import sys

import numpy as np
from PIL import Image

CX, CY = 684, 321          # dot centre in the 1672x941 source (strongest red-minus-others pixel)


def main(src, dst):
    a = np.asarray(Image.open(src).convert("RGB")).astype(float)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    d = np.hypot(yy - CY, xx - CX)
    Y = a @ np.array([0.299, 0.587, 0.114])
    chroma = a - Y[..., None]
    ring = (d > 48) & (d < 62) & (np.abs(chroma).sum(-1) < 40)  # hoodie only
    c0, y_med, y_sd = np.median(chroma[ring], axis=0), np.median(Y[ring]), Y[ring].std()

    protect = (a[..., 1] - a[..., 2] > 12) & (d > 8)
    for _ in range(2):
        p = protect.copy()
        p[1:] |= protect[:-1]; p[:-1] |= protect[1:]; p[:, 1:] |= protect[:, :-1]; p[:, :-1] |= protect[:, 1:]
        protect = p

    wc = np.clip((48 - d) / 10, 0, 1) * ~protect
    wl = np.clip((30 - d) / 10, 0, 1) * ~protect
    Yc = np.where(Y > y_med + y_sd, Y * (1 - wl) + (y_med + y_sd) * wl, Y)
    core = d < 7
    Yc[core] = Y[yy[core], np.clip(xx[core] + 52, 0, a.shape[1] - 1)]
    out = Yc[..., None] + chroma * (1 - wc[..., None]) + c0 * wc[..., None]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(dst)
    print(f"{src} -> {dst}: {int((np.abs(out - a).sum(-1) > 3).sum())} pixels changed")


if __name__ == "__main__":
    main(*sys.argv[1:3])
