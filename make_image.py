"""Turn a generated source image into the 224x224 head_cam frame.

Same path as the YAM adapter: a 4:3 frame at 640x480, then a plain resize to
224x224 (cv2.resize default is bilinear, so PIL BILINEAR here).

    python make_image.py images/source/hostage_v2.webp images/hostage_v2_224.png
"""
import sys

from PIL import Image

X_NUDGE = 60  # px right of centre, keeps the robot hand inside the 4:3 crop


def main(src, dst):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    cw = int(h * 4 / 3)
    x0 = min(max((w - cw) // 2 + X_NUDGE, 0), w - cw)
    frame = im.crop((x0, 0, x0 + cw, h)).resize((640, 480), Image.BICUBIC)
    frame.resize((224, 224), Image.BILINEAR).save(dst)
    print(f"{src} {w}x{h} -> crop x0={x0} w={cw} -> 640x480 -> 224x224 -> {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
