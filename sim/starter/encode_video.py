"""Encode rendered PNG frames into an mp4.
Authored by Shritej Gunda.
"""

import argparse, glob, os
import imageio_ffmpeg, numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="frames")
ap.add_argument("--out", default="demo.mp4")
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.frames, "*.png")))
if not files:
    raise SystemExit(f"no PNGs in {a.frames}")
h, w = cv2.imread(files[0]).shape[:2]
print(f"{len(files)} frames, {w}x{h} @ {a.fps} fps -> {a.out}")

writer = imageio_ffmpeg.write_frames(
    a.out, (w, h), fps=a.fps, quality=8, macro_block_size=1
)
writer.send(None)
for f in files:
    writer.send(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).tobytes())
writer.close()
print(
    f"wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB, "
    f"{len(files)/a.fps:.1f} s)"
)
