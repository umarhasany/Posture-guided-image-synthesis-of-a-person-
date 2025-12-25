import numpy as np
import cv2
import os
import sys

from VideoSkeleton import VideoSkeleton, combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import GenVanillaNN
from GenGAN import GenGAN


class DanceDemo:
    """Run a dance demo: apply source motion to target appearance."""
    def __init__(self, filename_src, filename_tgt, typeOfGen=1):
        # Target: person whose appearance we want
        self.target = VideoSkeleton(filename_tgt)
        # Source: motion provider
        self.source = VideoReader(filename_src)

        if typeOfGen == 1:          # Nearest neighbor
            print("Generator: GenNeirest (nearest neighbor)")
            self.generator = GenNeirest(self.target)

        elif typeOfGen == 2:        # Vanilla NN from 26D skeleton
            print("Generator: GenVanillaNN (ske26 -> image)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=1)

        elif typeOfGen == 3:        # Vanilla NN from stick image
            print("Generator: GenVanillaNN (stick -> image)")
            self.generator = GenVanillaNN(self.target, loadFromFile=True, optSkeOrImage=2)

        elif typeOfGen == 4:        # GAN (stick -> image + discriminator)
            print("Generator: GenGAN (conditional GAN)")
            self.generator = GenGAN(self.target, loadFromFile=True)

        else:
            raise ValueError("DanceDemo: typeOfGen must be 1..4")

    def draw(self):
        ske = Skeleton()
        image_err = np.zeros((256, 256, 3), dtype=np.uint8)

        for i in range(self.source.getTotalFrames()):
            im = self.source.readFrame(i)
            if im is None:
                break

            # Extract skeleton from source frame (mediapipe)
            ske.fromImage(im)

            # Generate target frame
            try:
                out = self.generator.generate(ske)
            except Exception as e:
                print("Generation error at frame", i, ":", e)
                out = image_err

            # Visualization: source + generated + stick
            stick = np.zeros_like(out)
            ske.draw_reduced(stick, image_size=out.shape[0] if out.ndim == 3 else 256)

            left = cv2.resize(im, (out.shape[1], out.shape[0]))
            vis = np.concatenate([left, out, stick], axis=1)
            cv2.imshow("DanceDemo: [source | generated | stick]", vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


def _usage():
    print("""Usage:
  python DanceDemo.py <source_video.mp4> <target_video.mp4> [gen_type]

gen_type:
  1 = Nearest neighbor
  2 = Vanilla NN (ske26 -> image)
  3 = Vanilla NN (stick -> image)
  4 = GAN (stick -> image)

Examples:
  python DanceDemo.py data/taichi2.mp4 data/karate1.mp4 1
  python DanceDemo.py data/taichi2.mp4 data/karate1.mp4 4
""")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        _usage()
        sys.exit(1)

    filename_src = sys.argv[1]
    filename_tgt = sys.argv[2]
    gen_type = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    print("DanceDemo: CWD=", os.getcwd())
    print("DanceDemo: source=", filename_src)
    print("DanceDemo: target=", filename_tgt)
    print("DanceDemo: gen_type=", gen_type)

    demo = DanceDemo(filename_src, filename_tgt, typeOfGen=gen_type)
    demo.draw()
