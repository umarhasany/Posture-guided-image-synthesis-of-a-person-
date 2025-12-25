# Motion Pose Transfer

This repository implements motion transfer from a source video to a target appearance using three trained models:

- ske26-to-result (vanilla network: 26D reduced skeleton vector -> RGB image)
- skeim-to-result (vanilla network: stick/skeleton image -> RGB image)
- GAN (conditional GAN: stick/skeleton image -> RGB image, with discriminator)

The provided notebook runs end-to-end in Google Colab: it prepares data, trains each model, and generates result videos.

## Demo video

A short demo video (around 2 minutes) is included as part of the submission. It shows:

- unzipping the provided solution code
- preparing the input videos and skeleton precomputation
- running inference with an already-trained model to produce an output video

(If the grader opens the demo video file, it should be clear which commands were run and what output files were produced.)

## Repository and file structure

The workflow assumes this layout inside the Colab runtime:

/content/
  TP_solution_code_v2.zip
  tp/
    DanceDemo.py
    GenVanillaNN.py
    GenGAN.py
    GenNearest.py
    Skeleton.py
    (project helper files copied into tp/)
  tp/data/
    taichi2.mp4
    karate1.mp4
    taichi2/          (frames + skeleton outputs)
    karate1/          (frames + skeleton outputs)
    Dance/            (saved model checkpoints)
    out/              (generated output frames/videos)

### About TP_solution_code_v2.zip

The assignment starter/solution code is packaged as TP_solution_code_v2.zip and is unzipped into /content/tp at the start of the notebook.

In the initial setup cells, the zip is extracted and then a few helper files from the course/project distribution are placed alongside the extracted files (for example VideoSkeleton.py / VideoReader.py / Vec3.py, depending on the distribution you were given). Those helper files are required because:

- DanceDemo.py and the generators depend on VideoSkeleton / video reading utilities
- the skeleton code uses vector utilities (Vec3) for joint computations

If those files are missing, you will typically see import errors or runtime errors when trying to load skeletons/images.

## Environment

Tested in Google Colab (Python 3.11) with:

- PyTorch (newer versions where torch.load defaults to weights_only=True)
- OpenCV (headless build in Colab; GUI calls like imshow/waitKey are not supported)

The code was patched to be compatible with Colab’s headless OpenCV and PyTorch’s safer default loading behavior.

## Quickstart (run with trained networks)

The fastest path is:

1) Unzip the code to /content/tp
2) Ensure input videos are placed in /content/tp/data/
3) Ensure skeleton precomputation files exist (or recompute)
4) Run DanceDemo.py with the desired generator type

### 1) Unzip solution code

In Colab:

%%bash
set -e
mkdir -p /content/tp
unzip -o /content/TP_solution_code_v2.zip -d /content/tp
ls -lah /content/tp

### 2) Put videos in the expected location

The notebook assumes:

/content/tp/data/taichi2.mp4
/content/tp/data/karate1.mp4

Example:

%%bash
set -e
mkdir -p /content/tp/data
mv /content/taichi2.mp4 /content/tp/data/taichi2.mp4 2>/dev/null || true
mv /content/karate1.mp4 /content/tp/data/karate1.mp4 2>/dev/null || true
ls -lah /content/tp/data/*.mp4

### 3) Precompute skeletons (creates .pkl and frame folders)

If .pkl files do not exist or were corrupted, recompute them:

%%bash
set -e
cd /content/tp
python - <<'PY'
from VideoSkeleton import VideoSkeleton
import os

for vid in ["data/taichi2.mp4", "data/karate1.mp4"]:
    base = os.path.splitext(vid)[0]
    pkl = base + ".pkl"
    if os.path.exists(pkl):
        print("Found:", pkl)
    else:
        print("Computing:", vid)
        VideoSkeleton(vid)  # will compute and save pkl
PY

### 4) Generate a result video (inference)

DanceDemo.py takes:

python DanceDemo.py <src_video> <tgt_video> <gen_type>

Where gen_type matches the notebook:

- 2 : Vanilla GenVanillaNN (ske26-to-result OR skeim-to-result depending on GenVanillaNN optSkeOrImage)
- 3 : GAN (GenGAN)

The notebook patches DanceDemo.py to save outputs to data/out/ (JPG frames and result.mp4), instead of trying to open GUI windows.

Example inference run:

%%bash
set -e
cd /content/tp
rm -rf data/out && mkdir -p data/out
python DanceDemo.py data/taichi2.mp4 data/karate1.mp4 3
ls -lah data/out

## Training

Training is done per target video (appearance). In the notebook, karate1.mp4 is used as the target training set.

### A) Train ske26-to-result (vanilla, reduced skeleton vector)

This uses GenVanillaNN with optSkeOrImage = 1.

%%bash
set -e
cd /content/tp
python GenVanillaNN.py data/karate1.mp4

Expected checkpoint:

data/Dance/DanceGenVanillaFromSke26.pth

Notes:
- GUI calls (imshow/waitKey) are disabled for Colab.
- tensor2image() is patched to convert to uint8 before cv2.cvtColor.

### B) Train skeim-to-result (vanilla, stick image)

This uses GenVanillaNN with optSkeOrImage = 2.

%%bash
set -e
cd /content/tp
python GenVanillaNN.py data/karate1.mp4

Expected checkpoint:

data/Dance/DanceGenVanillaFromSkeim.pth

### C) Train GAN

%%bash
set -e
cd /content/tp
python GenGAN.py data/karate1.mp4

Expected checkpoints:

data/Dance/DanceGenGAN_G.pth
data/Dance/DanceGenGAN_D.pth

Notes:
- GenGAN.py is patched to avoid OpenCV GUI calls.
- Saving is done as state_dict for safer loading.
- Loading uses torch.load(..., weights_only=False) and/or state_dict loading logic to remain compatible with PyTorch 2.6+ behavior.

## What was changed (patch summary)

The notebook applies targeted patches so the code runs reliably in Colab:

1) Removed / disabled OpenCV GUI usage
   - cv2.imshow, cv2.waitKey, cv2.destroyAllWindows are not supported in Colab’s OpenCV build.
   - Output is written to disk (JPG previews + MP4 video).

2) PyTorch checkpoint loading compatibility
   - Newer PyTorch defaults torch.load to weights_only=True, which breaks loading full module pickles from older checkpoints.
   - The code was updated to load safely, preferring state_dict checkpoints and explicitly setting weights_only=False when needed.

3) Robust image type handling for OpenCV conversion
   - Some generated tensors were converted to numpy float64, causing cv2.cvtColor to throw “Unsupported depth CV_64F”.
   - tensor2image() was patched to clip/scale and cast to uint8 before calling cv2.cvtColor.

4) Notebook operational fixes
   - Working directory issues after runtime disconnect were handled by always cd’ing to a valid directory before bash operations.
   - Skeleton caches (.pkl) were recomputed when empty or inconsistent.

## Outputs

After a successful run, outputs are written to:

- data/out/result.mp4
- data/out/vis_XXXX.jpg (optional intermediate frames)

Three result videos are included with the submission (one per model):
- ske26-to-result
- skeim-to-result
- GAN

## Troubleshooting

- “cv2.error … function is not implemented … imshow/waitKey”
  The code must be patched to disable GUI calls and write outputs to disk.

- “Weights only load failed … torch.load”
  Use state_dict checkpoints or load with weights_only=False only when the checkpoint source is trusted.

- “IndexError: index 0 is out of bounds … size 0”
  The target .pkl may be empty/corrupted. Delete the target folder and .pkl and recompute skeletons.

- “shell-init: error retrieving current directory”
  This happens if a directory was deleted while it was the current working directory. Start bash cells with: cd /content

