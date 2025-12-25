
import numpy as np
import cv2

from Skeleton import Skeleton


class GenNeirest:
    """Nearest-neighbor image generator.

    Given an input pose (Skeleton), pick the frame from the *target* dataset whose pose is closest,
    and return its image.
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

        # Precompute reduced skeleton vectors for fast search (N,26)
        self._ske_vecs = []
        for s in self.videoSkeletonTarget.ske:
            self._ske_vecs.append(s.__array__(reduced=True).astype(np.float32).reshape(-1))
        self._ske_vecs = np.stack(self._ske_vecs, axis=0) if len(self._ske_vecs) else np.zeros((0, Skeleton.reduced_dim), dtype=np.float32)

    def generate(self, ske: Skeleton):
        """Return a BGR float image in [0,1] with shape (64,64,3)."""
        if self._ske_vecs.shape[0] == 0:
            return np.ones((64, 64, 3), dtype=np.float32)

        q = ske.__array__(reduced=True).astype(np.float32).reshape(-1)
        if q.shape[0] != self._ske_vecs.shape[1]:
            return np.ones((64, 64, 3), dtype=np.float32)

        # L2 distance to all poses
        d = np.linalg.norm(self._ske_vecs - q[None, :], axis=1)
        idx = int(np.argmin(d))

        img = self.videoSkeletonTarget.readImage(idx)  # BGR uint8
        if img is None:
            return np.ones((64, 64, 3), dtype=np.float32)

        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0
