"""Synthetic latents with known structure, for the GPU-free blend tests (test_blend.py). Nothing here is used in runs.

A scene is a smooth random feature field over scene coordinates plus planted objects (rectangles that hide the field
and carry their own feature vector and texture). Photo v sees the scene through a view transform: scene = centre + (photo - centre) / zoom[v] + (pan[v], 0), so
with pan[v] = 0.12 * v the content moves LEFT in the image from one photo to the next (as when the camera turns right),
and with zoom[v] > 1 it grows. CLIP, VGGT and DINOv3 are different random projections of the same scene features plus
their own noise, on their own grids (CLIP 16x16 over the photo, VGGT/DINOv3 on the VGGT grid with white padding rows).
FakeClip stands in for ClipText: a fixed projection 1024 -> 768, and a planted phrase's text vector points AWAY from its
object (as CLIP's last-layer tokens do on the real data), so only the negated map (flip_text_map) finds it.
"""
import zlib

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates

DIMS = {"clip": 1024, "vggt": 2048, "dinov3": 1024}
K = 48                      # scene feature dimension
X0, Y0, RES = -1.0, -1.0, 40  # field covers scene x, y in [-1, 3] x [-1, 2] at 40 samples per unit


class FakeClip:
    def __init__(self, seed=0):
        self.W = np.random.default_rng(seed).standard_normal((1024, 768)).astype(np.float32) / 32
        self.vocab = {}

    def plant(self, phrase, clip_dir):
        t = -(clip_dir @ self.W)
        self.vocab[phrase] = (t / np.linalg.norm(t)).astype(np.float32)

    def __call__(self, text):
        if text in self.vocab:
            return self.vocab[text]
        r = np.random.default_rng(zlib.crc32(text.encode())).standard_normal(768).astype(np.float32)  # same in every process
        return r / np.linalg.norm(r)

    def project_patches(self, x):
        p = x.reshape(-1, x.shape[-1]) @ self.W
        p /= np.linalg.norm(p, axis=-1, keepdims=True).clip(1e-6)
        return p.reshape(*x.shape[:2], -1)


class Scene:
    def __init__(self, seed=0, objects=None, amp=3.0):
        rng = np.random.default_rng(seed)
        f = gaussian_filter(rng.standard_normal((3 * RES, 4 * RES, K)), (4, 4, 0))
        self.field = (f / f.std()).astype(np.float32)
        self.objects = objects or {}
        self.obj = {o: rng.standard_normal(K).astype(np.float32) for o in self.objects}
        self.amp = amp
        self.proj = {m: rng.standard_normal((K, d)).astype(np.float32) / np.sqrt(K) for m, d in DIMS.items()}
        self.pad = {m: rng.standard_normal(d).astype(np.float32) for m, d in DIMS.items()}
        self.rng = rng

    def _field_at(self, x, y):
        r, c = (y - Y0) * RES, (x - X0) * RES
        return np.stack([map_coordinates(self.field[..., i], [r, c], order=1, mode="nearest") for i in range(K)], -1)

    def features(self, sxy):
        """Scene features at scene coordinates sxy (..., 2) -> (..., K). An object hides the background: inside it the
        features are its own vector plus its own texture, which moves with it (taken from a far corner of the field at
        object-local coordinates)."""
        f = self._field_at(sxy[..., 0], sxy[..., 1])
        for i, (o, (x0, y0, x1, y1)) in enumerate(self.objects.items()):
            inside = (sxy[..., 0] >= x0) & (sxy[..., 0] <= x1) & (sxy[..., 1] >= y0) & (sxy[..., 1] <= y1)
            if inside.any():
                tex = self._field_at(sxy[..., 0][inside] - x0 + 2.0 - 0.3 * i, sxy[..., 1][inside] - y0 + 0.8)
                f[inside] = self.amp * self.obj[o] + 0.6 * tex
        return f

    def latent(self, model, xy, valid, pan=0.0, zoom=1.0, noise=0.5):
        sxy = 0.5 + (xy - 0.5) / zoom
        sxy[..., 0] += pan
        x = self.features(sxy) @ self.proj[model] + noise * self.rng.standard_normal(xy.shape[:2] + (DIMS[model],)).astype(np.float32)
        x[~valid] = self.pad[model] + 0.1 * self.rng.standard_normal(((~valid).sum(), DIMS[model])).astype(np.float32)
        return x.astype(np.float32)

    def clip_dir(self, o):
        return self.obj[o] @ self.proj["clip"]


def write_question(helpers_cls, out, tag, scene, sizes, pans, zooms=None, models=("clip", "vggt", "dinov3"),
                   other=None, noise=0.5):
    """Write photos (blank, of the given sizes) and latent files for one synthetic question -> (images, latents, fake clip).
    other: {model: Scene} to take that latent from another scene (e.g. DINOv3 from elsewhere)."""
    zooms = zooms or [1.0] * len(sizes)
    images = []
    for v, s in enumerate(sizes):
        p = out / f"{tag}_{v}.png"
        Image.new("RGB", s).save(p)
        images.append(str(p))
    rows = max(min(round(h * 518 / w / 14) * 14, 518) for w, h in sizes) // 14
    probe = helpers_cls(FakeClip())
    vg = out / f"{tag}_probe.npy"
    np.save(vg, np.zeros((len(sizes), rows, 37, 1), np.float32))
    probe.load({"images": images, "latents": {"vggt": str(vg)}, "blind": True})
    c = (np.arange(16) + 0.5) / 16
    cxy = np.stack(np.meshgrid(c, c), -1)
    latents = {}
    for m in models:
        sc = (other or {}).get(m, scene)
        views = []
        for v in range(len(sizes)):
            xy, valid = (cxy, np.ones((16, 16), bool)) if m == "clip" else (probe.vggt_xy[v], probe.vggt_valid[v])
            views.append(sc.latent(m, xy.copy(), valid, pans[v], zooms[v], noise))
        p = out / f"{tag}_{m}.npy"
        np.save(p, np.stack(views))
        latents[m] = str(p)
    clip = FakeClip()
    for o in scene.objects:
        clip.plant(o, scene.clip_dir(o))
    return images, latents, clip
