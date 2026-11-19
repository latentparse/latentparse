"""Python sandbox of the LatentParse analyst: the raw latent features plus read-only analysis tools.

Runs in a separate process (see Sandbox). Each question gets a fresh namespace with
  F["clip"][v]  (16, 16, 1024)   CLIP ViT-L/14 patch tokens of view v (photo resized to 224x224, no crop)
  F["vggt"][v]  (h, w, 2048)     VGGT last aggregator layer, patch tokens (14-px patches, width 518)
  F["dinov3"][v] (h, w, 1024)    DINOv3 ViT-L/16 patch tokens on the same grid as VGGT (only in --latents runs with it)
and the helpers below. Nothing here decodes latents (no VGGT heads are loaded) and no
other model looks at the pixels: the only extra model is CLIP's frozen text encoder,
which turns a phrase into a query vector compared against the CLIP latent.

With question["pixels"] (the Pixels source control) F holds the photo's own pixels on the same two grids instead:
  F["clip"][v]  (16, 16, 588)    14x14 RGB patches of the photo resized to 224x224
  F["vggt"][v]  (h, w, 588)      14x14 RGB patches of the photo preprocessed as for VGGT
and text_map is unavailable (it needs CLIP features).

With question["noise"] (the Random Noise control) F holds Gaussian noise with each latent's per-channel mean and std.
With question["flip_text_map"] text_map returns the negated map: CLIP's last-layer patch tokens score the named
object lowest, so the flipped map puts it on top (agent.py --flip-text-map).
With question["only"] ("clip", "vggt" or "dinov3") F holds that model's latent alone; with question["use"] (a list) F holds those
latents only. Latents loaded but not in use stay out of the analyst's reach (VGGT always gives the grid geometry).
With question["no_show"] (every --latents run) show() is left out of the namespace: no picture of any kind reaches
the analyst, in Agent-Directed or Fixed-Sequence runs. With question["basic_tools"] (every --latent run) the namespace holds the basic
tools only, as when the one-latent runs were first made, so Agent-Directed and Fixed-Sequence one-latent runs share one tool set.

Blend tools (agent.py --latents; the same namespace in the Agent-Directed and Fixed-Sequence modes): support, iou, locate, locate_all,
motion, track, relate, layout. Every cross-latent judgement in them uses support(), which never puts the seed into its
own result, so in noise the latents do not "agree" (grow() does put it in, so it is never used for agreement).

Code may import only numpy, scipy, math and a few standard-library modules; there is
no file or network access from the helpers, and each call has a time limit.
"""
import builtins
import collections
import contextlib
import io
import itertools
import math
import multiprocessing as mp
import traceback

import numpy as np

# Latents on the VGGT grid (DINOv3 is extracted on VGGT's crop, patch for patch), so they share its geometry.
ORDER = ("clip", "vggt", "dinov3")
GEOM = ("vggt", "dinov3")
GENERIC_PHRASES = ["background", "a wall", "the floor", "the sky", "a surface"]
ALLOWED_IMPORTS ={"numpy", "scipy", "math", "statistics", "itertools", "collections", "functools", "json", "re"}
MAX_OUTPUT = 2500
MAX_SHOWS = 2

# Blend tools. PATCH (one VGGT/DINOv3 column, 0.027 of the photo width) is the dead zone of every direction.
PATCH = 1 / 37
# Thresholds frozen from calib_blend.py on the calibration sample c1 (150 questions in neither s1 nor s2; judged boxes
# and Noise percentiles, results/calib_blend_c1/calib.json): high boxes hit 65% (43), medium 54% (168), random 3%.
SUPPORT_Z = 1.5                  # support(): cells above median + z * MAD (robust sd) of their similarity to the seed
CORE_FRAC = 0.04                 # locate(): VGGT and DINOv3 are seeded from CLIP's core (top 4% of the map, in the blob)
KEEP_MIN = 0.3                   # a geometry latent "keeps" CLIP's blob when its support region holds >= half of it
IOU_GG = 0.2                     # tier high: VGGT and DINOv3 support regions overlap at least this much
AREA_MIN, AREA_MAX = 0.02, 0.40  # tier high / medium: consensus box area (flip test: boxes under 0.02 hit 9%)
PEAK_HIGH, PEAK_MED = 4.0, 3.0   # tier high / medium: CLIP's pooled text-map peak (z units)
STANDOUT_HIGH, STANDOUT_MED = 1.0, 0.25  # ... and how far it stands out: peak minus the highest value outside the blob
MIN_MATCHES = 30                 # motion(): a latent with fewer mutual matches (or 1/4 of its cells) is unreliable
MIN_INLIERS = 0.20               # ... or with fewer matches explained by one shift+scale+rotation (noise, n >= 15: p99 0.18)
MOVING_MIN = 0.30                # median shift over the moving matches when at least this share moves, else over all
SAME_WAY = 0.60                  # a direction needs this share of the moving matches going the same way
TRACK_MIN = 5                    # track(): mutual matches a latent needs for a region
ZOOM = (0.93, 1.08)              # motion(): scale below / above this is "shrinks" / "grows"
PT_AGREE_HIGH = 0.1              # prompt rule: a motion direction is high only with pt_agree >= this (noise: p99 0.02)
TIERS = ("none", "low", "medium", "high")


def _r(x, n=3):
    return round(float(x), n)


def tier_of(keep, iou_gg, area, peak, standout, th=None):
    """locate()'s tier from its numbers. keep: {geometry latent: share of CLIP's core it keeps}; iou_gg: VGGT~DINOv3
    overlap (None with one geometry latent); area: consensus box area (None: no consensus cells). th overrides the
    module thresholds (calib_blend.py tries other values with this same function)."""
    th = dict(dict(KEEP_MIN=KEEP_MIN, IOU_GG=IOU_GG, AREA_MIN=AREA_MIN, AREA_MAX=AREA_MAX, PEAK_HIGH=PEAK_HIGH,
                   PEAK_MED=PEAK_MED, STANDOUT_HIGH=STANDOUT_HIGH, STANDOUT_MED=STANDOUT_MED), **(th or {}))
    if area is None:
        return "none"
    kept = [x >= th["KEEP_MIN"] for x in keep.values()]
    size_ok = th["AREA_MIN"] <= area <= th["AREA_MAX"]
    if (len(kept) == 2 and all(kept) and iou_gg >= th["IOU_GG"] and size_ok and peak >= th["PEAK_HIGH"]
            and standout >= th["STANDOUT_HIGH"]):
        return "high"
    if any(kept) and size_ok and peak >= th["PEAK_MED"] and standout >= th["STANDOUT_MED"]:
        return "medium"
    return "low"


class Found(dict):
    """A blend tool's result: a dict that prints as one compact line. Keys starting with "_" or "mask" (masks, maps)
    are kept for chaining but not printed; a nested dict prints as name(key value ...); "view" prints as the photo
    number (view + 1)."""

    def __repr__(self):
        parts = []
        for k, x in self.items():
            if str(k).startswith(("_", "mask")):
                continue
            if k == "view":
                parts.append(f"photo {x + 1}")
            elif isinstance(x, dict):
                parts.append(f"{k}(" + " ".join(f"{a} {b}" for a, b in x.items() if not str(a).startswith(("_", "mask"))) + ")")
            else:
                parts.append(f"{k} {x}")
        return "; ".join(parts)

    __str__ = __repr__


def _similarity_fit(p, q, thr, iters=200, seed=0):
    """Robust 2-D similarity transform q ~ a * p + c (complex numbers: a = scale * e^(i rot)): RANSAC on 2-point samples,
    then a least-squares refit on the inliers. p, q: (n, 2) in aspect-corrected photo units. Returns (scale, rotation in
    degrees, share of all points within thr of the fitted transform)."""
    zp, zq = p[:, 0] + 1j * p[:, 1], q[:, 0] + 1j * q[:, 1]
    n = len(zp)
    if n < 3:
        return 1.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(zp[i] - zp[j]) < 2 * thr:
            continue
        a = (zq[i] - zq[j]) / (zp[i] - zp[j])
        inl = np.abs(a * zp + (zq[i] - a * zp[i]) - zq) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 3:
        return 1.0, 0.0, 0.0
    P, Q = zp[best], zq[best]
    pm, qm = P.mean(), Q.mean()
    a = np.vdot(P - pm, Q - qm) / max(np.vdot(P - pm, P - pm).real, 1e-12)
    inl = np.abs(a * (zp - pm) + qm - zq) < thr
    return float(abs(a)), float(np.degrees(np.angle(a))), float(inl.mean())


def _safe_import(name, *args, **kwargs):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name} is not allowed in the sandbox")
    return builtins.__import__(name, *args, **kwargs)


SAFE_BUILTINS = {k: getattr(builtins, k) for k in dir(builtins) if k not in {"open", "exec", "eval", "compile", "input", "breakpoint", "__import__"}}
SAFE_BUILTINS["__import__"] = _safe_import


def _patchify(a, p=14):
    """(H, W, 3) array -> (H/p, W/p, p*p*3): each patch's pixels as one vector."""
    h, w = a.shape[0] // p, a.shape[1] // p
    return a[:h * p, :w * p].reshape(h, p, w, p, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, p * p * 3)


def clip_pixels(im):
    """The photo on the CLIP grid: resized to 224x224 without cropping, as for the CLIP latent -> (16, 16, 588) in [0, 1]."""
    from PIL import Image

    return _patchify(np.asarray(im.resize((224, 224), Image.BICUBIC), dtype=np.float32) / 255)


def vggt_pixels(im, rows):
    """The photo on the VGGT grid, preprocessed as for the VGGT latent (width 518, height a multiple of 14,
    centre-cropped to 518, padded with white to the scene's `rows` patch rows) -> (rows, 37, 588) in [0, 1]."""
    from PIL import Image

    W, H = im.size
    new_h = round(H * 518 / W / 14) * 14
    a = np.asarray(im.resize((518, new_h), Image.BICUBIC), dtype=np.float32) / 255
    if new_h > 518:
        top = (new_h - 518) // 2
        a = a[top:top + 518]
    canvas = np.ones((rows * 14, 518, 3), np.float32)
    pad_top = (rows * 14 - a.shape[0]) // 2
    canvas[pad_top:pad_top + a.shape[0]] = a
    return _patchify(canvas)


def load_latent(path, noise=None):
    """A latent file -> (views, h, w, d) float32. With noise (a key such as "3dsr/ID/clip"; noise control), Gaussian
    noise of the same shape with each view's per-channel mean and std, seeded by the key: it looks like a latent
    but carries nothing about the photo."""
    import zlib

    x = np.load(path).astype(np.float32)
    if noise is None:
        return x
    rng = np.random.default_rng(zlib.crc32(noise.encode()))
    mu, sd = x.mean((1, 2), keepdims=True), x.std((1, 2), keepdims=True)
    return mu + sd * rng.standard_normal(x.shape, dtype=np.float32)


class Helpers:
    """Per-question state: latents, photo geometry, CLIP text encoder, shown maps."""

    def __init__(self, clip_text):
        self.clip_text = clip_text
        self.shows = []
        self._embeds = {}  # phrase -> CLIP text embedding (independent of the question, so kept across questions)

    def load(self, question):
        from PIL import Image

        self.q = question
        # Blind: keep only the photo sizes (needed to place the VGGT grid), never the pixels.
        self.blind = bool(question.get("blind"))
        if self.blind:
            self.photos, self.sizes = None, []
            for p in question["images"]:
                with Image.open(p) as im:
                    self.sizes.append(im.size)
        else:
            self.photos = [Image.open(p).convert("RGB") for p in question["images"]]
            self.sizes = [im.size for im in self.photos]
        n = len(self.sizes)
        self.pixels = bool(question.get("pixels"))
        if self.pixels:
            # Photo-only control: the photo's own pixels on the same two patch grids (the latent file is read
            # only for its VGGT row count, which depends on the other views of the scene).
            rows = np.load(question["latents"]["vggt"], mmap_mode="r").shape[1]
            self.F = {"clip": [clip_pixels(im) for im in self.photos], "vggt": [vggt_pixels(im, rows) for im in self.photos]}
        else:
            # A wrong-latent donor can have a different number of views: reuse its views in order
            # (values untouched) so every photo still has one. With question["noise"] the values are noise.
            noise = question.get("noise")
            self.F = {k: [views[v % len(views)] for v in range(n)]
                      for k, views in ((k, list(load_latent(p, noise and f"{noise}/{k}"))) for k, p in question["latents"].items())}
        # Every latent but CLIP sits on the VGGT grid cell for cell (DINOv3 is extracted on VGGT's crop): check it.
        for k in self.F:
            for v in range(n if k != "clip" else 0):
                if self.F[k][v].shape[:2] != self.F["vggt"][v].shape[:2]:
                    raise ValueError(f"{k} photo {v} has grid {self.F[k][v].shape[:2]}, VGGT has {self.F['vggt'][v].shape[:2]}")
        # text_map always reads the raw CLIP tokens: CLIP's own layer norm and projection expect them.
        self.clip_raw = self.F.get("clip")
        if question.get("clip_text"):
            # Layer ablation: F["clip"] holds another CLIP layer, but text_map still reads the last one, the only layer CLIP
            # projects into its text space (agent.py --last-latents-root).
            views = list(load_latent(question["clip_text"]))
            self.clip_raw = [views[v % len(views)] for v in range(n)]
        # Latents in use: the others stay loaded just for grid geometry, out of the analyst's reach.
        self.only = question.get("only")
        self.use = set(question.get("use") or ([self.only] if self.only else self.F))
        self.flip = bool(question.get("flip_text_map"))
        self.no_show = bool(question.get("no_show"))
        self.basic = bool(question.get("basic_tools"))
        self.shows = []
        self._cache = {}  # per question: projected CLIP patches, raw text maps, patch matches, row components
        self._vggt_geometry()

    def _check(self, model):
        if model not in self.use:
            raise ValueError(f"the {model} latent is not available in this run; available: {sorted(self.use)}")

    def _geo(self):
        """The geometry latents (VGGT, DINOv3) in use, in a fixed order."""
        return [k for k in GEOM if k in self.use]

    def _valid(self, model, v):
        """Real cells of `model`'s grid in photo v: all 16x16 for CLIP, vggt_valid[v] for VGGT and DINOv3."""
        return np.ones((16, 16), bool) if model == "clip" else self.vggt_valid[v]

    def _vggt_geometry(self):
        """Photo coordinates (x, y in [0, 1]) of each VGGT patch centre, and which patches are padding."""
        self.vggt_xy, self.vggt_valid = [], []
        for v, (W, H) in enumerate(self.sizes):
            gh, gw, _ = self.F["vggt"][v].shape
            new_h = round(H * 518 / W / 14) * 14
            crop = (new_h - 518) // 2 if new_h > 518 else 0
            eff_h = min(new_h, 518)
            pad_top = (gh * 14 - eff_h) // 2
            ys = np.arange(gh) * 14 + 7 - pad_top
            xs = (np.arange(gw) * 14 + 7) / (gw * 14)
            valid_rows = (ys >= 0) & (ys < eff_h)
            py = (ys + crop) / new_h
            self.vggt_xy.append(np.stack(np.meshgrid(xs, py), -1))
            self.vggt_valid.append(np.repeat(valid_rows[:, None], gw, 1))

    # ---- helpers exposed to the analyst -------------------------------------------------
    def grid(self, model, v=0):
        return self.F[model][v].shape[:2]

    def unit(self, model, v=0):
        self._check(model)
        x = self.F[model][v]
        return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-6)

    def _embed(self, text):
        if text not in self._embeds:
            self._embeds[text] = self.clip_text(text)
        return self._embeds[text]

    def _text_raw(self, text, v):
        """text_map of photo v before z-scoring (negated with flip_text_map); cached per phrase and photo."""
        from scipy.ndimage import gaussian_filter

        if self.pixels:
            raise RuntimeError("text_map compares a phrase with CLIP features and is not available on pixel grids")
        self._check("clip")
        key = ("text", text, v)
        if key not in self._cache:
            if ("proj", v) not in self._cache:
                self._cache[("proj", v)] = self.clip_text.project_patches(self.clip_raw[v])
            p = self._cache[("proj", v)]
            s = p @ self._embed(text)
            s = s - np.mean([p @ self._embed(g) for g in GENERIC_PHRASES], axis=0)
            s = gaussian_filter(s, 0.7)
            self._cache[key] = -s if self.flip else s
        return self._cache[key]

    def text_map(self, text, v=0, pooled=False):
        """CLIP: similarity of every patch to a phrase minus its similarity to generic background
        phrases, lightly smoothed and z-scored over the image -> (16, 16). pooled=True z-scores over all
        photos together instead, so peaks compare across photos."""
        s = self._text_raw(text, v)
        if pooled:
            allv = np.stack([self._text_raw(text, u) for u in range(len(self.sizes))])
            return (s - allv.mean()) / (allv.std() + 1e-6)
        return (s - s.mean()) / (s.std() + 1e-6)

    def _xy(self, model, v):
        """Photo coordinates (x, y in [0, 1]) of every patch centre of `model`'s grid -> (h, w, 2)."""
        if model != "clip":  # VGGT and DINOv3 share the VGGT grid
            return self.vggt_xy[v]
        c = (np.arange(16) + 0.5) / 16
        return np.stack(np.meshgrid(c, c), -1)

    def _grid_of(self, m, v):
        shape = np.shape(m)
        if shape == (16, 16):
            return "clip"
        if shape == self.F["vggt"][v].shape[:2]:
            return "vggt"
        raise ValueError(f"array of shape {shape} is on neither grid of photo {v}: CLIP (16, 16), "
                         f"VGGT/DINOv3 {self.F['vggt'][v].shape[:2]}")

    def _mask(self, m, model, v, name="mask"):
        """Accept a bool or 0/1 float mask on `model`'s grid; explain how to fix a grid mismatch."""
        m = np.asarray(m)
        grid = (16, 16) if model == "clip" else self.F["vggt"][v].shape[:2]
        if m.shape != grid:
            raise ValueError(f"{name} has shape {m.shape} but {model} photo {v} needs {grid}; "
                             f"use {'to_clip' if model == 'clip' else 'to_vggt'}({name}, {v}) first")
        return m if m.dtype == bool else m > 0.5

    def _resample(self, m, target, v):
        from scipy.ndimage import map_coordinates

        source = self._grid_of(m, v)
        was_bool = np.asarray(m).dtype == bool
        if source == target:
            return np.asarray(m).copy()
        xy = self._xy(target, v)
        x, y = xy[..., 0], xy[..., 1]
        if source == "clip":
            r, c = y * 16 - 0.5, x * 16 - 0.5
        else:
            src = self.vggt_xy[v]
            r = np.interp(y, src[:, 0, 1], np.arange(src.shape[0]))
            c = np.interp(x, src[0, :, 0], np.arange(src.shape[1]))
        out = map_coordinates(np.asarray(m, dtype=np.float32), [r, c], order=1, mode="nearest")
        return out > 0.5 if was_bool else out

    def to_vggt(self, m, v=0):
        """Move a map or mask (from either grid) onto photo v's VGGT grid. Masks stay masks."""
        return self._resample(m, "vggt", v)

    def to_clip(self, m, v=0):
        """Move a map or mask (from either grid) onto the CLIP 16x16 grid. Masks stay masks."""
        return self._resample(m, "clip", v)

    def box(self, x0, y0, x1, y1, model="vggt", v=0):
        """Cells of the `model` grid inside a rectangle in 0-1 coordinates across the grid (0 = left/top, 1 = right/bottom).
        A cell is in when its centre is; VGGT cells are placed by their position in the image, so padding rows never count."""
        xy = self._xy(model, v)
        inside = (xy[..., 0] >= x0) & (xy[..., 0] <= x1) & (xy[..., 1] >= y0) & (xy[..., 1] <= y1)
        return inside & self._valid(model, v)

    def top_region(self, m, frac=0.08):
        """Largest connected blob among the top `frac` of map values -> boolean mask."""
        from scipy import ndimage

        thr = np.quantile(m, 1 - frac)
        lab, n = ndimage.label(m >= thr)
        if n == 0:
            return m >= thr
        sizes = ndimage.sum(np.ones_like(m), lab, range(1, n + 1))
        return lab == (1 + int(np.argmax(sizes)))

    def sim_to(self, model, mask, v=0, mask_view=None):
        """Cosine similarity of every patch of photo v to the mean feature of `mask` (taken from photo mask_view, default v)."""
        mv = v if mask_view is None else mask_view
        mask = self._mask(mask, model, mv)
        c = self.unit(model, mv)[mask].mean(0)
        return self.unit(model, v) @ (c / (np.linalg.norm(c) + 1e-6))

    def grow(self, model, seed, v=0, z=1.5):
        """Grow a seed mask: patches whose similarity to the seed mean is > mean + z*std, connected to the seed.
        The result always contains the seed, so it never confirms a region (use support for that)."""
        from scipy import ndimage

        seed = self._mask(seed, model, v, "seed")
        s = self.sim_to(model, seed, v)
        lab, _ = ndimage.label(s > s.mean() + z * s.std())
        keep = np.unique(lab[seed & (lab > 0)])
        return np.isin(lab, keep[keep > 0]) | seed

    def region_stats(self, m, mask, v=0):
        """mean/std/median of a map inside mask (same grid), area fraction and centroid (x, y in [0, 1] of the photo)."""
        model = self._grid_of(m, v)
        mask = self._mask(mask, model, v)
        if not mask.any():
            raise ValueError("mask is empty")
        xy = self._xy(model, v)[mask]
        vals = np.asarray(m)[mask]
        return {"mean": float(vals.mean()), "std": float(vals.std()), "median": float(np.median(vals)),
                "area_frac": float(mask.mean()), "centroid_xy": [round(float(a), 3) for a in xy.mean(0)]}

    def pca(self, model, v=0, k=3, mask=None):
        """Top-k principal components of the patch features (fitted on `mask` if given).
        Returns (maps, explained_variance): maps[i] is the (h, w) map of component i."""
        self._check(model)
        x = self.F[model][v]
        h, w, d = x.shape
        flat = x.reshape(-1, d)
        sel = self._valid(model, v).copy()
        if mask is not None:
            sel &= self._mask(mask, model, v)
        fit = flat[sel.reshape(-1)]
        mu = fit.mean(0)
        _, s, vt = np.linalg.svd(fit - mu, full_matrices=False)
        comps = ((flat - mu) @ vt[:k].T).reshape(h, w, k)
        ev = (s ** 2 / (s ** 2).sum())[:k]
        return [comps[..., i] for i in range(comps.shape[-1])], [round(float(e), 3) for e in ev]

    def kmeans(self, model, v=0, k=5, seed=0, mask=None):
        """k-means on L2-normalised patch features -> (h, w) label map. With mask (e.g. vggt_valid[v]) it is fitted on
        those cells only and the others get -1."""
        from scipy.cluster.vq import kmeans2

        u = self.unit(model, v)
        h, w, d = u.shape
        if mask is None:
            _, lab = kmeans2(u.reshape(-1, d).astype(np.float64), k, minit="++", seed=seed)
            return lab.reshape(h, w)
        sel = self._mask(mask, model, v)
        out = np.full((h, w), -1)
        out[sel] = kmeans2(u[sel].astype(np.float64), k, minit="++", seed=seed)[1]
        return out

    def corr(self, a, b, mask=None):
        """Pearson r between two maps on the same grid (inside mask if given)."""
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise ValueError(f"maps have shapes {a.shape} and {b.shape}; move one with to_vggt / to_clip")
        if mask is not None:
            mask = np.asarray(mask) if np.asarray(mask).dtype == bool else np.asarray(mask) > 0.5
            a, b = a[mask], b[mask]
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    def _mutual(self, va, vb, model, mask_a):
        """Mutual nearest neighbours from view va to vb: source cell indices, target cell indices, their similarities."""
        ua, ub = self.unit(model, va), self.unit(model, vb)
        d = ua.shape[-1]
        A, B = ua.reshape(-1, d), ub.reshape(-1, d)
        ia = np.arange(len(A)) if mask_a is None else np.nonzero(self._mask(mask_a, model, va, "mask_a").reshape(-1))[0]
        sim = A[ia] @ B.T
        ab = sim.argmax(1)
        mutual = (A @ B[ab].T).argmax(0) == ia
        return ia[mutual], ab[mutual], sim.max(1)[mutual]

    def match(self, va, vb, model="vggt", mask_a=None):
        """Mutual nearest-neighbour patch matches from view va to vb: count and mean shift in photo fractions."""
        src, dst, sims = self._mutual(va, vb, model, mask_a)
        if not len(src):
            return {"n": 0}
        xa, xb = self._xy(model, va).reshape(-1, 2), self._xy(model, vb).reshape(-1, 2)
        shift = xb[dst] - xa[src]
        return {"n": int(len(src)), "mean_dx": round(float(shift[:, 0].mean()), 3),
                "mean_dy": round(float(shift[:, 1].mean()), 3), "median_sim": round(float(sims.mean()), 3)}

    # ---- blend tools (--latents): the latents checked against each other ----------------------------------
    def support(self, model, seed, v=0, z=None):
        """Where latent `model` by itself puts the region a seed mask marks: the cells whose cosine similarity to the
        seed's mean feature is above median + z * MAD (robust sd) over the photo's real cells, in the connected parts
        that hold a seed cell. Each seed cell is scored against the mean of the OTHER seed cells and the seed is never
        added to the result, so a latent that does not single the region out returns little or none of it (unlike
        grow, whose result always contains the seed) -> mask on the model's grid."""
        from scipy import ndimage

        self._check(model)
        z = SUPPORT_Z if z is None else z
        valid = self._valid(model, v)
        seed = self._mask(seed, model, v, "seed") & valid
        if seed.sum() < 2:
            raise ValueError("support needs a seed of at least 2 real cells")
        u = self.unit(model, v)
        S = u[seed]
        tot = S.sum(0)
        s = u @ (tot / (np.linalg.norm(tot) + 1e-6))
        loo = tot[None] - S  # each seed cell against the mean of the others
        s[seed] = (S * loo).sum(1) / (np.linalg.norm(loo, axis=1) + 1e-6)
        vals = s[valid]
        med = np.median(vals)
        hi = (s > med + z * 1.4826 * np.median(np.abs(vals - med))) & valid
        lab, _ = ndimage.label(hi)
        keep = np.unique(lab[seed & hi])
        return np.isin(lab, keep[keep > 0])

    def _shared(self, m, v):
        """A mask of photo v (either grid) as a mask on the VGGT/DINOv3 grid, real cells only."""
        m = np.asarray(m)
        m = m if m.dtype == bool else m > 0.5
        if m.shape == (16, 16):
            m = self.to_vggt(m, v)
        return self._mask(m, "vggt", v) & self.vggt_valid[v]

    def iou(self, a, b, v=0):
        """Overlap (intersection over union, 0-1) of two masks of photo v; a CLIP mask is moved to the VGGT/DINOv3 grid
        first and only real cells count."""
        a, b = self._shared(a, v), self._shared(b, v)
        u = int((a | b).sum())
        return _r((a & b).sum() / u) if u else 0.0

    def locate(self, phrase, v=None, frac=0.08, z=None):
        """Find a named object by agreement between latents.
        1. CLIP: text_map(phrase, pooled=True) of photo v (default: the photo where it peaks highest); the blob is the
           connected part holding the peak among the top `frac` of the map, its core the part of it among the top
           CORE_FRAC (it spills less into the background); standout = peak minus the highest map value outside the blob
           grown by 2 cells (z units).
        2. VGGT and DINOv3 (those in use), each on its own: support(model, core) -> region; keep = share of the core's
           cells that region holds.
        3. Consensus = cells in at least two of the regions (CLIP blob, VGGT, DINOv3) within 2 cells of the CLIP blob,
           largest connected part -> box, area (of the box, photo fraction), centroid. VGGT and DINOv3 draw the outline,
           CLIP says where.
        tier: high = VGGT and DINOv3 both keep >= KEEP_MIN of the core, their regions overlap (iou_vggt_dinov3 >= IOU_GG),
        area within [AREA_MIN, AREA_MAX], peak >= PEAK_HIGH and standout >= STANDOUT_HIGH; medium = at least one geometry
        latent keeps >= KEEP_MIN, the area is within limits, peak >= PEAK_MED and standout >= STANDOUT_MED (with one
        geometry latent, medium is the best tier); low = consensus cells but weaker support; none = no consensus cells. Agreement shows that CLIP's blob is one coherent region for
        the other latents; that it is the named object rests on CLIP's peak and standout.
        Kept for chaining, not printed: mask (consensus, VGGT grid), mask_clip (seed), mask_vggt, mask_dinov3."""
        from scipy import ndimage

        self._check("clip")
        maps = {u: self.text_map(phrase, u, pooled=True) for u in (range(len(self.sizes)) if v is None else [v])}
        v = max(maps, key=lambda u: float(maps[u].max()))
        m = maps[v]
        lab, _ = ndimage.label(m >= np.quantile(m, 1 - frac))
        blob = lab == lab[np.unravel_index(int(np.argmax(m)), m.shape)]  # the top blob that holds the peak
        outside = ~ndimage.binary_dilation(blob, iterations=2)
        standout = float(m.max() - m[outside].max()) if outside.any() else 0.0
        valid = self.vggt_valid[v]
        seed = self.to_vggt(blob, v) & valid
        centre_y = self._xy("clip", v)[blob].mean(0)[1]
        rows_y = self.vggt_xy[v][valid][:, 1]
        if not seed.any() and not (rows_y.min() <= centre_y <= rows_y.max()):
            # the blob lies in the band VGGT's centre crop removes: no geometry check is possible there
            out = Found(view=v, tier="none", _phrase=phrase, outside_crop=True, peak=_r(m.max(), 2), standout=_r(standout, 2))
            out.update(mask=np.zeros_like(valid), mask_clip=np.zeros_like(valid))
            return out
        if seed.sum() < 2:  # blob thinner than a cell: the real cells nearest its centre
            d = ((self.vggt_xy[v] - self._xy("clip", v)[blob].mean(0)) ** 2).sum(-1)
            d[~valid] = np.inf
            one = np.zeros_like(valid)
            one[np.unravel_index(int(np.argmin(d)), d.shape)] = True
            seed = ndimage.binary_dilation(one) & valid
        lab, _ = ndimage.label((m >= np.quantile(m, 1 - CORE_FRAC)) & blob)
        core = self.to_vggt(lab == lab[np.unravel_index(int(np.argmax(m)), m.shape)], v) & valid
        core = core if core.sum() >= 2 else seed
        regs = {"clip": seed}
        keep = {}
        for g in self._geo():
            regs[g] = self.support(g, core, v, z)
            keep[g] = _r((regs[g] & core).sum() / core.sum(), 2)
        near = ndimage.binary_dilation(seed, iterations=2) & valid
        cons = (np.sum(list(regs.values()), axis=0) >= min(2, len(regs))) & near
        lab, k = ndimage.label(cons)
        if k:
            cons = lab == 1 + int(np.argmax(ndimage.sum(cons, lab, range(1, k + 1))))
        out = Found(view=v, tier="none", _phrase=phrase)
        if cons.any():
            box = self.bbox(cons, v)
            out.update(box=box, area=_r((box[2] - box[0]) * (box[3] - box[1])),
                       centroid=[_r(c) for c in self.vggt_xy[v][cons].mean(0)])
        out.update(peak=_r(m.max(), 2), standout=_r(standout, 2))
        if keep:
            out["keep"] = keep
        if len(keep) == 2:
            out["iou_vggt_dinov3"] = self.iou(regs["vggt"], regs["dinov3"], v)
        if len(maps) > 1:
            out["peaks"] = {u + 1: _r(maps[u].max(), 1) for u in maps}
        out["tier"] = tier_of(keep, out.get("iou_vggt_dinov3"), out.get("area"), out["peak"], out["standout"])
        # VGGT and DINOv3 cannot check what lies in the band VGGT's centre crop removes: if most of CLIP's blob is
        # there, their agreement is about the edge of the crop, not the object
        blob_y = self._xy("clip", v)[blob][:, 1]
        half = (rows_y[1] - rows_y[0]) / 2 if len(rows_y) > 1 else 0
        in_frac = float(np.mean((blob_y >= rows_y.min() - half) & (blob_y <= rows_y.max() + half)))
        if in_frac < 0.5 and out["tier"] in ("high", "medium"):
            out["tier"], out["outside_crop"] = "low", True
        out.update(mask=cons, **{f"mask_{k}": r for k, r in regs.items()})
        return out

    def locate_all(self, phrases, frac=0.08, z=None, v=None):
        """locate() each phrase (in photo v, as locate() takes it); when two phrases land on the same region of the same
        photo (IoU >= 0.5), the one with the lower peak drops to tier low and gets same_as -> {phrase: result}."""
        phrases = list(dict.fromkeys(phrases))
        found = {p: self.locate(p, v=v, frac=frac, z=z) for p in phrases}
        for a, b in itertools.combinations(phrases, 2):
            ra, rb = found[a], found[b]
            if "box" in ra and "box" in rb and ra["view"] == rb["view"] and self.iou(ra["mask"], rb["mask"], ra["view"]) >= 0.5:
                lo, hi = (a, b) if ra["peak"] < rb["peak"] else (b, a)
                found[lo]["tier"], found[lo]["same_as"] = "low", hi
        return found

    def _field(self, model, va, vb):
        """Mutual nearest neighbours on one latent between the real cells of photo va and of photo vb (cached) ->
        (positions in va, positions in vb, source cell index, target cell index); positions in photo fractions."""
        key = ("field", model, va, vb)
        if key not in self._cache:
            ua, ub = self.unit(model, va), self.unit(model, vb)
            d = ua.shape[-1]
            ia, ib = np.nonzero(self._valid(model, va).reshape(-1))[0], np.nonzero(self._valid(model, vb).reshape(-1))[0]
            S = ua.reshape(-1, d)[ia] @ ub.reshape(-1, d)[ib].T
            ab, ba = S.argmax(1), S.argmax(0)
            mutual = ba[ab] == np.arange(len(ia))
            src, dst = ia[mutual], ib[ab[mutual]]
            xa, xb = self._xy(model, va).reshape(-1, 2), self._xy(model, vb).reshape(-1, 2)
            self._cache[key] = (xa[src], xb[dst], src, dst)
        return self._cache[key]

    def _motion1(self, model, va, vb):
        """motion() statistics of one latent; _ok = reliable, _dir = direction on x and y, _zoom (see motion)."""
        pa, pb, _, _ = self._field(model, va, vb)
        n = len(pa)
        cell = 1 / 16 if model == "clip" else PATCH
        min_n = max(15, min(MIN_MATCHES, int(self._valid(model, va).sum()) // 4))
        if n < 3:
            return {"n": n, "_ok": False, "_dir": ["?", "?"], "_zoom": "?"}
        d = pb - pa
        moving = np.hypot(d[:, 0], d[:, 1]) >= cell
        dm = d[moving] if moving.mean() >= MOVING_MIN else d  # mostly static: the camera barely moved
        dx, dy = float(np.median(dm[:, 0])), float(np.median(dm[:, 1]))
        (wa, ha), (wb, hb) = self.sizes[va], self.sizes[vb]
        # One tolerance for every latent (1.5 VGGT cells): with CLIP's own larger cells, chance matches would pass.
        scale, rot, inl = _similarity_fit(pa * [1, ha / wa], pb * [1, hb / wb], 1.5 * PATCH)
        ok = n >= min_n and inl >= MIN_INLIERS
        dirs = []
        for x, med in ((dm[:, 0], dx), (dm[:, 1], dy)):
            if not ok:
                dirs.append("?")
            elif abs(med) < PATCH:
                dirs.append("0")
            elif np.mean(np.sign(x) == np.sign(med)) >= SAME_WAY:
                dirs.append("+" if med > 0 else "-")
            else:
                dirs.append("~")  # coherent but mixed (e.g. zoom or rotation)
        zoom = "?" if not ok else "+" if scale > ZOOM[1] else "-" if scale < ZOOM[0] else "0"
        if not ok:  # an unreliable latent prints only why: too few matches, or too few explained by one transform
            return {"n": n, "inl": _r(inl, 2), "_ok": False, "_dir": dirs, "_zoom": zoom}
        return {"n": n, "dx": _r(dx), "dy": _r(dy), "static": _r(1 - moving.mean(), 2), "inl": _r(inl, 2),
                "scale": _r(scale, 2), "rot": _r(rot, 1), "_ok": ok, "_dir": dirs, "_zoom": zoom}

    @staticmethod
    def _vote(votes, words):
        """One word from the reliable latents' directions ({model: "+", "-", "0", "~"}): "left 3/3", "left 2/3 (clip
        right)", "left (only vggt)", "mixed", "disagree (...)" or "unreliable" (no reliable latent)."""
        if not votes:
            return "unreliable"
        known = {k: d for k, d in votes.items() if d in words}
        if not known:
            return "mixed"
        top, c = collections.Counter(known.values()).most_common(1)[0]
        rest = ", ".join(f"{k} {words.get(d, 'mixed')}" for k, d in votes.items() if d != top)
        if c == 1 and len(known) == 1:
            return f"{words[top]} (only {next(iter(known))}" + (f"; {rest})" if rest else ")")
        if c >= 2 and c > len(known) / 2:
            return f"{words[top]} {c}/{len(votes)}" + (f" ({rest})" if rest else "")
        return "disagree (" + ", ".join(f"{k} {words.get(d, 'mixed')}" for k, d in votes.items()) + ")"

    def motion(self, va, vb, models=None):
        """How the content moves from photo va to photo vb, from mutual nearest-neighbour matches of real cells in each
        latent (default: every latent in use). Per latent: n matches; dx, dy = median shift of the moving matches (of all
        matches when under 30% move), photo fractions, + = right / down; static = share of matches moving less than one
        cell; inl = share of matches that one shift + scale + rotation explains to 1.5 cells (RANSAC; Gaussian noise
        stays under 0.2); scale (> 1: content grows); rot (degrees). A latent is reliable with n >= 30 (or 1/4 of its
        cells, at least 15) and inl >= MIN_INLIERS; reliable lists them, and an unreliable latent shows only n and inl.
        x, y, zoom: the direction the reliable latents give and how many agree ("left 3/3", "left 2/3 (clip right)",
        "left (only vggt)"), or "disagree", "mixed", "unreliable"; a shift under one cell (0.027) is "none". pt_agree: of the cells VGGT and DINOv3 both match, the
        share whose two targets lie within one cell of each other (both = how many; chance is about 0.01)."""
        models = [k for k in ("vggt", "dinov3", "clip") if k in self.use] if models is None else list(models)
        for k in models:
            self._check(k)
        per = {k: self._motion1(k, va, vb) for k in models}
        rel = [k for k in models if per[k]["_ok"]]
        out = Found(per)
        out["reliable"] = " ".join(rel) or "none"
        words = {"x": {"+": "right", "-": "left", "0": "none"}, "y": {"+": "down", "-": "up", "0": "none"},
                 "zoom": {"+": "grows", "-": "shrinks", "0": "none"}}
        for i, ax in enumerate(("x", "y")):
            out[ax] = self._vote({k: per[k]["_dir"][i] for k in rel}, words[ax])
        out["zoom"] = self._vote({k: per[k]["_zoom"] for k in rel}, words["zoom"])
        if "vggt" in per and "dinov3" in per:
            a, b = self._field("vggt", va, vb), self._field("dinov3", va, vb)
            common, ia, ib = np.intersect1d(a[2], b[2], return_indices=True)
            w = self.vggt_xy[vb].shape[1]
            ta, tb = a[3][ia], b[3][ib]
            near = np.maximum(np.abs(ta // w - tb // w), np.abs(ta % w - tb % w)) <= 1
            out["pt_agree"] = _r(near.mean(), 2) if len(common) else 0.0
            out["both"] = int(len(common))
        return out

    def track(self, region, vb, va=None):
        """Where a region of one photo lands in photo vb. region: a locate() result (its consensus mask, photo and
        phrase) or a mask with va. Per geometry latent in use: n = mutual matches (motion's) from the region's cells,
        at = median landing position (x, y), spread = median distance of the landings from it; spread_a is the same for
        the region in its own photo. found: "yes" when every geometry latent has n >= 5, lands compactly (spread <=
        1.5 * spread_a + 0.03) and their positions lie within max(0.1, spread_a) of each other; "no" when they have the
        matches but not that; "unclear" when one has too few matches ("(only vggt)" with one geometry latent).
        Only with found yes: shift = landing position minus the region's centroid; beyond_camera = shift minus
        motion()'s median shift of the reliable latents, i.e. the region's own movement or parallax. text_at / text_peak: the CLIP text map of the
        phrase (pooled z) at the landing cell and its peak in photo vb, when CLIP is in use."""
        if isinstance(region, dict):
            va, mask, phrase = region["view"], region["mask"], region.get("_phrase")
        elif va is None:
            raise ValueError("pass va (the region's photo) with a plain mask")
        else:
            mask, phrase = region, None
        if va == vb:
            raise ValueError("track carries a region into another photo: vb must differ from the region's photo")
        mask = self._shared(mask, va)
        geo = self._geo()
        if not geo:
            raise ValueError("track needs VGGT or DINOv3")
        if not mask.any():
            raise ValueError("region is empty")
        pts = self.vggt_xy[va][mask]
        c0 = pts.mean(0)
        spread_a = float(np.median(np.linalg.norm(pts - np.median(pts, 0), axis=1)))
        out, ats = Found(), {}
        for k in geo:
            _, pb, src, _ = self._field(k, va, vb)
            pb = pb[mask.reshape(-1)[src]]
            r = {"n": int(len(pb))}
            if len(pb):
                at = np.median(pb, 0)
                r.update(at=[_r(a, 2) for a in at], spread=_r(np.median(np.linalg.norm(pb - at, axis=1)), 2))
                ats[k] = (at, r["spread"], r["n"])
            out[k] = r
        ok = [k for k in geo if k in ats and ats[k][2] >= TRACK_MIN]
        if len(ok) < len(geo):
            found = "unclear"
        else:
            P = np.array([ats[k][0] for k in ok])
            close = np.linalg.norm(P[:, None] - P[None], axis=-1).max() <= max(0.1, spread_a)
            found = "yes" if close and all(ats[k][1] <= 1.5 * spread_a + 0.03 for k in ok) else "no"
        out["found"] = found + (f" (only {geo[0]})" if len(geo) == 1 else "")
        out["spread_a"] = _r(spread_a, 2)
        if found == "yes":  # a landing place only when the latents found the region
            at = np.mean([x[0] for x in ats.values()], 0)
            out["shift"] = [_r(a) for a in at - c0]
            cam = [c for c in (self._motion1(k, va, vb) for k in geo) if c["_ok"]]
            if cam:
                out["beyond_camera"] = [_r(a) for a in at - c0 - np.mean([[c["dx"], c["dy"]] for c in cam], 0)]
            if phrase and "clip" in self.use:
                m = self.text_map(phrase, vb, pooled=True)
                out["text_at"] = _r(m[min(15, int(at[1] * 16)), min(15, int(at[0] * 16))], 1)
                out["text_peak"] = _r(m.max(), 1)
        return out

    def relate(self, a, b, v=None):
        """Region b relative to region a in the same photo (two locate() results, or two masks with v): dx, dy =
        centroid of b minus centroid of a (photo fractions; + = b is right of / lower in the image than a; under one
        cell, 0.027, is no difference), size = box area of b / box area of a, iou of the masks, touch = the masks
        overlap or are adjacent."""
        from scipy import ndimage

        def unpack(r):
            if isinstance(r, dict):
                return r["view"], r["mask"]
            if v is None:
                raise ValueError("pass v with plain masks")
            return v, r

        (va, ma), (vb, mb) = unpack(a), unpack(b)
        if va != vb:
            raise ValueError("relate compares two regions of one photo; locate(phrase, v) finds an object in photo v")
        ma, mb = self._shared(ma, va), self._shared(mb, va)
        if not (ma.any() and mb.any()):
            raise ValueError("a region is empty")
        xy = self.vggt_xy[va]
        ca, cb = xy[ma].mean(0), xy[mb].mean(0)
        ba, bb = self.bbox(ma, va), self.bbox(mb, va)
        area_a, area_b = (ba[2] - ba[0]) * (ba[3] - ba[1]), (bb[2] - bb[0]) * (bb[3] - bb[1])
        return Found(dx=_r(cb[0] - ca[0]), dy=_r(cb[1] - ca[1]), size=_r(area_b / max(area_a, 1e-6), 2),
                     iou=self.iou(ma, mb, va), touch=bool((ndimage.binary_dilation(ma) & mb).any()))

    def _pcs(self, model, v, dims=16):
        """Top `dims` principal-component maps of a latent at the real cells of the shared grid (CLIP's moved there),
        centred -> (cells, dims)."""
        maps, _ = self.pca(model, v, dims)
        if model == "clip":
            maps = [self.to_vggt(mp, v) for mp in maps]
        X = np.stack([np.asarray(mp)[self.vggt_valid[v]] for mp in maps], 1).astype(np.float64)
        return X - X.mean(0)

    def layout(self, v=0, k=5):
        """How the latents' maps of photo v relate. segments: k-means segments (real cells) that VGGT and DINOv3 share
        (a VGGT segment and its best DINOv3 segment overlap with IoU >= 0.5; the cells both hold): box, area (share of
        the real cells), iou; largest 4 (with one geometry latent, its own largest segments). cka: linear CKA between
        each pair of latents in use over the real cells of the shared grid (top 16 principal components each; 0 =
        unrelated, 1 = the same structure; Gaussian noise gives about 0.1 or less)."""
        geo = self._geo()
        if not geo:
            raise ValueError("layout needs VGGT or DINOv3")
        valid = self.vggt_valid[v]
        labs = {g: self.kmeans(g, v, k, mask=valid) for g in geo}
        segs = []
        for i in range(k):
            a = labs[geo[0]] == i
            if not a.any():
                continue
            s = ""
            if len(geo) == 2:
                ious = [self.iou(a, labs[geo[1]] == j, v) for j in range(k)]
                j = int(np.argmax(ious))
                if ious[j] < 0.5:
                    continue
                a, s = a & (labs[geo[1]] == j), f" iou {ious[j]}"
            segs.append((int(a.sum()), f"{self.bbox(a, v)} area {_r(a.sum() / valid.sum(), 2)}{s}"))
        out = Found(view=v, segments=[s for _, s in sorted(segs, key=lambda x: -x[0])[:4]] or "none")
        X = {m: self._pcs(m, v) for m in ORDER if m in self.use}
        cka = {}
        for a, b in itertools.combinations(X, 2):
            num = np.linalg.norm(X[a].T @ X[b]) ** 2
            cka[f"{a}~{b}"] = _r(num / (np.linalg.norm(X[a].T @ X[a]) * np.linalg.norm(X[b].T @ X[b]) + 1e-12), 2)
        if cka:
            out["cka"] = cka
        return out

    def bbox(self, mask, v=0):
        """Box [x0, y0, x1, y1] in photo fractions around a mask's cells (either grid), each cell at its full size."""
        model = self._grid_of(mask, v)
        mask = self._mask(mask, model, v)
        if model == "vggt":
            mask = mask & self.vggt_valid[v]
        if not mask.any():
            raise ValueError("mask is empty")
        xy = self._xy(model, v)
        hx, hy = (xy[0, 1, 0] - xy[0, 0, 0]) / 2, (xy[1, 0, 1] - xy[0, 0, 1]) / 2
        pts = xy[mask]
        box = [pts[:, 0].min() - hx, pts[:, 1].min() - hy, pts[:, 0].max() + hx, pts[:, 1].max() + hy]
        return [round(float(np.clip(b, 0, 1)), 3) for b in box]

    def describe(self, m):
        """Quantiles and a coarse 3x3 layout (mean per cell) of a map."""
        q = np.quantile(m, [0, 0.25, 0.5, 0.75, 1])
        h, w = m.shape
        cells = [[float(m[i * h // 3:(i + 1) * h // 3, j * w // 3:(j + 1) * w // 3].mean()) for j in range(3)] for i in range(3)]
        return "quantiles " + " ".join(f"{x:.2f}" for x in q) + " | 3x3 " + " / ".join(" ".join(f"{c:.2f}" for c in row) for row in cells)

    def show(self, m, title="", v=0):
        """Draw a map or mask on its own grid; the picture is attached to your next observation."""
        if len(self.shows) >= MAX_SHOWS:
            return "show() limit reached for this step"
        self._grid_of(m, v)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        W, H = self.sizes[v]
        fig, ax = plt.subplots(figsize=(4.5, 4.5 * H / W), dpi=100)
        # No photo underneath: the map alone, on its grid, with a scale so values can be read.
        a = np.asarray(m, dtype=float)
        if m.shape != (16, 16):
            a = np.ma.masked_where(~self.vggt_valid[v], a)
        im = ax.imshow(a, cmap="jet", interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title[:60], fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        self.shows.append((title, buf.getvalue()))
        return f"shown: {title}"

    def namespace(self):
        F = {k: x for k, x in self.F.items() if k in self.use}
        ns = {"__builtins__": SAFE_BUILTINS, "np": np, "math": math, "F": F, "latents": [k for k in ORDER if k in self.use],
              "n_views": len(self.sizes), "photo_size": self.sizes, "vggt_valid": self.vggt_valid}
        for name in TOOLS + (() if self.basic else BLEND_TOOLS):
            if not (name == "show" and self.no_show):  # --latents runs: no picture reaches the analyst
                ns[name] = getattr(self, name)
        return ns


# The analyst's tools: the original 16 and the blend tools (the same in Agent-Directed and Fixed-Sequence runs).
TOOLS = ("grid", "text_map", "to_vggt", "to_clip", "box", "top_region", "sim_to", "grow", "region_stats", "pca", "kmeans",
         "corr", "match", "bbox", "describe", "show")
BLEND_TOOLS = ("support", "iou", "locate", "locate_all", "motion", "track", "relate", "layout")


class ClipText:
    """CLIP ViT-L/14 text tower and the vision-side projection, used only to compare phrases with the latent."""

    def __init__(self, name="openai/clip-vit-large-patch14"):
        import torch
        from transformers import CLIPModel, CLIPTokenizer

        self.torch = torch
        self.model = CLIPModel.from_pretrained(name, dtype=torch.float32).eval()
        self.tok = CLIPTokenizer.from_pretrained(name)

    def __call__(self, text):
        with self.torch.no_grad():
            ids = self.tok([f"a photo of {text}."], return_tensors="pt", padding=True)
            t = self.model.get_text_features(**ids)
            t = t.pooler_output if hasattr(t, "pooler_output") else t
            t = t[0].numpy()
        return t / np.linalg.norm(t)

    def project_patches(self, x):
        """Patch tokens -> CLIP's joint space (post layer norm + visual projection), unit length."""
        with self.torch.no_grad():
            p = self.torch.from_numpy(x.reshape(-1, x.shape[-1]))
            p = self.model.visual_projection(self.model.vision_model.post_layernorm(p)).numpy()
        p = p / np.linalg.norm(p, axis=-1, keepdims=True).clip(1e-6)
        return p.reshape(*x.shape[:2], -1)


def _worker(conn):
    helpers = Helpers(ClipText())
    ns = None
    while True:
        cmd, arg = conn.recv()
        if cmd == "load":
            try:
                helpers.load(arg)
                ns = helpers.namespace()
                conn.send("ok")
            except Exception:  # this question's data is bad: report it and keep the worker for the next one
                ns = None
                conn.send(("error", traceback.format_exc(limit=3)))
        elif cmd == "run":
            helpers.shows = []
            out = io.StringIO()
            np.set_printoptions(precision=3, suppress=True, threshold=60, edgeitems=3)
            try:
                with contextlib.redirect_stdout(out):
                    exec(compile(arg, "<analyst>", "exec"), ns)
            except Exception:
                out.write("\n" + traceback.format_exc(limit=2))
            text = out.getvalue()
            if len(text) > MAX_OUTPUT:
                text = text[:MAX_OUTPUT] + f"\n... [output cut at {MAX_OUTPUT} characters; print summaries, not arrays]"
            conn.send((text, list(helpers.shows)))
        elif cmd == "stop":
            return


class Sandbox:
    """One worker process for a whole run; restarted if a call times out."""

    def __init__(self, timeout=90):
        self.timeout = timeout
        self._start()

    def _start(self):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child,), daemon=True)
        self.proc.start()
        self.question = None

    def load(self, question):
        self.question = question
        try:
            self.conn.send(("load", question))
            r = self.conn.recv()
        except (EOFError, OSError):  # the worker died (e.g. out of memory): restart it so later questions still run
            self.proc.kill()
            self._start()
            raise
        if isinstance(r, tuple) and r[0] == "error":
            raise RuntimeError(r[1])
        return r

    def run(self, code):
        self.conn.send(("run", code))
        if self.conn.poll(self.timeout):
            return self.conn.recv()
        self.proc.kill()
        question = self.question
        self._start()
        if question is not None:
            self.load(question)
        return f"[timed out after {self.timeout}s; the sandbox was restarted with the same data, variables you set are gone]", []

    def close(self):
        try:
            self.conn.send(("stop", None))
        except Exception:
            pass
