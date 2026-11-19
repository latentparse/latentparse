"""Marks drawn into the photos (agent.py --marks): MV-RoboBench's colored boxes, SPAR-Bench-Tiny's points
and boxes.

Both benchmarks point at objects with marks drawn into the photos ("outlined with a red bounding box", "chair (red
point)", "sofa (green bbox)"), which the blind analyst cannot see. detect_marks() finds them from the pixels alone
(numpy, PIL, scipy; no model), so the analyst can be told where they are:
  mvrb: thin rectangle outlines in six exact colors (PNG, so the colors survive exactly): per photo and color, the
        8-connected pieces of exactly that color of at least MIN_PIXELS, merged into one box (_box: the largest piece
        and those that continue its outline) -> that box if its smaller side is at least MIN_SIDE.
  spar: JPEG, so the colors only come close: red near (254, 0, 0), blue near (0, 0, 254), green dark, near (0, 128, 0),
        yellow near (255, 255, 0) (SPAR_RANGES, tight enough to leave natural dark greens such as curtains out). A
        question marks its objects with points ("chair (red point)") or with thick outlines ("(red bbox)", or "sink
        (bbox)" with no color), and agent.py passes which (kind):
        point: red, green or blue filled discs, about 41 px across on every photo size: per photo and color, every
               8-connected piece of at least POINT_PIXELS that fills >= FILL_MIN of its bounding box, with sides
               within a factor 2 of each other and at most MAX_POINT px -> its centre and bounding box;
        box: outlines 8-19 px thick, also yellow: per photo and color, the pieces of at least BOX_PIXELS that are a
             bar (one side at most BAR px) or hollow (the centre ninth of its bounding box under 10% filled) with at
             least two straight sides (70% or more of the side covered within BAR px of it), merged as for mvrb ->
             that box if its smaller side is at least BAR.
Any other benchmark has no marks. Positions are photo fractions (x from the left, y from the top), rounded to 3 decimals.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

BOX_COLORS = {"red": (231, 76, 60), "blue": (52, 152, 219), "yellow": (241, 196, 15), "pink": (232, 62, 140),
              "green": (46, 204, 113), "purple": (155, 89, 182)}
MIN_PIXELS = 20   # mvrb: smaller pieces are stray pixels, not a mark
MIN_SIDE = 8      # mvrb: a narrower box is a speck of natural pixels in the exact color
EDGE = 4          # mvrb: outline pieces whose edges lie within EDGE px continue each other (_box)
SPAR_RANGES = {"red": ((180, 255), (0, 70), (0, 70)), "green": ((0, 50), (90, 170), (0, 50)),
               "blue": ((0, 70), (0, 70), (180, 255)), "yellow": ((180, 255), (180, 255), (0, 70))}  # (R, G, B) ranges
POINT_COLORS = ("red", "green", "blue")
POINT_PIXELS, FILL_MIN, MAX_POINT = 300, 0.6, 50
BOX_PIXELS, BAR = 400, 28
EIGHT = np.ones((3, 3), int)


def _pieces(mask, min_pixels):
    """8-connected pieces of a boolean mask with at least min_pixels pixels -> [(pixels, [x0, y0, x1, y1] in pixels, the
    piece's own mask over that box)], largest first."""
    lab, n = ndimage.label(mask, structure=EIGHT)
    if not n:
        return []
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return sorted(((int(s), [x.start, y.start, x.stop, y.stop], lab[y, x] == i)
                   for i, (s, (y, x)) in enumerate(zip(sizes, ndimage.find_objects(lab)), 1) if s >= min_pixels),
                  key=lambda p: -p[0])


def _box(boxes, t, min_side, w, h):
    """One outline from its pieces' boxes, largest first: the first, grown by each later one that continues it (an edge
    within t px of the same edge, not on the photo border, and a gap of at most 2t) -> [x0, y0, x1, y1] in pixels, or
    None without pieces or if its smaller side is under min_side."""
    if not boxes:
        return None
    box = boxes[0]
    for b in boxes[1:]:
        share = any(abs(box[i] - b[i]) <= t and abs(box[i] - lim) > t for i, lim in enumerate((0, 0, w, h)))
        near = b[0] <= box[2] + 2 * t and box[0] <= b[2] + 2 * t and b[1] <= box[3] + 2 * t and box[1] <= b[3] + 2 * t
        if share and near:
            box = [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
    return box if min(box[2] - box[0], box[3] - box[1]) >= min_side else None


def _outline_piece(b, sub):
    """spar: a piece of a thick outline, a bar or hollow with at least two straight sides."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(bw, bh) <= BAR:
        return True
    if sub[bh // 3: 2 * bh // 3, bw // 3: 2 * bw // 3].mean() >= 0.1:
        return False
    sides = (sub[:BAR].any(0).mean(), sub[-BAR:].any(0).mean(), sub[:, :BAR].any(1).mean(), sub[:, -BAR:].any(1).mean())
    return sum(s >= 0.7 for s in sides) >= 2


def _point(n, b):
    """spar: a filled disc."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    return n >= FILL_MIN * bw * bh and max(bw, bh) <= 2 * min(bw, bh) and max(bw, bh) <= MAX_POINT


def _mark(k, color, kind, b, w, h):
    x0, y0, x1, y1 = b
    return {"photo": k, "color": color, "kind": kind,
            "box": [round(x0 / w, 3), round(y0 / h, 3), round(x1 / w, 3), round(y1 / h, 3)],
            "center": [round((x0 + x1) / 2 / w, 3), round((y0 + y1) / 2 / h, 3)]}


def detect_marks(image_paths, bench, kind=None):
    """The marks in a question's photos -> [{"photo": k (1-based), "color", "kind": "box" | "point", "box": [x0, y0, x1,
    y1], "center": [x, y]}], by photo, then color, then left to right; [] for a benchmark without marks. kind (spar):
    "point" or "box", the marks the question names (agent.py reads it from the question); None looks for both."""
    if bench not in ("mvrb", "spar"):
        return []
    out = []
    for k, path in enumerate(image_paths, 1):
        a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
        h, w = a.shape[:2]
        if bench == "mvrb":
            for color, rgb in BOX_COLORS.items():
                b = _box([b for _, b, _ in _pieces((a == rgb).all(-1), MIN_PIXELS)], EDGE, MIN_SIDE, w, h)
                if b:
                    out.append(_mark(k, color, "box", b, w, h))
            continue
        for color, rng in SPAR_RANGES.items():
            mask = np.all([(a[..., c] >= lo) & (a[..., c] <= hi) for c, (lo, hi) in enumerate(rng)], 0)
            if kind != "box" and color in POINT_COLORS:
                found = [_mark(k, color, "point", b, w, h) for n, b, _ in _pieces(mask, POINT_PIXELS) if _point(n, b)]
                out += sorted(found, key=lambda m: m["center"])
            if kind != "point":
                b = _box([b for _, b, sub in _pieces(mask, BOX_PIXELS) if _outline_piece(b, sub)], BAR, BAR, w, h)
                if b:
                    out.append(_mark(k, color, "box", b, w, h))
    return out
