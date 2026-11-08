import cv2 as cv
import numpy as np
import glob

def _centers(ys):
    if len(ys) == 0:
        return np.array([])
    groups = np.split(ys, np.where(np.diff(ys) > 2)[0] + 1)
    return np.array([g.mean() for g in groups])

def find_ruler(gray, thr=90, margin=0.15, min_ticks=5, min_spacing=20):
    """Find the vertical ruler (short horizontal dashes, regularly spaced) in the
    left/right margins of the ORIGINAL image.
    Returns {"pitch_px", "n_intervals", ...} or None."""
    h, w = gray.shape
    b = gray > thr
    up = np.zeros_like(b); up[3:] = b[:-3]
    dn = np.zeros_like(b); dn[:-3] = b[3:]
    iso = b & ~up & ~dn                      # bright pixels isolated vertically = thin dashes

    cols = list(range(0, int(margin * w))) + list(range(int((1 - margin) * w), w))
    best = None
    for x in cols:
        c = _centers(np.nonzero(iso[:, x])[0])
        if len(c) < min_ticks:
            continue
        d = np.diff(c); d = d[d > min_spacing]
        if len(d) < min_ticks - 1:
            continue
        # dominant spacing (±5 %)
        base = d[int(np.argmax([(np.abs(d - v) < 0.05 * v).sum() for v in d]))]

        # keep only the centers lying on a regular grid (drops stray marks)
        top = None
        for c0 in c:
            k = np.round((c - c0) / base)
            ok = np.abs(c - c0 - k * base) < 0.1 * base
            if top is None or ok.sum() > top.sum():
                top = ok
        t = c[top]
        if len(t) < min_ticks:
            continue
        n_int = int(round((t.max() - t.min()) / base))
        if n_int == 0 or len(t) / (n_int + 1) < 0.6:     # too many missing ticks: dubious grid
            continue
        if best is None or len(t) > best["n_ticks"]:
            best = {"pitch_px": (t.max() - t.min()) / n_int, "n_intervals": n_int,
                    "n_ticks": len(t), "first_y": float(t.min()),
                    "last_y": float(t.max()), "x": x}
    return best


def mm_per_pixel(gray, depth_cm=None, tick_cm=None):
    """Returns (mm_per_pixel, explanation); mm_per_pixel is None if the scale can't be deduced."""
    ruler = find_ruler(gray)
    if ruler is None:
        return None, "no ruler found"

    pitch, n = ruler["pitch_px"], ruler["n_intervals"]

    if tick_cm is None:
        if depth_cm is None:
            return None, f"ruler found ({pitch:.1f} px/tick) but tick value unknown"
        guess = depth_cm / n                          # depth = n intervals × tick value
        tick_cm = min((0.5, 1.0), key=lambda t: abs(t - guess))
        if abs(tick_cm - guess) > 0.15 * tick_cm:
            return None, f"inconsistent: {depth_cm} cm / {n} intervals = {guess:.2f} cm per tick"

    return tick_cm * 10 / pitch, f"{n} intervals of {pitch:.1f} px, {tick_cm} cm each"

def survey(folder, pitch_tol=1.5, patterns=("*.tif", "*.tiff", "*.png")):
    """Regroupe les images en familles (même taille d'origine + même espacement de graduations)."""
    paths = sorted(p for pat in patterns for p in glob.glob(f"{folder}/**/{pat}", recursive=True))
    families = []
    for path in paths:
        g = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if g is None:
            continue
        r = find_ruler(g)
        pitch = r["pitch_px"] if r else None
        for f in families:
            same_pitch = (pitch is None and f["pitch"] is None) or \
                         (pitch is not None and f["pitch"] is not None and abs(f["pitch"] - pitch) <= pitch_tol)
            if f["shape"] == g.shape and same_pitch:
                f["paths"].append(path)
                break
        else:
            families.append({"shape": g.shape, "pitch": pitch,
                             "n_int": r["n_intervals"] if r else None, "paths": [path]})

    for i, f in enumerate(families):
        pitch = f"{f['pitch']:.1f} px" if f["pitch"] else "pas de règle"
        print(f"famille {i} : {f['shape'][1]}x{f['shape'][0]}, graduations {pitch}, "
              f"{f['n_int']} intervalles, {len(f['paths'])} images, exemple : {f['paths'][0]}")
    return families

def mm_per_px_for_family(f, depth_cm):
    """Échelle d'une famille, une fois la profondeur lue sur une image d'exemple."""
    return depth_cm * 10 / (f["n_int"] * f["pitch"])

# (largeur, hauteur, espacement_px, intervalles) -> profondeur lue à la main (cm)
FAMILIES = {
    (1200, 800, 71.4, 9): 4.5,    # famille 0
    (1200, 800, 80.4, 8): 4.0,    # famille 1
    (1200, 800, 76.1, 8): 4.0,    # famille 2
    (1200, 800, 67.7, 9): 4.5,    # famille 7
    (800,  652, 47.2, 10): 5.0,   # famille 23 (hypothèse : 0,5 cm par graduation)
    (1200, 800, 111.3, 6): 6.0,   # famille 62
}

def scale_for_image(gray, tol_px=1.0):
    """mm/px pour une image d'une famille connue, sinon None."""
    r = find_ruler(gray)
    if r is None:
        return None
    h, w = gray.shape
    for (fw, fh, pitch, n_int), depth_cm in FAMILIES.items():
        if (fw, fh) == (w, h) and r["n_intervals"] == n_int and abs(r["pitch_px"] - pitch) <= tol_px:
            return depth_cm * 10 / (n_int * r["pitch_px"])
    return None
