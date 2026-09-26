"""
DeepFHR-style CWT image generation: db2/sym2 mother wavelets at scales {4,5,6} -> 6 images per 20-min segment
(docs/deepfhr_original.yaml). Uses scipy.signal.cwt with the discrete wavelet's own reconstructed function,
dilated by the integer scale -- the closest available reconstruction of MATLAB's legacy discrete-wavelet CWT
(see the protocol's SUBSTITUTION note). Output: 64x64x3 RGB via a fixed colormap.
"""
import numpy as np
import pywt
from matplotlib import cm

IMG_SIZE = 64


def _cwt_convolve(data, wavelet_fn, widths):
    """scipy.signal.cwt's own (removed in scipy>=1.12) algorithm, reimplemented directly: for each width,
    sample the wavelet at min(10*width, len(data)) points and convolve with the signal ('same' mode)."""
    out = np.zeros((len(widths), len(data)), dtype=np.float64)
    for i, w in enumerate(widths):
        n = min(int(10 * w), len(data))
        wav = wavelet_fn(n, w)
        out[i] = np.convolve(data, wav, mode="same")
    return out


def _wavelet_function(name, points):
    """The wavelet's own reconstruction function, sampled at `points` points, for use as a CWT kernel."""
    w = pywt.Wavelet(name)
    _, psi, _ = w.wavefun(level=8)
    x = np.linspace(0, 1, len(psi))
    xs = np.linspace(0, 1, points)
    return np.interp(xs, x, psi).astype(np.float64)


def _make_wavelet_callable(name):
    cache = {}
    def wavelet(length, scale):
        key = (name, length, round(float(scale), 4))
        if key not in cache:
            n = max(int(round(length)), 4)
            psi = _wavelet_function(name, n)
            cache[key] = psi
        return cache[key]
    return wavelet


def cwt_scalogram(signal_1d, wavelet_name, n_scales):
    """(len,) FHR segment -> (n_scales, len) real-valued CWT magnitude across scales 1..n_scales, using the
    discrete wavelet's own reconstruction function as the CWT kernel at each scale.

    INTERPRETATION NOTE (protocol section 0/1): the paper's "wavelet scale of 4/5/6" is read here as the number
    of scales computed (1..K), not a single scale value -- consistent with Fig. 3's illustrative "wavelet scale
    of 24" example, which shows a genuine 2D time-frequency image (a single scale would give one frequency row,
    not an image) and with a real-valued CWT needing a scale range to have a frequency axis at all."""
    fn = _make_wavelet_callable(wavelet_name)
    widths = np.arange(1, n_scales + 1, dtype=np.float64)
    coeffs = _cwt_convolve(signal_1d, fn, widths)
    return np.abs(coeffs)


def to_image(scalogram_2d, size=IMG_SIZE, cmap_name="viridis"):
    """(n_scales, N) scalogram -> (size,size,3) RGB image: resize both axes, colour-mapped."""
    lo, hi = scalogram_2d.min(), scalogram_2d.max()
    norm = (scalogram_2d - lo) / (hi - lo + 1e-12)
    n_rows, n_cols = norm.shape
    x_old = np.linspace(0, 1, n_cols); x_new = np.linspace(0, 1, size)
    resized_cols = np.stack([np.interp(x_new, x_old, norm[r]) for r in range(n_rows)], axis=0)
    y_old = np.linspace(0, 1, n_rows); y_new = np.linspace(0, 1, size)
    resized = np.stack([np.interp(y_new, y_old, resized_cols[:, c]) for c in range(size)], axis=1)
    rgb = cm.get_cmap(cmap_name)(resized)[:, :, :3]
    return rgb.astype(np.float32)


def build_images(fhr_segment, wavelets=("db2", "sym2"), scales=(4, 5, 6), size=IMG_SIZE, cmap_name="viridis"):
    """One preprocessed (4800,) FHR segment -> list of 6 (64,64,3) images, in a fixed (wavelet, scale) order."""
    out = []
    for wname in wavelets:
        for s in scales:
            mag = cwt_scalogram(fhr_segment, wname, s)
            out.append(to_image(mag, size=size, cmap_name=cmap_name))
    return out
