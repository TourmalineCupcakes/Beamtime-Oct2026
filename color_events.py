"""FeCN XES calibration data: events within MIN/MAX_CUTOFF, colored by events per pixel.

Every non-zero pixel in every frame is one event. They are kept in the
table `events` (saved to EVENTS_FILE), one row per event:
    energy  pixel value
    x       detector column (0-383)
    y       detector row (0-727)
    frame   frame number within its file (0-49)
    file    index into FILES

Writes to OUT_DIR:
    color_energy_histogram.png  x = pixel energy, y = count, lines at the cutoffs
    color_event_image.png       pixels with 2-10+ events, one color per count

Requires: pip install h5py numpy hdf5plugin matplotlib
"""
from pathlib import Path

import hdf5plugin  # noqa: F401  (registers the LZ4 filter; must precede reading)
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

DATA_DIR = Path("/home/firewing/beamtime1/data/FeCN_XES_calibration")
# Spyder on Windows: DATA_DIR = Path(r"\\wsl.localhost\Ubuntu\home\firewing\beamtime1\data\FeCN_XES_calibration")
OUT_DIR = Path("/home/firewing/beamtime1/fecn_results")  # delete events.npy to recompute
# Spyder on Windows: OUT_DIR = Path(r"\\wsl.localhost\Ubuntu\home\firewing\beamtime1\fecn_results")
OUT_DIR.mkdir(exist_ok=True)
EVENTS_FILE = OUT_DIR / "events.npy"  # shared with analyze_events.py
MIN_CUTOFF = 5  # events with energy below this are dropped
MAX_CUTOFF = 70  # events with energy above this are dropped
BIN_WIDTH = 5   # histogram bin width, in pixel-energy units
MIN_EVENTS = 2   # pixels with fewer events than this are not drawn
MAX_EVENTS = 10  # color scale tops out here (pixels with more get the top color)
# One distinct color per count, starting with dark purple
COLORS = ["#440154", "#2a4ea0", "#1f9bd1", "#1fb5a0", "#52c23a",
          "#f0e61f", "#f5a623", "#e8452c", "#ff2fa0"]  # 9 colors for counts 2..10

#%% Build (or load) the event table
FILES = sorted(DATA_DIR.glob("*_data_*.h5"))
print(f"{len(FILES)} data files found")

dtype = [("energy", "f4"), ("x", "u2"), ("y", "u2"), ("frame", "u1"), ("file", "u2")]
if EVENTS_FILE.exists():
    events = np.load(EVENTS_FILE)
else:
    parts = []
    for i, path in enumerate(FILES):
        with h5py.File(path, "r", locking=False) as f:
            frames = f["entry/data/frame_data"][:, 0]  # (50, 728, 384)
        frame, y, x = np.nonzero(frames)
        part = np.empty(len(frame), dtype=dtype)
        part["energy"] = frames[frame, y, x]
        part["x"], part["y"], part["frame"], part["file"] = x, y, frame, i
        parts.append(part)
    events = np.concatenate(parts)
    np.save(EVENTS_FILE, events)

# Event count per pixel; the most active pixel is the one that dominates the data
counts = np.zeros((728, 384))
np.add.at(counts, (events["y"], events["x"]), 1)
hot_y, hot_x = np.unravel_index(counts.argmax(), counts.shape)
is_hot = (events["x"] == hot_x) & (events["y"] == hot_y)
n_frames = 50 * len(FILES)
print(f"{len(events)} events; most active pixel (x={hot_x}, y={hot_y}) has "
      f"{is_hot.sum()} of them ({is_hot.sum() / n_frames:.0%} of frames)")


def pixels_in_bin(low, high, include_hot_pixel=False):
    """Events with low <= energy < high, as rows with energy, x, y, frame, file."""
    sel = (events["energy"] >= low) & (events["energy"] < high)
    if not include_hot_pixel:
        sel &= ~is_hot
    return events[sel]


#%% Histogram: x = pixel energy, y = number of events at that energy
bins = np.arange(0, events["energy"].max() + BIN_WIDTH, BIN_WIDTH)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(events["energy"], bins=bins, label=f"all events ({len(events)})")
ax.hist(events["energy"][~is_hot], bins=bins,
        label=f"without most active pixel ({(~is_hot).sum()})")
ax.axvline(MIN_CUTOFF, color="green", linestyle="--", label=f"min cutoff = {MIN_CUTOFF}")
ax.axvline(MAX_CUTOFF, color="red", linestyle="--", label=f"max cutoff = {MAX_CUTOFF}")
ax.set_yscale("log")
ax.set_xlabel("pixel energy (pixel value)")
ax.set_ylabel("count")
ax.legend()
fig.savefig(OUT_DIR / "color_energy_histogram.png", dpi=120, bbox_inches="tight")
plt.show()

#%% Keep only events with MIN_CUTOFF <= energy <= MAX_CUTOFF (x, y stay attached)
in_range = (events["energy"] >= MIN_CUTOFF) & (events["energy"] <= MAX_CUTOFF)
kept = events[in_range]
print(f"{len(kept)} of {len(events)} events kept ({MIN_CUTOFF} <= energy <= {MAX_CUTOFF}); "
      f"{is_hot[in_range].sum()} of them from the most active pixel")

#%% Event image from the kept events only
img = np.zeros((728, 384))
np.add.at(img, (kept["y"], kept["x"]), 1)
rows, cols = np.nonzero(img >= MIN_EVENTS)
n_colors = MAX_EVENTS - MIN_EVENTS + 1
cmap = ListedColormap(COLORS[:n_colors])
norm = BoundaryNorm(np.arange(MIN_EVENTS - 0.5, MAX_EVENTS + 1.5), n_colors, clip=True)
fig, ax = plt.subplots(figsize=(5, 9))
ax.set_facecolor("white")
sc = ax.scatter(cols, rows, c=img[rows, cols], s=14, cmap=cmap, norm=norm)
ax.set_xlim(0, 384)
ax.set_ylim(728, 0)
ax.set_aspect("equal")
cbar = fig.colorbar(sc, label=f"events per pixel (>{MAX_EVENTS} shown as {MAX_EVENTS})")
cbar.set_ticks(range(MIN_EVENTS, MAX_EVENTS + 1))
ax.set_title(f"Energy {MIN_CUTOFF}-{MAX_CUTOFF}, {MIN_EVENTS}+ events ({len(rows)} px)", fontsize=10)
fig.savefig(OUT_DIR / "color_event_image.png", dpi=120, bbox_inches="tight")
plt.show()
