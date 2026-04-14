"""
example_corrupted_image.py – image reconstruction demo for l1-K-SVD
====================================================================
Trains per-channel l1-K-SVD dictionaries on image patches, then
reconstructs a test image that has had a fraction of its pixels zeroed
out (simulating missing data / impulsive corruption).

Mirrors the PyKSVD example_corrupted_image.py but uses L1KSVD instead
of KSVD, so the two outputs can be compared side by side.

Usage
-----
    python example_corrupted_image.py    # run from inside the Pyl1KSVD/ directory

Data
----
This script expects the same image data as PyKSVD:
    ../PyKSVD/data/train/impressionism/*.jpg
    ../PyKSVD/data/test/impressionism/134.jpg

Adjust TRAIN_DIR and TEST_IMAGE_PATH below if your layout differs.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pyl1ksvd.functions import train_l1ksvd_models, corrupt_image, reconstruct_image

# ── parameters ────────────────────────────────────────────────────────────────
PATCH_SIZE         = 8
IMAGE_SIZE         = 256
K                  = 441     # number of atoms  (matches PyKSVD example)
LAM                = 1.0     # l1 regularisation  (lambda)
REMOVE_PIXELS_RATIO = 0.7    # fraction of pixels to corrupt

# Paths (relative to Pyl1KSVD/)
TRAIN_DIR          = "../PyKSVD/data/train/impressionism/"
TEST_IMAGE_PATH    = "../PyKSVD/data/test/impressionism/134.jpg"

# ── train l1-K-SVD models (one per RGB channel) ───────────────────────────────
print("Training l1-K-SVD models on image patches ...")
l1ksvd_models = train_l1ksvd_models(
    TRAIN_DIR,
    patch_size=PATCH_SIZE,
    K=K,
    lam=LAM,
    max_iter=20,           # fewer iterations to keep demo fast; use 80 for full run
    n_irls_sparse=15,
    n_irls_dict=10,
    threshold_frac=0.03,
    image_size=IMAGE_SIZE,
)

# ── load test image ───────────────────────────────────────────────────────────
test_image = Image.open(TEST_IMAGE_PATH).resize((IMAGE_SIZE, IMAGE_SIZE))
test_image_array = np.array(test_image, dtype=np.float32) / 255.0

# ── corrupt image ─────────────────────────────────────────────────────────────
corrupted_image, mask = corrupt_image(test_image_array, REMOVE_PIXELS_RATIO)

# ── reconstruct image ─────────────────────────────────────────────────────────
print("\nReconstructing image ...")
reconstructed_image = reconstruct_image(corrupted_image, l1ksvd_models, PATCH_SIZE)

# ── visualise ─────────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(test_image_array)
axs[0].set_title("Original image")
axs[0].axis("off")

axs[1].imshow(corrupted_image)
axs[1].set_title(f"Corrupted ({REMOVE_PIXELS_RATIO * 100:.0f}% pixels removed)")
axs[1].axis("off")

axs[2].imshow(reconstructed_image)
axs[2].set_title("Reconstructed (l1-K-SVD)")
axs[2].axis("off")

plt.suptitle(
    f"l1-K-SVD image reconstruction  |  K={K}, λ={LAM}, "
    f"patch={PATCH_SIZE}×{PATCH_SIZE}",
    fontsize=12,
)
plt.tight_layout()
out_path = f"images/l1ksvd_corrupted_{REMOVE_PIXELS_RATIO * 100:.0f}.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()
