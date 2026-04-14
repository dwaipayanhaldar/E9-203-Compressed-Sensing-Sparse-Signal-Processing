import os
import glob
import numpy as np
from PIL import Image
from pyl1ksvd.pyl1ksvd import L1KSVD


def extract_patches_custom(image, patch_size):
    """
    Extract non-overlapping patches from an image.

    Parameters
    ----------
    image : ndarray, shape (H, W, C)
        Input image array.
    patch_size : int
        Side length of each square patch.

    Returns
    -------
    patches : ndarray, shape (n_patches, patch_size, patch_size, C)
    """
    h, w, c = image.shape
    patches = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if i + patch_size <= h and j + patch_size <= w:
                patch = image[i : i + patch_size, j : j + patch_size, :]
                if patch.shape == (patch_size, patch_size, c):
                    patches.append(patch)

    return np.array(patches)


def train_l1ksvd_models(
    image_dir,
    patch_size,
    K,
    lam=1.0,
    max_iter=80,
    n_irls_sparse=30,
    n_irls_dict=10,
    threshold_frac=0.03,
    image_size=256,
):
    """
    Train one L1KSVD model per colour channel on image patches.

    Parameters
    ----------
    image_dir : str
        Directory containing training JPEG images.
    patch_size : int
        Patch side length (pixels).  Patches are (patch_size)^2-dimensional.
    K : int
        Number of dictionary atoms.
    lam : float
        l1 regularisation parameter passed to L1KSVD.
    max_iter : int
        Outer iterations for L1KSVD fitting.
    n_irls_sparse : int
        IRLS iterations for the sparse-coding step.
    n_irls_dict : int
        IRLS iterations for Algorithm 1 (dictionary update).
    threshold_frac : float
        Pruning threshold fraction for sparse codes.
    image_size : int
        Images are resized to (image_size x image_size) before patching.

    Returns
    -------
    l1ksvd_models : list of L1KSVD
        One trained model per colour channel (RGB order).
    """
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))

    channel_patches = [[] for _ in range(3)]

    for image_file in image_files:
        img = Image.open(image_file).resize((image_size, image_size))
        img_array = np.array(img, dtype=np.float32) / 255.0

        patches = extract_patches_custom(img_array, patch_size)

        for channel in range(3):
            channel_patches[channel].append(
                patches[..., channel].reshape(-1, patch_size * patch_size)
            )

    l1ksvd_models = []
    for channel in range(3):
        channel_data = np.concatenate(channel_patches[channel])  # (N, d)

        model = L1KSVD(
            K=K,
            lam=lam,
            max_iter=max_iter,
            n_irls_sparse=n_irls_sparse,
            n_irls_dict=n_irls_dict,
            threshold_frac=threshold_frac,
        )
        model.fit_with_mean(channel_data.T)   # expects (d, N)
        l1ksvd_models.append(model)

    return l1ksvd_models


def corrupt_image(image, remove_ratio):
    """
    Corrupt an image by zeroing out a random fraction of pixels.

    Parameters
    ----------
    image : ndarray, shape (H, W, C)
    remove_ratio : float
        Fraction of pixels to remove (set to 0).

    Returns
    -------
    corrupted_image : ndarray, shape (H, W, C)
    mask : ndarray of bool, shape (H, W)
        True where pixels were removed.
    """
    corrupted_image = image.copy()
    mask = np.random.random(image.shape[:2]) < remove_ratio
    corrupted_image[mask] = 0
    return corrupted_image, mask


def reconstruct_image(corrupted_image, l1ksvd_models, patch_size):
    """
    Reconstruct a corrupted image using learned l1-K-SVD dictionaries.

    Parameters
    ----------
    corrupted_image : ndarray, shape (H, W, C)
        Image with missing (zero) pixels.
    l1ksvd_models : list of L1KSVD
        One trained model per colour channel.
    patch_size : int
        Patch side length used during training.

    Returns
    -------
    reconstructed_image : ndarray, shape (H, W, C)  (values clipped to [0, 1])
    """
    corrupted_patches = extract_patches_custom(corrupted_image, patch_size)

    h, w = corrupted_image.shape[:2]
    reconstructed_image = np.zeros_like(corrupted_image)

    for channel_idx, model in enumerate(l1ksvd_models):
        channel_patches = corrupted_patches[..., channel_idx].reshape(
            -1, patch_size * patch_size
        )

        # Sparse encode with null-value-aware transform
        X_corrupted, _ = model.transform_with_mean_signal_with_null_values(
            channel_patches.T
        )

        # Reconstruct patches from dictionary and codes
        Y_reconstructed = model.D @ X_corrupted           # (d, N_patches)
        reconstructed_patches = Y_reconstructed.T.reshape(-1, patch_size, patch_size)

        channel_reconstructed = np.zeros((h, w), dtype=float)
        patch_idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if (
                    i + patch_size <= h
                    and j + patch_size <= w
                    and patch_idx < len(reconstructed_patches)
                ):
                    channel_reconstructed[i : i + patch_size, j : j + patch_size] = (
                        reconstructed_patches[patch_idx]
                    )
                    patch_idx += 1

        reconstructed_image[..., channel_idx] = channel_reconstructed

    return np.clip(reconstructed_image, 0, 1)
