import numpy as np

def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray
        the input linear value in range 0-1
    :param threshold: float threshold 0-255
        the threshold for transition from linear to log mapping
    Returns: the log value
    """
    # converting x into np.float64.
    if x.dtype is not np.float64:  # note float64 to get rounding to work
        x = x.astype(np.float64)

    x = np.maximum(x * 255.0, 1.0e-8)
    f = (1.0 / threshold) * np.log(threshold)
    y = np.where(x <= threshold, x * f, np.log(x))

    rounding = 1e8
    y = np.round(y * rounding) / rounding

    return y.astype(x.dtype)


def generate_indices(i, j, kernel_size):
    """
    Generate the indices to get the user-defined training postions
    """
    cut_num = i // kernel_size + 1
    base_indices = [k * i // cut_num for k in range(cut_num)] + [i - 1]
    indices_x = [index + i * l for l in range(4) for index in base_indices]
    return indices_x

def auto_hdr(image):
    hist, bins = np.histogram(image.flatten(), bins=65536)
    cumsum = np.cumsum(hist)
    cumsum = cumsum / cumsum[-1] # Normalize to [0,1]
    low_percentile = 15/65535  # Bottom 0.000015%
    high_percentile = 65000/65535 # Top 0.000015%
    low_value = bins[np.where(cumsum > low_percentile)[0][0]]
    high_value = bins[np.where(cumsum > high_percentile)[0][0]]
    if high_value - low_value < 1/4:
        image = high_value
        return image
    exposure_factor = 1.0 / (high_value - low_value)
    image = (image - low_value) * exposure_factor
    return image