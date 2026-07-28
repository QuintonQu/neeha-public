import numpy as np
import pyfftw # For FFTW library access
import pyfftw.interfaces.numpy_fft as pyfftw_fft # For numpy.fft compatible interface
import numba
from numba import prange
import os # For CPU count
from scipy import sparse
import time
import gc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Configuration for multi-threading (same as before) ---
try:
    default_threads = os.cpu_count()
    if 'NUMBA_NUM_THREADS' in os.environ:
        default_threads = int(os.environ['NUMBA_NUM_THREADS'])
    elif 'OMP_NUM_THREADS' in os.environ:
        default_threads = int(os.environ['OMP_NUM_THREADS'])
    pyfftw.config.NUM_THREADS = default_threads
    os.environ['OMP_NUM_THREADS'] = str(default_threads)  # For OpenMP operations
    os.environ['MKL_NUM_THREADS'] = str(default_threads)  # For MKL operations
    os.environ['OPENBLAS_NUM_THREADS'] = str(default_threads)  # For OpenBLAS operations
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(default_threads)  # For Accelerate framework
    os.environ['NUMEXPR_NUM_THREADS'] = str(default_threads) 
except Exception:
    pyfftw.config.NUM_THREADS = 1 # Fallback
print(f"pyFFTW will use up to {pyfftw.config.NUM_THREADS} threads.")
# Numba's threading is typically controlled by the NUMBA_NUM_THREADS env var.

# Numba JIT function for division in RFFT space (adapted for rfft frequencies)
@numba.njit(parallel=True, cache=True, nogil=True)
def _compute_u_rfft_inplace(F_div_g_rfft, kx_rfft_freqs_scaled, ky_fft_freqs_scaled):
    """
    Computes F_div_g_rfft / laplacian using RFFT frequencies, modifying
    F_div_g_rfft (complex64) in place. Handles DC component.
    *_scaled frequencies already include the 2*pi factor and are float32.
    F_div_g_rfft is complex64.
    """
    h = ky_fft_freqs_scaled.shape[0]
    w_rfft = kx_rfft_freqs_scaled.shape[0]

    # Define constants explicitly with np.float32 for clarity and correctness
    # These will interact with results of np.cos (which will be float32
    # if inputs kx/ky_fft_freqs_scaled are float32) and with components
    # of F_div_g_rfft (which are float32 for real/imag parts).
    four_f32 = np.float32(4.0)
    two_f32 = np.float32(2.0)
    # one_f32 = np.float32(1.0) # Not strictly needed if DC is handled by pass then zeroing
    zero_c64 = np.complex64(0.0) # Equivalent to 0.0 + 0.0j for complex64

    # Safeguard threshold, also as float32
    epsilon_f32 = np.float32(1e-7)

    for i in prange(h):
        # np.cos on a float32 scalar input will produce a float32 scalar in Numba
        cos_ky = np.cos(ky_fft_freqs_scaled[i])
        for j in range(w_rfft):
            if i == 0 and j == 0:
                # DC component: Handled by explicit zeroing after this function.
                # No operation needed here if laplacian_val would be 1.0 and then overwritten.
                pass
            else:
                cos_kx = np.cos(kx_rfft_freqs_scaled[j])
                laplacian_val = (four_f32 - two_f32 * cos_kx - two_f32 * cos_ky)

                # Check for laplacian_val being too close to zero
                if abs(laplacian_val) < epsilon_f32:
                     F_div_g_rfft[i, j] = zero_c64 # Set to zero to avoid large numbers/NaN
                else:
                    F_div_g_rfft[i, j] = F_div_g_rfft[i, j] / laplacian_val
    # Note: Modification happens in-place on F_div_g_rfft

@numba.njit(parallel=True, cache=True, nogil=True)
def _compute_u_rfft_with_brightness_inplace(F_div_g_rfft, kx_rfft_freqs_scaled, ky_fft_freqs_scaled, lam, u0, h, w):
    """
    Computes (F_div_g_rfft + lam * u0_fft) / (lam + laplacian) using RFFT frequencies, 
    modifying F_div_g_rfft (complex64) in place. Handles brightness constraint.
    """
    h_freq = ky_fft_freqs_scaled.shape[0]
    w_rfft = kx_rfft_freqs_scaled.shape[0]

    # Define constants explicitly with np.float32
    four_f32 = np.float32(4.0)
    two_f32 = np.float32(2.0)
    zero_c64 = np.complex64(0.0)
    epsilon_f32 = np.float32(1e-7)
    
    # Convert parameters to float32
    lam_f32 = np.float32(lam)
    u0_f32 = np.float32(u0)
    h_f32 = np.float32(h)
    w_f32 = np.float32(w)

    for i in prange(h_freq):
        cos_ky = np.cos(ky_fft_freqs_scaled[i])
        for j in range(w_rfft):
            if i == 0 and j == 0:
                # DC component: (div_g_fft[0,0] + lam * u0 * h * w) / (lam + 0)
                # Since laplacian[0,0] = 0, denom = lam
                F_div_g_rfft[i, j] = (F_div_g_rfft[i, j] + lam_f32 * u0_f32 * h_f32 * w_f32) / lam_f32
            else:
                cos_kx = np.cos(kx_rfft_freqs_scaled[j])
                laplacian_val = (four_f32 - two_f32 * cos_kx - two_f32 * cos_ky)
                denom = lam_f32 + laplacian_val

                # Check for denominator being too close to zero
                if abs(denom) < epsilon_f32:
                    F_div_g_rfft[i, j] = zero_c64
                else:
                    F_div_g_rfft[i, j] = F_div_g_rfft[i, j] / denom

@numba.njit(parallel=True, cache=True, nogil=True)
def _calculate_divergence_numba(grad_x, grad_y, div_g_out):
    """
    Calculates divergence and stores it in the pre-allocated div_g_out.
    div_g_out must be zero-initialized before calling this function.
    grad_x, grad_y, and div_g_out should have the same float dtype (e.g., np.float32).
    """
    h, w = grad_x.shape

    # Loop order (prange on outer loop) is chosen for good parallel efficiency.
    for i in prange(h): # Parallelize over rows
        for j in range(w):
            # Calculate d(gx)/dx component using backward difference
            term_dx = np.float32(0.0) # Initialize with the correct dtype
            if w == 1: # Special case: width is 1
                term_dx = grad_x[i, 0]
            elif j == 0: # Left boundary (j=0)
                term_dx = grad_x[i, 0]
            else: # Internal points for dx (j > 0)
                term_dx = grad_x[i, j] - grad_x[i, j-1]

            # Calculate d(gy)/dy component using backward difference
            term_dy = np.float32(0.0) # Initialize with the correct dtype
            if h == 1: # Special case: height is 1
                term_dy = grad_y[0, j]
            elif i == 0: # Top boundary (i=0)
                term_dy = grad_y[0, j]
            else: # Internal points for dy (i > 0)
                term_dy = grad_y[i, j] - grad_y[i-1, j]
            
            div_g_out[i, j] = term_dx + term_dy

@numba.njit(parallel=True, cache=True, nogil=True)
def _calculate_divergence_numba_x(grad_x, div_g_out):
    h, w = grad_x.shape

    # Loop order (prange on outer loop) is chosen for good parallel efficiency.
    for i in prange(h): # Parallelize over rows
        for j in range(w):
            # Calculate d(gx)/dx component using backward difference
            term_dx = np.float32(0.0) # Initialize with the correct dtype
            if w == 1: # Special case: width is 1
                term_dx = grad_x[i, 0]
            elif j == 0: # Left boundary (j=0)
                term_dx = grad_x[i, 0]
            else: # Internal points for dx (j > 0)
                term_dx = grad_x[i, j] - grad_x[i, j-1]

            div_g_out[i, j] += term_dx

@numba.njit(parallel=True, cache=True, nogil=True)
def _calculate_divergence_numba_y(grad_y, div_g_out):
    h, w = grad_y.shape

    for i in prange(h): # Parallelize over rows
        for j in range(w):
            # Calculate d(gy)/dy component using backward difference
            term_dy = np.float32(0.0) # Initialize with the correct dtype
            if h == 1: # Special case: height is 1
                term_dy = grad_y[0, j]
            elif i == 0: # Top boundary (i=0)
                term_dy = grad_y[0, j]
            else: # Internal points for dy (i > 0)
                term_dy = grad_y[i, j] - grad_y[i-1, j]
                
            div_g_out[i, j] += term_dy

            
def poisson_solver(grad_x, grad_y, u0=1.0, lam=1e-3):
    """
    Solve Poisson equation using FFT with brightness constraint.
    Optimized for MEMORY and multi-core performance using:
    - float32 data type throughout
    - rfft2/irfft2 for real inputs (reduces frequency domain size)
    - pyfftw for multi-threaded FFTs
    - Numba for parallelized division in frequency space (avoids large Laplacian array)
    
    Args:
        grad_x, grad_y: Input gradients (dense or sparse matrices)
        u0: Target average brightness (default: 1.0)
        lam: Regularization parameter for brightness constraint (default: 1e-3)
    
    Supports both dense and sparse matrix inputs.
    """
    start_time = time.time()

    h, w = grad_x.shape
    dtype = np.float32  # Use float16 throughout
    complex_dtype = np.complex64

    # print(f"Input shape: {h}x{w}, Using data type: {dtype}")

    # --- Compute Divergence ---
    divergence_start = time.time()
    # Ensure calculations stay in float16
    div_g = np.zeros((h, w), dtype=dtype)
    
    # Handle sparse inputs
    if sparse.issparse(grad_x):
        # Process x gradients
        grad_x_f32 = grad_x[:, :-1].toarray().astype(dtype, copy=False)
        _calculate_divergence_numba_x(grad_x_f32, div_g)
        del grad_x, grad_x_f32
        
        # Process y gradients
        grad_y_f32 = grad_y[:-1, :].toarray().astype(dtype, copy=False)
        _calculate_divergence_numba_y(grad_y_f32, div_g)
        del grad_y, grad_y_f32
    else:
        # Process both gradients at once for dense inputs
        grad_x_f32 = grad_x[:, :-1].astype(dtype, copy=False)
        grad_y_f32 = grad_y[:-1, :].astype(dtype, copy=False)
        _calculate_divergence_numba(grad_x_f32, grad_y_f32, div_g)
        del grad_x, grad_y, grad_x_f32, grad_y_f32

    # print(f"Divergence calculation took {time.time() - divergence_start:.3f} seconds")

    # --- Frequency Domain Calculations ---
    freq_calc_start = time.time()
    # Use rfftfreq for width (axis where rfft reduces size)
    # Use fftfreq for height
    # Scale by 2*pi here and ensure correct dtype
    kx_rfft_freqs_scaled = (pyfftw_fft.rfftfreq(w, d=1.0) * (2.0 * np.pi)).astype(dtype)
    ky_fft_freqs_scaled = (pyfftw_fft.fftfreq(h, d=1.0) * (2.0 * np.pi)).astype(dtype)
    # print(f"Frequency calculations took {time.time() - freq_calc_start:.3f} seconds")

    # --- Forward FFT (Real to Complex Half-Domain) ---
    fft_start = time.time()
    # Use pyfftw.interfaces.numpy_fft (threaded, NumPy-compatible)
    fft_output_buffer = pyfftw_fft.rfft2(div_g, axes=(0, 1))
    del div_g
    # print(f"Forward FFT took {time.time() - fft_start:.3f} seconds")

    # --- Division by Laplacian in Frequency Space (using Numba) ---
    laplacian_start = time.time()
    
    # Modifies u_rfft (fft_output_buffer) in-place with brightness constraint
    _compute_u_rfft_with_brightness_inplace(fft_output_buffer, kx_rfft_freqs_scaled, ky_fft_freqs_scaled, lam, u0, h, w)
    del kx_rfft_freqs_scaled, ky_fft_freqs_scaled  # Free frequency arrays

    # Save FFT debug images
    # save_fft_debug_images(fft_output_buffer, h, w)

    # print(f"Laplacian division with brightness constraint took {time.time() - laplacian_start:.3f} seconds")

    # --- Inverse FFT (Complex Half-Domain to Real) ---
    ifft_start = time.time()
    ifft_output_buffer = pyfftw_fft.irfft2(fft_output_buffer, s=(h, w), axes=(0, 1))
    del fft_output_buffer
    # print(f"Inverse FFT took {time.time() - ifft_start:.3f} seconds")

    # print(f"Total time: {time.time() - start_time:.3f} seconds")
    return ifft_output_buffer

def poisson_solver_with_brightness(grad_x, grad_y, u0=1.0, lam=1e-3):
    """
    Convenience function that matches the original poisson_solver_with_brightness interface.
    Calls the optimized poisson_solver with brightness constraint.
    """
    return poisson_solver(grad_x, grad_y, u0, lam)

def save_fft_debug_images(fft_output_buffer, h, w, output_dir="../debug/fft"):
    """
    Save FFT output buffer as debug images showing magnitude and phase.
    
    Args:
        fft_output_buffer: Complex FFT output buffer
        h, w: Height and width of original image
        output_dir: Directory to save debug images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate magnitude and phase
    magnitude = np.abs(fft_output_buffer)
    phase = np.angle(fft_output_buffer)
    
    # Normalize magnitude for visualization (log scale for better visibility)
    magnitude_log = np.log10(magnitude + 1e-10)  # Add small value to avoid log(0)
    
    # Create figure with subplots - show full FFT region
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot magnitude (log scale) - full region
    im1 = axes[0].imshow(magnitude_log, cmap='viridis', aspect='auto')
    axes[0].set_title('FFT Magnitude (log scale) - Full Region')
    axes[0].set_xlabel('Frequency (x)')
    axes[0].set_ylabel('Frequency (y)')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot phase - full region
    im2 = axes[1].imshow(phase, cmap='twilight', aspect='auto')
    axes[1].set_title('FFT Phase - Full Region')
    axes[1].set_xlabel('Frequency (x)')
    axes[1].set_ylabel('Frequency (y)')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    
    # Save the combined image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    debug_filename = os.path.join(output_dir, f"fft_debug_{h}x{w}_{timestamp}.png")
    plt.savefig(debug_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save individual components as numpy arrays
    np.save(os.path.join(output_dir, f"fft_magnitude_{h}x{w}_{timestamp}.npy"), magnitude)
    np.save(os.path.join(output_dir, f"fft_phase_{h}x{w}_{timestamp}.npy"), phase)
    np.save(os.path.join(output_dir, f"fft_complex_{h}x{w}_{timestamp}.npy"), fft_output_buffer)
    
    print(f"FFT debug images saved to: {debug_filename}")
    print(f"FFT data arrays saved to: {output_dir}/")
    
    return debug_filename