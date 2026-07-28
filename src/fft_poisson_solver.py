import os
os.environ["MKL_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"
import numpy as np
import pprint
from scipy.fft import dst, idst, dct, idct
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve

def poisson_solver(grad_x, grad_y):
    """
    Solve Poisson equation given gradients with Dirichlet boundary conditions (u = 0 at the edges)
    """
    h, w = grad_x.shape
    
    # Compute divergence (div g = d(g_x)/dx + d(g_y)/dy)
    div_g = np.zeros((h, w))
    div_g[:, :-1] += grad_x[:, :-1]  # d(g_x)/dx (forward difference)
    div_g[:, 1:] -= grad_x[:, :-1]

    div_g[:-1, :] += grad_y[:-1, :]  # d(g_y)/dy (forward difference)
    div_g[1:, :] -= grad_y[:-1, :]

    # Fourier space solution
    kx = np.fft.fftfreq(w) * 2 * np.pi
    ky = np.fft.fftfreq(h) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    laplacian = (4 - 2 * np.cos(KX) - 2 * np.cos(KY))
    laplacian[0, 0] = 1  # Avoid division by zero at DC component

    # Solve in Fourier domain
    u_fft = np.fft.fft2(div_g) / laplacian
    u_fft[0, 0] = 0  # Ensure mean is zero (Dirichlet boundary condition)

    # Inverse FFT to get result
    u = np.fft.ifft2(u_fft).real
    return u


def poisson_solver_with_brightness(grad_x, grad_y, u0=1.0, lam=1e-3):
    h, w = grad_x.shape
    div_g = np.zeros((h, w))
    div_g[:, :-1] += grad_x[:, :-1]
    div_g[:, 1:]  -= grad_x[:, :-1]
    div_g[:-1, :] += grad_y[:-1, :]
    div_g[1:, :]  -= grad_y[:-1, :]

    kx = np.fft.fftfreq(w) * 2 * np.pi
    ky = np.fft.fftfreq(h) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    laplacian = (4 - 2*np.cos(KX) - 2*np.cos(KY))

    # 避免除零，0频加lambda
    denom = lam + laplacian
    denom[0,0] = 1  # 避免除零

    div_g_fft = np.fft.fft2(div_g)
    u0_fft = np.fft.fft2(np.full((h, w), u0))

    u_fft = (div_g_fft + lam * u0_fft) / denom
    u_fft[0,0] = u0 * h * w  # 保证平均亮度等于u0

    u = np.fft.ifft2(u_fft).real

    return u


def poisson_solver_fixed_boundary(grad_x, grad_y, u0=1.0):
    h, w = grad_x.shape
    
    # --- Step 1: Compute Fixed Boundaries via Integration ---
    # Initialize the image with zeros
    u = np.zeros((h, w))
    u[0, 0] = u0

    # Integrate Top Edge (y=0)
    u[0, 1:] = u0 + np.cumsum(grad_x[0, :-1])
    
    # Integrate Left Edge (x=0)
    u[1:, 0] = u0 + np.cumsum(grad_y[:-1, 0])
    
    # Integrate Bottom Edge (y=h-1) from Bottom-Left
    u[h-1, 1:] = u[h-1, 0] + np.cumsum(grad_x[h-1, :-1])
    
    # Integrate Right Edge (x=w-1) from Top-Right
    u[1:, w-1] = u[0, w-1] + np.cumsum(grad_y[:-1, w-1])

    # --- Step 2: Compute Divergence ---
    # This calculation creates (u_neigh - 2u) logic
    div_g = np.zeros((h, w))
    div_g[:, :-1] += grad_x[:, :-1]
    div_g[:, 1:]  -= grad_x[:, :-1]
    div_g[:-1, :] += grad_y[:-1, :]
    div_g[1:, :]  -= grad_y[:-1, :]
    
    # Extract ONLY the interior divergence (size H-2, W-2)
    # The Laplacian Operator "4u - neighbors" equals NEGATIVE divergence
    # So we flip the sign of div_g here to match the solver
    rhs = -div_g[1:-1, 1:-1]

    # --- Step 3: Inject Boundary Conditions ---
    # The equation is: (4u - neighbors) = rhs
    # If a neighbor is on the boundary, we move it to the RHS.
    
    # Top boundary (y=0) affects the first row of the interior
    rhs[0, :] += u[0, 1:-1]
    # Bottom boundary (y=h-1) affects the last row of the interior
    rhs[-1, :] += u[-1, 1:-1]
    # Left boundary (x=0) affects the first col of the interior
    rhs[:, 0] += u[1:-1, 0]
    # Right boundary (x=w-1) affects the last col of the interior
    rhs[:, -1] += u[1:-1, -1]

    # --- Step 4: Solve Interior using DST-I ---
    h_inner, w_inner = rhs.shape
    
    # Forward DST (Type 1) over rows and columns
    # scipy.fft.dst is unnormalized
    u_dst = dst(dst(rhs, type=1, axis=1), type=1, axis=0)

    # Create Eigenvalue Grid for DST-I
    # Eigenvalues: 2 - 2cos(pi * k / (N+1))
    xx = np.arange(1, w_inner + 1)
    yy = np.arange(1, h_inner + 1)
    KX, KY = np.meshgrid(xx, yy)
    
    denom = (2 - 2*np.cos(np.pi * KX / (w_inner + 1))) + \
            (2 - 2*np.cos(np.pi * KY / (h_inner + 1)))
            
    # Divide by eigenvalues
    u_dst = u_dst / denom

    # Inverse DST (Type 1)
    u_inner = idst(idst(u_dst, type=1, axis=0), type=1, axis=1)
    
    # --- Step 5: Normalize and Fill ---
    # Scipy DST-I/IDST-I are unnormalized.
    # The total scaling factor for 2D DST+IDST is 2(W+1) * 2(H+1)
    scale_factor = (2 * (w_inner + 1)) * (2 * (h_inner + 1))
    u_inner = u_inner / scale_factor

    # Fill the interior
    u[1:-1, 1:-1] = u_inner

    return u


def poisson_solver_neumann(grad_x, grad_y, u0=0.0):
    h, w = grad_x.shape
    
    # --- Step 1: Compute Divergence (The "Source" of the field) ---
    # In a Neumann solver, we carefully handle the image boundaries 
    # to ensure the "flux" leaving the image is zero (Reflection).
    
    div_g = np.zeros((h, w))
    
    # Calculate standard divergence
    # grad_x is forward difference: g_x[x] = u[x+1] - u[x]
    # Divergence is backward difference of the gradients
    
    # X-axis contributions
    div_g[:, :-1] += grad_x[:, :-1]
    div_g[:, 1:]  -= grad_x[:, :-1]
    
    # Y-axis contributions
    div_g[:-1, :] += grad_y[:-1, :]
    div_g[1:, :]  -= grad_y[:-1, :]

    # --- Step 2: Prepare DCT Eigenvalues ---
    # We use DCT-II (The standard "DCT"), which implies Neumann (zero-slope) 
    # boundary conditions at the outer edges of the image.
    
    cx = np.arange(w)
    cy = np.arange(h)
    CX, CY = np.meshgrid(cx, cy)
    
    # Eigenvalues for 2D DCT-II: 2 - 2*cos(pi*k / N)
    # This corresponds to the Laplacian operator in the DCT domain
    denom = (2 - 2 * np.cos(np.pi * CX / w)) + \
            (2 - 2 * np.cos(np.pi * CY / h))
            
    # Avoid division by zero at the DC component (frequency 0,0)
    denom[0, 0] = 1.0 

    # --- Step 3: Solve in Frequency Domain ---
    
    # Forward DCT (Type 2, Orthogonal normalization is easier to reason about)
    rhs_dct = dct(dct(div_g, type=2, axis=1, norm='ortho'), type=2, axis=0, norm='ortho')
    
    # Divide by the Laplacian Eigenvalues (Invert the Laplacian)
    # The Laplacian is negative, divergence is "u_neigh - 4u", so we divide directly.
    # Note: If your divergence definition is "4u - u_neigh", multiply rhs by -1.
    # Standard divergence usually implies negative Laplacian eigenvalues.
    u_dct = rhs_dct / (-denom) 
    
    # --- Step 4: Handle the DC Component (Brightness) ---
    # The (0,0) frequency represents the sum/average of the image.
    # The Laplacian kills the average (derivative of a constant is 0).
    # We must restore it manually using u0.
    
    # In 'ortho' norm, DC component = sum(u) / sqrt(N).
    # We want average value to be u0.
    # Total Sum = u0 * h * w
    # DC_val = Total Sum / sqrt(h*w) = u0 * sqrt(h*w)
    u_dct[0, 0] = u0 * np.sqrt(h * w)

    # --- Step 5: Inverse DCT ---
    u = idct(idct(u_dct, type=2, axis=0, norm='ortho'), type=2, axis=1, norm='ortho')

    return u

# def poisson_solver_with_bright_lowfreq(grad_x, grad_y, u0=1.0, lam=1e-3, lowfreq_radius=10, lowfreq_factor=1e3):
#     h, w = grad_x.shape
    
#     # 计算散度 div_g
#     div_g = np.zeros((h, w))
#     div_g[:, :-1] += grad_x[:, :-1]
#     div_g[:, 1:]  -= grad_x[:, :-1]
#     div_g[:-1, :] += grad_y[:-1, :]
#     div_g[1:, :]  -= grad_y[:-1, :]
    
#     # 频域坐标
#     kx = np.fft.fftfreq(w) * 2 * np.pi
#     ky = np.fft.fftfreq(h) * 2 * np.pi
#     KX, KY = np.meshgrid(kx, ky)
    
#     laplacian = 4 - 2*np.cos(KX) - 2*np.cos(KY)
#     denom = lam + laplacian
#     denom[0, 0] = 1  # 避免除零
    
#     div_g_fft = np.fft.fft2(div_g)
#     u0_fft = np.fft.fft2(np.full((h, w), u0))
    
#     u_fft = (div_g_fft + lam * u0_fft) / denom
    
#     # 制造低频放大掩码，包括直流（中心位置）
#     cy, cx = h // 2, w // 2
#     y = np.arange(h) - cy
#     x = np.arange(w) - cx
#     X, Y = np.meshgrid(x, y)
#     dist = np.sqrt(X**2 + Y**2)
#     mask = (dist <= lowfreq_radius).astype(float)
#     mask = np.fft.ifftshift(mask)  # 转换频率原点到角落
    
#     # 低频放大：包含直流频率
#     u_fft = u_fft * (1 + (lowfreq_factor - 1) * mask)
    
#     # 调整直流分量让最终均值为desired_mean*h*w
#     u_fft[0, 0] = u0 * h * w
    
#     # 逆变换
#     u = np.fft.ifft2(u_fft).real
    
#     return u

def gaussian_lowfreq_mask(h, w, sigma):
    cy, cx = h // 2, w // 2
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    X, Y = np.meshgrid(x, y)
    dist_sq = X**2 + Y**2
    g = np.exp(-dist_sq / (2 * sigma ** 2))
    mask = np.fft.ifftshift(g)
    return mask

def poisson_solver_reduce_lowfreq(grad_x, grad_y, lowfreq_radius=100, reduction_factor=1e-4):
    """
    Solve Poisson equation with Dirichlet BC,
    reduce low-frequency components in frequency domain.
    
    Args:
        grad_x, grad_y: input gradients
        lowfreq_radius: 低频圆形半径，单位为频域坐标像素
        reduction_factor: 低频幅值缩减比例,0代表完全消除,1代表不缩减
    """
    h, w = grad_x.shape
    
    # 计算梯度散度
    div_g = np.zeros((h, w))
    div_g[:, :-1] += grad_x[:, :-1]
    div_g[:, 1:]  -= grad_x[:, :-1]
    div_g[:-1, :] += grad_y[:-1, :]
    div_g[1:, :]  -= grad_y[:-1, :]
    
    # 傅里叶坐标
    kx = np.fft.fftfreq(w) * 2 * np.pi
    ky = np.fft.fftfreq(h) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    
    laplacian = (4 - 2 * np.cos(KX) - 2 * np.cos(KY))
    laplacian[0, 0] = 1  # 避免除零
    
    div_g_fft = np.fft.fft2(div_g)
    u_fft = div_g_fft / laplacian
    dc = u_fft[0, 0]
    
    # 构造低频掩码
    cy, cx = h // 2, w // 2
    y = np.arange(h) - cy
    x = np.arange(w) - cx
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X**2 + Y**2)
    
    mask = (dist <= lowfreq_radius).astype(np.float64)
    mask = np.fft.ifftshift(mask)  # 将0频点移至左上角，与FFT输出对应

    # sigma = lowfreq_radius / 3
    # mask = gaussian_lowfreq_mask(h, w, sigma)
    
    
    # 对低频部分做幅度缩减
    u_fft = u_fft * (1 - mask * (1 - reduction_factor))
    
    # 保证直流分量为0 (你的边界条件要求)
    u_fft[0, 0] = dc
    
    u = np.fft.ifft2(u_fft).real
    
    return u

def screened_poisson_solver_dct(grad_x, grad_y, lam=1e-4):
    """
    Solves (Laplacian - lambda) * I = div(G) using DCT.
    
    Args:
        grad_x, grad_y: Gradients from events.
        lam: Screening parameter (lambda). 
             lam = 0 -> Pure integration (susceptible to drift).
             lam > 0 -> Damps low frequencies, removes 'cloudy' drift.
                        Try values like 1e-4, 1e-3, 1e-2.
    """
    h, w = grad_x.shape
    
    # 1. Compute Divergence
    # 为了保持 Neumann 边界的一致性，边缘处的梯度处理要小心
    # 这里使用简单的中心差分或前向差分近似
    div_g = np.zeros((h, w))
    div_g[:, :-1] += grad_x[:, :-1]
    div_g[:, 1:]  -= grad_x[:, :-1]
    div_g[:-1, :] += grad_y[:-1, :]
    div_g[1:, :]  -= grad_y[:-1, :]

    # 2. Forward DCT (Type II)
    rho = dct(dct(div_g, axis=0, norm='ortho'), axis=1, norm='ortho')

    # 3. Compute Eigenvalues of Laplacian - lambda
    kx = np.arange(w)
    ky = np.arange(h)
    KX, KY = np.meshgrid(kx, ky)
    
    # Laplacian eigenvalues for DCT
    eigen_laplacian = (2 * np.cos(np.pi * KX / w) - 2) + \
                      (2 * np.cos(np.pi * KY / h) - 2)
    
    # Add screening parameter (This is the magic fix for low-freq drift)
    # The equation is (Laplacian - lambda) * I = div_g
    # So in frequency domain: (eig - lambda) * I_hat = rho
    denominator = eigen_laplacian - lam

    # 4. Solve
    phi = rho / denominator

    # 5. Inverse DCT
    img = idct(idct(phi, axis=1, norm='ortho'), axis=0, norm='ortho')
    
    return img


def poisson_solver_dct(grad_x, grad_y):
    """
    Solve Poisson equation using DCT (Neumann Boundary Conditions).
    This is usually much better for image reconstruction than FFT.
    """
    h, w = grad_x.shape
    
    # 1. Compute Divergence
    # 注意：DCT 的 divergence 计算通常需要稍微不同的处理，或者直接处理 laplacian
    # 这里为了简单，我们计算 div_g，但在边界处需要小心
    div_g = np.zeros((h, w))
    
    # 计算散度 (Divergence)
    div_g[:, :-1] += grad_x[:, :-1]
    div_g[:, 1:]  -= grad_x[:, :-1]
    div_g[:-1, :] += grad_y[:-1, :]
    div_g[1:, :]  -= grad_y[:-1, :]

    # 2. 2D DCT (Type II is standard)
    # Norm='ortho' makes it unitary, simplifying the inverse
    rho = dct(dct(div_g, axis=0, norm='ortho'), axis=1, norm='ortho')

    # 3. Eigenvalues of the Laplacian matrix for DCT
    # DCT 的特征值公式与 FFT 不同，是 2*cos(pi*k/N) - 2
    kx = np.arange(w)
    ky = np.arange(h)
    KX, KY = np.meshgrid(kx, ky)
    
    # Eigenvalues for DCT-II
    # Lambda_x = 2 * cos(pi * k / N) - 2
    eigenvalues = (2 * np.cos(np.pi * KX / w) - 2) + \
                  (2 * np.cos(np.pi * KY / h) - 2)

    # 4. Inverse Laplacian (Avoid division by zero at DC)
    eigenvalues[0, 0] = 1.0  # Temporary to avoid Inf
    phi = rho / eigenvalues
    phi[0, 0] = 0.0          # Set mean to 0 (unknown DC component)

    # 5. Inverse 2D DCT
    img = idct(idct(phi, axis=1, norm='ortho'), axis=0, norm='ortho')
    
    return img

import numpy as np

def tv_l1_reconstruction(grad_x, grad_y, lambda_tv=1.0, n_iters=2000, theta=1.0):
    """
    使用 Chambolle-Pock 算法进行 TV-L1 重建。
    
    效果：
    1. 在有梯度的地方：严格遵守梯度。
    2. 在没梯度的地方：绝对平坦（等于最近的邻居），而不是平滑过渡。
    3. 彻底消除 Ripple（波纹）。
    
    Args:
        lambda_tv: 平滑项权重。
                   越小 -> 越相信输入的梯度（细节多）。
                   越大 -> 越倾向于把图像抹平（卡通画效果）。
                   建议范围: 0.5 到 2.0。
    """
    h, w = grad_x.shape
    
    # 初始化变量
    u = np.zeros((h, w), dtype=np.float32)      # 重建图像 (Primal variable)
    p = np.zeros((h, w, 2), dtype=np.float32)   # 对偶变量 (Dual variable)
    u_bar = np.zeros_like(u)                    # 临时变量
    
    # 步长参数 (对于 TV-L1 模型通常取这些值以保证收敛)
    L2 = 8.0
    tau = 1.0 / np.sqrt(L2)
    sigma = 1.0 / np.sqrt(L2)
    
    # 这里的 Data Term 是: || grad(u) - G ||_1
    # 这是一个特殊的 TV 重建形式，我们希望 u 的梯度接近 G
    
    for i in tqdm(range(n_iters)):
        # --- 1. 对偶步 (Dual Update) ---
        # 计算 u_bar 的梯度
        # 使用前向差分
        grad_u_x = np.roll(u_bar, -1, axis=1) - u_bar
        grad_u_y = np.roll(u_bar, -1, axis=0) - u_bar
        
        # 边界处理 (Neumann)
        grad_u_x[:, -1] = 0
        grad_u_y[-1, :] = 0
        
        # 更新 p
        # p = p + sigma * (Gradient(u_bar) - G_observed)
        p[:,:,0] += sigma * (grad_u_x - grad_x)
        p[:,:,1] += sigma * (grad_u_y - grad_y)
        
        # 投影到单位球 (Projection onto unit ball) - 这是 TV 正则化的核心
        # 我们限制 p 的模长不超过 lambda_tv
        norm_p = np.sqrt(p[:,:,0]**2 + p[:,:,1]**2)
        norm_p = np.maximum(1.0, norm_p / lambda_tv) # 避免除以0
        
        p[:,:,0] /= norm_p
        p[:,:,1] /= norm_p
        
        # --- 2. 原始步 (Primal Update) ---
        u_old = u.copy()
        
        # 计算散度 div(p)
        # 使用后向差分 (对应前向梯度的伴随算子)
        div_p = np.zeros_like(u)
        
        # x方向散度
        div_p[:, 1:-1] += p[:, 1:-1, 0] - p[:, :-2, 0]
        div_p[:, 0]    += p[:, 0, 0]    # 边界
        div_p[:, -1]   -= p[:, -2, 0]   # 边界
        
        # y方向散度
        div_p[1:-1, :] += p[1:-1, :, 1] - p[:-2, :, 1]
        div_p[0, :]    += p[0, :, 1]
        div_p[-1, :]   -= p[-2, :, 1]
        
        # 更新 u
        u = u + tau * div_p
        
        # --- 3. 动量步 (Relaxation) ---
        u_bar = u + theta * (u - u_old)
        
        if i % 100 == 0:
            # 简单的收敛监控
            change = np.mean(np.abs(u - u_old))
            # print(f"Iter {i}, Change: {change}")
            if change < 1e-6:
                break
                
    return u


def weighted_poisson_reconstruction(grad_x, grad_y, alpha=10.0, min_weight=1e-4, lam=1e-5):
    """
    Weighted Poisson Reconstruction (Anisotropic Diffusion).
    
    Args:
        grad_x, grad_y: Input gradients.
        alpha: Edge sensitivity. 
               alpha 越大 -> 对边缘越敏感（权重下降得越快），边缘越锐利。
               alpha 越小 -> 越像普通泊松（平滑）。
        min_weight: 最小权重 (防止矩阵奇异/图不连通)。
        lam: Screening 参数 (防止整体漂移，类似 u0=0 的约束)。
    """
    h, w = grad_x.shape
    n = h * w
    
    # --- 1. 计算边缘权重 (Edge Weights) ---
    # 计算梯度的模长
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # 核心公式: w = exp(-alpha * |G|)
    # 梯度越大，权重越接近 0。
    weights = np.exp(-alpha * grad_mag)
    
    # 保证最小权重，防止完全断开连接导致数值错误
    weights = np.maximum(weights, min_weight)
    
    # 我们定义 grid 之间的连接权重。
    # wx[i, j] 是 (i,j) 和 (i,j+1) 之间的权重
    # wy[i, j] 是 (i,j) 和 (i+1,j) 之间的权重
    # 为了简化，我们直接取两个像素权重的平均值，或者直接用中心像素权重
    
    # 这里使用简单的中心近似法：
    # 右边的连接权重
    wx = 0.5 * (weights[:, :-1] + weights[:, 1:])
    # 下边的连接权重
    wy = 0.5 * (weights[:-1, :] + weights[1:, :])
    
    # --- 2. 构建拉普拉斯稀疏矩阵 (Construct Laplacian) ---
    # 我们需要构建一个 (N, N) 的大矩阵，N = h * w
    # 这是一个 5-point stencil (中心, 上, 下, 左, 右)
    
    # 展平权重以便构建对角线
    # X方向连接 (Horizontal connections)
    # 我们需要构造偏移量为 1 的对角线
    wx_flat = wx.flatten()
    
    # Y方向连接 (Vertical connections)
    # 我们需要构造偏移量为 w 的对角线
    wy_flat = wy.flatten()
    
    # 构造 Diagonals
    # D_left: 从中心向左连 (-wx)
    # D_right: 从中心向右连 (-wx)
    # D_up: 从中心向上连 (-wy)
    # D_down: 从中心向下连 (-wy)
    
    # 注意 scipy.sparse.diags 的偏移逻辑
    # 我们构建 Upper triangular 部分，利用对称性
    
    # 构造主对角线 (Main Diagonal): sum of all outgoing weights
    # 为了方便，我们先创建一个全零的图，然后把权重加进去
    
    # 使用 COO 格式构建矩阵更直观
    row_idx = []
    col_idx = []
    data = []
    
    # --- Vectorized Matrix Construction ---
    # 这是一个稍微硬核的构建过程，但速度很快
    
    # 1. Horizontal Edges (Right neighbors)
    # Pixel (i, j) connects to (i, j+1) with weight wx[i,j]
    # Indices in flattened array:
    # u[y, x]   -> idx = y*w + x
    # u[y, x+1] -> idx = y*w + x + 1
    
    # Grid coordinates
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Valid horizontal links (excluding last column)
    valid_x = X[:, :-1].flatten()
    valid_y = Y[:, :-1].flatten()
    idx_curr = valid_y * w + valid_x
    idx_right = idx_curr + 1
    w_vals = wx.flatten()
    
    # Add (curr, right) and (right, curr)
    # Off-diagonals are negative
    row_idx.extend(idx_curr)
    col_idx.extend(idx_right)
    data.extend(-w_vals)
    
    row_idx.extend(idx_right)
    col_idx.extend(idx_curr)
    data.extend(-w_vals)
    
    # Add to diagonal (positive)
    # We will accumulate diagonal later using np.bincount or similar, 
    # but constructing explicit lists is easier for now
    
    # 2. Vertical Edges (Down neighbors)
    valid_x = X[:-1, :].flatten()
    valid_y = Y[:-1, :].flatten()
    idx_curr = valid_y * w + valid_x
    idx_down = idx_curr + w
    w_vals = wy.flatten()
    
    row_idx.extend(idx_curr)
    col_idx.extend(idx_down)
    data.extend(-w_vals)
    
    row_idx.extend(idx_down)
    col_idx.extend(idx_curr)
    data.extend(-w_vals)
    
    # 3. Create the sparse matrix (structure only, no main diagonal yet)
    # size N x N
    L_off = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n, n))
    
    # 4. Main Diagonal
    # The main diagonal is simply the negative sum of off-diagonal elements in each row
    # Because Laplacian row sum should be 0 (for pure Neumann)
    diagonal_values = -np.array(L_off.sum(axis=1)).flatten()
    
    # Add screening / weak regularization (prevents drift)
    diagonal_values += lam
    
    L_diag = sparse.diags(diagonal_values)
    
    # Final Laplacian Matrix
    A = L_diag + L_off
    
    # --- 3. 构建右端项 (RHS: Weighted Divergence) ---
    # div(w * G) = d(w*Gx)/dx + d(w*Gy)/dy
    # 这里需要非常小心，离散格式必须和矩阵A完全对应（伴随算子）
    
    # b[i] = w_right * Gx_right - w_left * Gx_left + ...
    # 这种思考太复杂。
    # 更简单的方法：RHS 是 梯度 G 加权后的 散度。
    # 对应于矩阵构建：
    # 如果我们把问题看作最小化能量 E = sum w * (du - g)^2
    # 对应的线性方程是 L * u = Div_Weighted_G
    
    # 手动计算加权散度:
    b = np.zeros(n)
    
    # Horizontal contribution
    # For pixel (y, x), contribution is:
    # + wx[y, x] * gx[y, x]      (flux leaving to right)
    # - wx[y, x-1] * gx[y, x-1]  (flux entering from left)
    
    # Let's vectorize
    # Weighted gradients at the edges
    WGx = wx * grad_x[:, :-1] # shape (h, w-1)
    WGy = wy * grad_y[:-1, :] # shape (h-1, w)
    
    # Map to flattened vector b
    b_mat = np.zeros((h, w))
    
    # Divergence x
    b_mat[:, :-1] += WGx  # contribution from right edge
    b_mat[:, 1:]  -= WGx  # contribution from left edge
    
    # Divergence y
    b_mat[:-1, :] += WGy
    b_mat[1:, :]  -= WGy
    
    b = b_mat.flatten()
    
    # --- 4. 求解 ---
    # 使用直接求解器 (Direct Solver) 通常对 2D 图像够快
    print("Solving weighted poisson system...")
    u_flat = spsolve(A.tocsr(), b)
    
    u = u_flat.reshape((h, w))
    
    return u