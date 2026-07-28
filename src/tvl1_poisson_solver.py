import torch
import torch.nn.functional as F
import numpy as np

def tv_l1_reconstruction_cuda(grad_x, grad_y, lambda_tv=1.0, n_iters=2000, theta=1.0, L=8.0, device='cuda'):
    """
    TV-L1 Reconstruction using Primal-Dual Hybrid Gradient (Chambolle-Pock).
    PyTorch + CUDA implementation.
    
    Minimizes: || grad(u) - G ||_1
    
    Args:
        grad_x, grad_y: Input gradient tensors (H, W).
        lambda_tv: Regularization strength. 
                   Controls strictness of the gradient constraint.
                   Typical values: 0.5 - 2.0.
        n_iters: Number of iterations.
        device: 'cuda' or 'cpu'.
        
    Returns:
        u: Reconstructed image (H, W).
    """
    # Ensure inputs are on the correct device and float32
    # We detach to ensure we don't track gradients for backprop (save memory)
    g_x = grad_x.to(device, dtype=torch.float32).detach()
    g_y = grad_y.to(device, dtype=torch.float32).detach()
    
    h, w = g_x.shape
    
    # --- Initialization ---
    u = torch.zeros((h, w), device=device, dtype=torch.float32)
    u_bar = torch.zeros_like(u)
    
    # Dual variables (p_x, p_y) packed into (H, W, 2)
    p = torch.zeros((h, w, 2), device=device, dtype=torch.float32)
    
    # Hyperparameters for convergence
    # L2 norm of the derivative operator is sqrt(8)
    L = 8.0 
    tau = 1.0 / (L ** 0.5)
    sigma = 1.0 / (L ** 0.5)
    
    # --- Operators Definition (Neumann Boundaries) ---
    
    def gradient(img):
        """
        Forward difference with Neumann boundary (last col/row grad = 0)
        """
        # dx: img[:, 1:] - img[:, :-1]
        # dy: img[1:, :] - img[:-1, :]
        
        # Pre-allocate output
        grad = torch.zeros((h, w, 2), device=device, dtype=torch.float32)
        
        # x-gradient
        grad[:, :-1, 0] = img[:, 1:] - img[:, :-1]
        
        # y-gradient
        grad[:-1, :, 1] = img[1:, :] - img[:-1, :]
        
        return grad

    def divergence(dual_p):
        """
        Backward difference (Adjoint of Gradient).
        div_x = p(x) - p(x-1)
        """
        # Unpack
        p_x = dual_p[..., 0]
        p_y = dual_p[..., 1]
        
        div = torch.zeros((h, w), device=device, dtype=torch.float32)
        
        # --- X Divergence ---
        # Body: p_x[:, 1:-1] - p_x[:, :-2]
        div[:, 1:-1] += p_x[:, 1:-1] - p_x[:, :-2]
        # Boundary Left: p_x[:, 0]
        div[:, 0]    += p_x[:, 0]
        # Boundary Right: -p_x[:, -2] (Matches the 0-padding in gradient)
        div[:, -1]   -= p_x[:, -2]
        
        # --- Y Divergence ---
        div[1:-1, :] += p_y[1:-1, :] - p_y[:-2, :]
        div[0, :]    += p_y[0, :]
        div[-1, :]   -= p_y[-2, :]
        
        return div

    # --- Main Loop ---
    # print(f"Starting TV-L1 reconstruction on {device}...")
    
    for i in range(n_iters):
        # 1. Dual Update (Ascent in Dual)
        # p = p + sigma * (grad(u_bar) - G_obs)
        grad_u_bar = gradient(u_bar)
        
        # Update vector p
        # We process x and y channels together
        p[..., 0] += sigma * (grad_u_bar[..., 0] - g_x)
        p[..., 1] += sigma * (grad_u_bar[..., 1] - g_y)
        
        # 2. Projection (Proximal Operator)
        # Project p onto ball of radius lambda_tv
        # This implements the "sparsity" or "TV" constraint
        norm_p = torch.sqrt(p[..., 0]**2 + p[..., 1]**2)
        
        # Avoid division by zero
        scale = torch.maximum(torch.ones_like(norm_p), norm_p / lambda_tv)
        
        p[..., 0] /= scale
        p[..., 1] /= scale
        
        # 3. Primal Update (Descent in Primal)
        u_old = u.clone()
        
        div_p = divergence(p)
        u += tau * div_p

        # Dirichlet boundary condition
        u[:, 0] = 0; u[:, -1] = 0; u[0, :] = 0; u[-1, :] = 0
        
        # 4. Relaxation (Momentum / Extrapolation)
        u_bar = u + theta * (u - u_old)
        
        # Optional: Print progress
        # if i % 100 == 0:
        #     diff = torch.mean(torch.abs(u - u_old)).item()
        #     print(f"Iter {i}, Update Diff: {diff:.6e}")
        #     if diff < 1e-7:
        #         print(f"Converged at iter {i}")
        #         break

    return u, p


def tv_l1_reconstruction_strict_neumann(grad_x, grad_y, lambda_tv=0.2, n_iters=2000, device='cuda'):
    """
    TV-L1 Reconstruction with Strict Zero-Gradient Boundary (Neumann).
    不使用 Padding，通过强制边界梯度为 0 来实现。
    
    效果：图像在接触边界时会自动“变平”，不会产生镜像或伪影。
    """
    # 1. 准备数据
    g_x = grad_x.to(device, dtype=torch.float32).detach()
    g_y = grad_y.to(device, dtype=torch.float32).detach()
    h, w = g_x.shape
    
    # --- [关键修改 1]：输入梯度掩膜 (Input Masking) ---
    # 既然我们要求边缘梯度为 0，那么输入数据里的边缘梯度也必须被忽略
    # 否则算法会试图去拟合边界上的梯度，导致冲突
    g_x[:, 0] = 0; g_x[:, -1] = 0
    g_x[0, :] = 0; g_x[-1, :] = 0
    
    g_y[:, 0] = 0; g_y[:, -1] = 0
    g_y[0, :] = 0; g_y[-1, :] = 0
    
    # 初始化
    u = torch.zeros((h, w), device=device, dtype=torch.float32)
    u_bar = torch.zeros_like(u)
    p = torch.zeros((h, w, 2), device=device, dtype=torch.float32)
    
    L = 8.0
    tau = 1.0 / (L ** 0.5)
    sigma = 1.0 / (L ** 0.5)
    theta = 1.0

    # --- [关键修改 2]：严格的梯度算子 ---
    def gradient_strict(img):
        """
        计算梯度，并强制边界梯度为 0。
        不使用 Pad，直接在内部计算，保持形状不变。
        """
        grad = torch.zeros((h, w, 2), device=device, dtype=torch.float32)
        
        # X 方向梯度: u[x+1] - u[x]
        # 最后一列 (w-1) 没有右邻居，梯度强制为 0
        grad[:, :-1, 0] = img[:, 1:] - img[:, :-1]
        grad[:, -1, 0] = 0  # <--- Explicit Zero Gradient constraint
        
        # Y 方向梯度: u[y+1] - u[y]
        # 最后一行 (h-1) 没有下邻居，梯度强制为 0
        grad[:-1, :, 1] = img[1:, :] - img[:-1, :]
        grad[-1, :, 1] = 0  # <--- Explicit Zero Gradient constraint
        
        return grad

    # --- [关键修改 3]：严格的散度算子 (Adjoint) ---
    def divergence_strict(dual_p):
        """
        梯度的负转置 (Negative Adjoint)。
        必须严格匹配 gradient_strict 的边界逻辑。
        """
        p_x = dual_p[..., 0]
        p_y = dual_p[..., 1]
        
        div = torch.zeros((h, w), device=device, dtype=torch.float32)
        
        # --- X Divergence (Backward Difference) ---
        # 核心区域: p_x[x] - p_x[x-1]
        div[:, 1:-1] += p_x[:, 1:-1] - p_x[:, :-2]
        
        # 左边界 (x=0): +p_x[0] (对应 grad 的左项)
        div[:, 0]    += p_x[:, 0]
        
        # 右边界 (x=w-1): 
        # 在 gradient_strict 中，grad[:, -1] 被强制设为 0。
        # 这意味着 u[w-1] 这一项从来没有被用到计算梯度中。
        # 根据伴随算子定义，这里应该减去 p_x[w-2]。
        div[:, -1]   -= p_x[:, -2]
        
        # --- Y Divergence ---
        div[1:-1, :] += p_y[1:-1, :] - p_y[:-2, :]
        div[0, :]    += p_y[0, :]
        div[-1, :]   -= p_y[-2, :]
        
        return div

    print("Starting Strict Neumann TV-L1...")
    
    for i in range(n_iters):
        # 1. Dual Update
        grad_u_bar = gradient_strict(u_bar)
        
        p[..., 0] += sigma * (grad_u_bar[..., 0] - g_x)
        p[..., 1] += sigma * (grad_u_bar[..., 1] - g_y)
        
        # Projection (L1 constraint)
        norm_p = torch.sqrt(p[..., 0]**2 + p[..., 1]**2)
        scale = torch.maximum(torch.ones_like(norm_p), norm_p / lambda_tv)
        p[..., 0] /= scale
        p[..., 1] /= scale
        
        # 2. Primal Update
        u_old = u.clone()
        
        div_p = divergence_strict(p)
        u += tau * div_p

        # 如果你想要黑边框 (Dirichlet)，解开这行注释：
        u[:, 0] = 0; u[:, -1] = 0; u[0, :] = 0; u[-1, :] = 0
        
        # 3. Relaxation
        u_bar = u + theta * (u - u_old)
        
        if i % 100 == 0:
            diff = torch.mean(torch.abs(u - u_old)).item()
            loss = torch.mean(torch.abs(grad_u_bar[..., 0] - g_x)) + torch.mean(torch.abs(grad_u_bar[..., 1] - g_y))
            print(f"Iter {i}, Diff: {diff:.2e}, Loss: {loss:.6e}")
        
        if i % 1000 == 0:
            lambda_tv = lambda_tv * 0.5
            print(f"lambda_tv: {lambda_tv}")

    return u


def tv_l1_adaptive_cuda(grad_x, grad_y, 
                        lambda_tv = 1.0,
                        n_iters=2000, device='cuda'):
    """
    Spatially Adaptive TV-L1 Reconstruction.
    """

    lambda_min = lambda_tv
    lambda_max = lambda_tv

    # 1. 准备数据
    g_x = grad_x.to(device, dtype=torch.float32).detach()
    g_y = grad_y.to(device, dtype=torch.float32).detach()
    h, w = g_x.shape
    
    # 强制 Neumann 边界 (输入梯度掩膜)
    g_x[:, 0] = 0; g_x[:, -1] = 0
    g_x[0, :] = 0; g_x[-1, :] = 0
    g_y[:, 0] = 0; g_y[:, -1] = 0
    g_y[0, :] = 0; g_y[-1, :] = 0

    # --- [核心步骤]：生成 Adaptive Lambda Map ---
    # 1. 计算梯度幅值
    grad_mag = torch.sqrt(g_x**2 + g_y**2)
    
    # 2. 稍微平滑一下幅值图 (关键！)
    # 如果不平滑，背景里的噪点会被误认为是边缘，导致噪点被保留。
    # 用一个 5x5 的高斯核或均值核模糊一下“信任图”
    # 这里用简单的 average pooling 模拟模糊
    weight_map = F.avg_pool2d(grad_mag.unsqueeze(0).unsqueeze(0), 
                              kernel_size=5, stride=1, padding=2).squeeze()
    
    # --- [修复开始]：解决 quantile tensor too large ---
    # 不要对全图算 quantile，而是先下采样。
    # 自动计算步长，确保采样后的元素数量在 100万以内 (排序非常快且不占显存)
    num_pixels = h * w
    target_samples = 1000000 # 1M samples is enough for accurate estimation
    step = max(1, int((num_pixels / target_samples) ** 0.5))
    
    # 使用切片 [::step, ::step] 进行稀疏采样
    sampled_map = weight_map[::step, ::step]
    
    # 在采样后的数据上算 quantile
    max_val = torch.quantile(sampled_map, 0.95)
    # --- [修复结束] ---
    
    if max_val == 0: max_val = 1.0
    weight_map = torch.clamp(weight_map / max_val, 0, 1)

    # --- 初始化 ---
    u = torch.zeros((h, w), device=device, dtype=torch.float32)
    u_bar = torch.zeros_like(u)
    p = torch.zeros((h, w, 2), device=device, dtype=torch.float32)
    
    L = 8.0
    tau = 1.0 / (L ** 0.5)
    sigma = 1.0 / (L ** 0.5)
    theta = 1.0

    # 严格 Neumann 算子 (同上一个版本)
    def gradient_strict(img):
        grad = torch.zeros((h, w, 2), device=device, dtype=torch.float32)
        grad[:, :-1, 0] = img[:, 1:] - img[:, :-1]
        grad[:, -1, 0] = 0 
        grad[:-1, :, 1] = img[1:, :] - img[:-1, :]
        grad[-1, :, 1] = 0 
        return grad

    def divergence_strict(dual_p):
        p_x = dual_p[..., 0]
        p_y = dual_p[..., 1]
        div = torch.zeros((h, w), device=device, dtype=torch.float32)
        div[:, 1:-1] += p_x[:, 1:-1] - p_x[:, :-2]
        div[:, 0]    += p_x[:, 0]
        div[:, -1]   -= p_x[:, -2]
        div[1:-1, :] += p_y[1:-1, :] - p_y[:-2, :]
        div[0, :]    += p_y[0, :]
        div[-1, :]   -= p_y[-2, :]
        return div

    print(f"Starting Adaptive TV-L1 (Min={lambda_min}, Max={lambda_max})...")
    tv_gt = torch.mean(torch.abs(g_x)) + torch.mean(torch.abs(g_y))
    print(f"tv_gt: {tv_gt}")
    
    for i in range(n_iters):
        # 1. Dual Update
        grad_u_bar = gradient_strict(u_bar)
        
        p[..., 0] += sigma * (grad_u_bar[..., 0] - g_x)
        p[..., 1] += sigma * (grad_u_bar[..., 1] - g_y)
        
        # --- [关键修改]：Adaptive Projection ---
        # 以前是除以常数，现在除以 lambda_map (逐像素调整约束半径)
        norm_p = torch.sqrt(p[..., 0]**2 + p[..., 1]**2).unsqueeze(-1)

        # 4. 映射到 [lambda_min, lambda_max]
        # 边缘处 (weight=1) -> lambda_max (保细节)
        # 背景处 (weight=0) -> lambda_min (强平滑)
        if i % 1000 == 0 and i > 0:
            lambda_min *= 0.5
            lambda_max *= 0.9
            print(f"lambda_min: {lambda_min}, lambda_max: {lambda_max}")
        lambda_map = lambda_min + (lambda_max - lambda_min) * weight_map
        
        # 这里的 lambda_map 是一个 (H, W) 的张量
        # 我们需要让它广播到 (H, W, 2) 以便用于投影 p
        lambda_map = lambda_map.unsqueeze(-1) 
        
        # scale = max(1, |p| / lambda(x))
        # 注意广播机制
        scale = torch.maximum(torch.ones_like(norm_p), norm_p / lambda_map)
        
        p /= scale
        
        # 2. Primal Update
        u_old = u.clone()
        div_p = divergence_strict(p)
        u += tau * div_p
        
        # 3. Relaxation
        u_bar = u + theta * (u - u_old)
        
        # --- [监控模块] ---
        # 每 N 次迭代打印一次 (不要每次都算，会拖慢 GPU)
        if i % 100 == 0:
            # (1) 计算收敛残差 (Diff)
            diff = torch.mean(torch.abs(u - u_old)).item()
            
            # (2) 计算 Data Fidelity Loss: || grad(u) - G_input ||_1
            # 重新计算当前的梯度 (注意：用 u 而不是 u_bar)
            curr_grad = gradient_strict(u)
            
            # 计算 L1 误差 (排除边界，因为边界被强制为0了)
            # Fidelity = |Gx_pred - Gx_gt| + |Gy_pred - Gy_gt|
            fid_loss = torch.mean(torch.abs(curr_grad[..., 0] - g_x)) + \
                       torch.mean(torch.abs(curr_grad[..., 1] - g_y))
            
            # (3) 计算 TV Loss (Regularization Term): || grad(u) ||_1
            # 这衡量了图像的平滑程度
            tv_loss = torch.mean(torch.abs(curr_grad[..., 0])) + \
                      torch.mean(torch.abs(curr_grad[..., 1]))
            
            # 打印
            print(f"{i:<6d} | {diff:.2e}     | {fid_loss.item():.6f}        | {tv_loss.item():.6f}")

            # [可选] 自动早停 (Early Stopping)
            if diff < 1e-7:
                print(f"Converged at iter {i}")
                break

    return u


# def reconstruct_color_consistent(grad_x_rgb, grad_y_rgb, 
#                                  solver_func,
#                                  lambda_lum=0.1, 
#                                  lambda_chrom=2.0,
#                                  n_iters=2000,
#                                  device='cuda'):
#     """
#     Color-Consistent Reconstruction (Luminance-Chrominance Decomposition).
#     解决 RGB 独立重建导致的色偏和杂色问题。
    
#     Args:
#         grad_x_rgb: (H, W, 3) tensor, RGB 三通道的 X 梯度
#         grad_y_rgb: (H, W, 3) tensor, RGB 三通道的 Y 梯度
#         solver_func: 之前写好的 TV-L1 求解器函数 (接受 gx, gy, lambda)
#         lambda_lum: 亮度通道的正则化 (保留细节，建议 0.1 - 0.2)
#         lambda_chrom: 色差通道的正则化 (强力去噪，建议 1.0 - 5.0)
#     """
#     # 1. 拆分通道
#     # 假设输入是 (H, W, 3)
#     gx_r, gx_g, gx_b = grad_x_rgb[..., 0], grad_x_rgb[..., 1], grad_x_rgb[..., 2]
#     gy_r, gy_g, gy_b = grad_y_rgb[..., 0], grad_y_rgb[..., 1], grad_y_rgb[..., 2]
    
#     print("--- Phase 1: Gradient Transformation ---")
#     # 2. 变换梯度到 L-C1-C2 空间
#     # L (Luminance): 亮度 = (R+G+B)/3
#     # C1 (Chrominance 1): 红绿差 = R - G
#     # C2 (Chrominance 2): 蓝绿差 = B - G
    
#     # 利用线性性: grad(L) = (grad(R) + grad(G) + grad(B)) / 3
#     gx_l = (gx_r + gx_g + gx_b) / 3.0
#     gy_l = (gy_r + gy_g + gy_b) / 3.0
    
#     gx_c1 = gx_r - gx_g
#     gy_c1 = gy_r - gy_g
    
#     gx_c2 = gx_b - gx_g
#     gy_c2 = gy_b - gy_g
    
#     print("--- Phase 2: Solving Independent Channels ---")
    
#     # 3. 分别重建 (关键在于 Lambda 的不同)
    
#     # (A) 重建亮度 L: 使用小 Lambda，保留所有纹理细节
#     print(f"Reconstructing Luminance (lambda={lambda_lum})...")
#     # 注意：solver_func 返回的是 numpy 还是 tensor，这里统一转回 tensor 处理
#     img_l = solver_func(gx_l, gy_l, lambda_tv=lambda_lum, n_iters=n_iters, device=device)
#     if isinstance(img_l, np.ndarray): img_l = torch.from_numpy(img_l).to(device)
    
#     # (B) 重建色差 C1 (R-G): 使用大 Lambda，强迫平滑
#     # 这步就是你想要的 "minimize TV(R-G)"
#     print(f"Reconstructing Chrominance R-G (lambda={lambda_chrom})...")
#     img_c1 = solver_func(gx_c1, gy_c1, lambda_tv=lambda_chrom, n_iters=n_iters, device=device)
#     if isinstance(img_c1, np.ndarray): img_c1 = torch.from_numpy(img_c1).to(device)

#     # (C) 重建色差 C2 (B-G): 使用大 Lambda
#     print(f"Reconstructing Chrominance B-G (lambda={lambda_chrom})...")
#     img_c2 = solver_func(gx_c2, gy_c2, lambda_tv=lambda_chrom, n_iters=n_iters, device=device)
#     if isinstance(img_c2, np.ndarray): img_c2 = torch.from_numpy(img_c2).to(device)

#     # (B) & (C) 联合重建色度 (REPLACED)
#     # 不再单独解 C1, C2，而是联合解，强制 Group Sparsity
#     # print(f"Reconstructing Coupled Chrominance (lambda={lambda_chrom})...")
    
#     # # ！！！ 调用新写的函数 ！！！
#     # img_c1, img_c2 = solve_coupled_chroma_proximal(
#     #                     gx_c1, gy_c1, gx_c2, gy_c2, 
#     #                     lambda_chrom=lambda_chrom, # 这里的 lambda 要设大一点，比如 2.0 - 5.0
#     #                     n_iters=n_iters, 
#     #                     device=device)
    
#     print("--- Phase 3: Inverse Transformation ---")
#     # 4. 逆变换回 RGB
#     # 解方程组:
#     # L = (R+G+B)/3
#     # C1 = R - G  => R = C1 + G
#     # C2 = B - G  => B = C2 + G
#     # 代入第一式: 3L = (C1+G) + G + (C2+G) = C1 + C2 + 3G
#     # => 3G = 3L - C1 - C2
#     # => G = L - (C1 + C2) / 3
    
#     img_g = img_l - (img_c1 + img_c2) / 3.0
#     img_r = img_c1 + img_g
#     img_b = img_c2 + img_g
    
#     # 5. 堆叠并返回
#     img_rgb = torch.stack([img_r, img_g, img_b], dim=-1)
    
#     # 可选：简单的白平衡/归一化 (因为积分会有常数漂移)
#     # 将每个通道的最小值对齐到 0 (或者根据背景色对齐)
#     # img_rgb -= img_rgb.min() 
    
#     return img_rgb.cpu().numpy()


