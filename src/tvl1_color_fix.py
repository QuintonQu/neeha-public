import torch
import torch.nn.functional as F
import numpy as np
import math
from matplotlib import pyplot as plt


def tv_l1_reconstruction_strict_neumann(grad_x, grad_y, lambda_tv=0.2, n_iters=2000, adaptive_iters=2000, device='cuda'):
    """
    TV-L1 Reconstruction with Strict Zero-Gradient Boundary (Neumann).
    不使用 Padding, 通过强制边界梯度为 0 来实现。
    
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
            print(f"Iter {i}, Diff: {diff:.2e}, Loss: {loss:.2e}")
        
        if i % adaptive_iters == 0 and i > 0:
            lambda_tv = lambda_tv * 0.5
            print(f"lambda_tv: {lambda_tv}")
            # plt.imsave(f"../results/simulation/rgb_test/debug/y_{i}.png", u.cpu().numpy())

    return u

def get_dynamic_threshold(u, method='mad', ratio=0.1):
    """
    动态计算稀疏化阈值
    Args:
        u: 当前的图像 (H, W, 2)
        method: 'mad' (噪声估计) 或 'quantile' (分位数)
        ratio: 
            - 如果是 'quantile'，表示切掉底部的比例 (e.g. 0.1 = 10%)
            - 如果是 'mad'，表示 sigma 的倍数 (通常 2.0 ~ 3.0)
    """
    # 1. 计算联合模长 (饱和度)
    # shape: (N, ) 展平
    magnitude = torch.sqrt(torch.sum(u ** 2, dim=-1)).view(-1)
    
    if method == 'quantile':
        # --- 方法 A: 直方图分位数法 (你的想法) ---
        # 找到第 k% 小的值作为阈值
        # 注意：为了效率，可以使用 kthvalue 或者 quantile
        if ratio <= 0: return 0.0
        k = int(magnitude.numel() * ratio)
        # 找到第 k 小的数 (这就相当于 Histogram cutoff)
        threshold = torch.kthvalue(magnitude, k).values.item()
        return threshold

    elif method == 'mad':
        # --- 方法 B: MAD 鲁棒噪声估计 (推荐) ---
        # 假设背景占大多数，中位数可以代表噪声水平
        median_val = torch.median(magnitude)
        
        # 对于 Rayleigh 分布 (两个高斯噪声的模长)，sigma ≈ median / 1.177
        # 但我们简单点，认为中位数就是噪声基准
        # 阈值设为 中位数 * 倍数
        # 如果背景也是有颜色的，这个假设会失效，但对于去白背景很有效
        
        # 改进版 MAD: 只计算非零值的中位数 (防止太多0影响估计)
        non_zero_mag = magnitude[magnitude > 1e-6]
        if non_zero_mag.numel() < 10:
            return 0.0
            
        sigma = torch.median(non_zero_mag)
        return sigma.item() * ratio # ratio 建议 2.0
    
    return 0.0


def calc_noise_min_energy(u, block_size=16, quantile=0.02, safety_limit=0.5):
    """
    寻找'能量最低'的区域来估计背景噪声。
    
    Args:
        block_size: 块大小
        quantile: 取最暗的百分之多少的块 (比如 0.05 表示只看最暗的 5% 区域)
        safety_limit: 安全阈值。如果最暗的块依然很亮(大于此值)，说明全图无背景，强制不截断。
    """
    # 1. 计算每个像素的模长 (H, W)
    magnitude = torch.sqrt(torch.sum(u**2, dim=-1))
    
    # 2. 切分成块
    # 使用 unfold 高效切块: (N_blocks, block_size, block_size)
    h, w = magnitude.shape
    # 简单的 view 变形前提是 h, w 能被 block_size 整除，为了通用性，我们用 unfold
    # 但 unfold 比较占显存，我们可以先粗暴地 resize 到小图再处理，或者只采样
    
    # 这种写法稍微繁琐但通用：
    patches = magnitude.unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    patches = patches.contiguous().view(-1, block_size * block_size)
    # patches shape: (num_blocks, num_pixels_per_block)
    
    # 3. 计算每个块的"平均能量" (Mean Magnitude)
    block_means = torch.mean(patches, dim=1)
    
    # 4. 找到能量最低的前 k% 的块
    # 这些块被认为是"背景候选者"
    k = max(1, int(block_means.numel() * quantile))
    
    # topk(largest=False) 返回最小的 k 个值
    # values 是这些最暗块的平均值
    bg_means, indices = torch.topk(block_means, k, largest=False)
    
    # 5. 安全检查：如果连最暗的块都很亮（说明没有白色背景）
    # 比如全图最小的均值都 > 0.1 (归一化后的值)，那说明全是颜色
    min_val = torch.min(bg_means)
    if min_val > safety_limit:
        print(f"No background found, returning 0.0")
        return 0.0 # 放弃截断，保留所有颜色
        
    # 6. 从这些最暗的块中，估计最大噪声
    # 我们取出这些块的原始像素，计算它们的最大值或者 3-sigma
    best_patches = patches[indices] # (k, pixels)
    
    # 这里的策略是：既然这些是背景，那么这里面的最大波动就是噪声上限
    # 为了稳健，取这些背景像素的 95% 分位数，或者 mean + 3*std
    
    bg_pixels = best_patches.view(-1)
    bg_mean = torch.mean(bg_pixels)
    bg_std = torch.std(bg_pixels)
    
    # 阈值 = 背景均值 + 3倍背景波动
    threshold = bg_mean + 3.0 * bg_std
    
    return threshold.item()



# def solve_coupled_chroma_strict_neumann(gx_c1, gy_c1, gx_c2, gy_c2, 
#                                         lambda_tv=0.2,
#                                         lambda_sparsity=0.05, 
#                                         n_iters=2000, 
#                                         adaptive_iters=2000,
#                                         device='cuda'):
#     """
#     Coupled TV-L1 Reconstruction with Strict Neumann Boundary.
#     融合版本：
#     1. 使用 Strict Neumann (强制边界梯度为0)。
#     2. 使用 Joint/Vectorial TV (强制 C1, C2 边缘对齐)。
#     """
#     # ==========================================
#     # 1. 数据准备 (Stacking & Masking)
#     # ==========================================
    
#     # 堆叠 -> (H, W, 2)
#     g_x = torch.stack([gx_c1, gx_c2], dim=-1).to(device, dtype=torch.float32).detach()
#     g_y = torch.stack([gy_c1, gy_c2], dim=-1).to(device, dtype=torch.float32).detach()
    
#     h, w, c = g_x.shape # c = 2
    
#     # --- Input Masking (你的核心逻辑) ---
#     # 强制输入梯度的边界为 0，防止算法去拟合边界上的伪梯度
#     g_x[:, 0, :] = 0; g_x[:, -1, :] = 0
#     g_x[0, :, :] = 0; g_x[-1, :, :] = 0
    
#     g_y[:, 0, :] = 0; g_y[:, -1, :] = 0
#     g_y[0, :, :] = 0; g_y[-1, :, :] = 0
    
#     # 将目标梯度组合成 (H, W, C, 2) 以便在 Dual Update 中直接相减
#     # g_target[..., 0] = gx, g_target[..., 1] = gy
#     g_target = torch.stack([g_x, g_y], dim=-1)
#     g_target[:, 0, :, :] = 0  # 左边缘
#     g_target[:, -1, :, :] = 0 # 右边缘
#     g_target[0, :, :, :] = 0  # 上边缘
#     g_target[-1, :, :, :] = 0 # 下边缘

#     # ==========================================
#     # 2. 初始化变量
#     # ==========================================
    
#     u = torch.zeros((h, w, c), device=device, dtype=torch.float32)
#     u_bar = torch.zeros_like(u)
    
#     # 对偶变量 p: (H, W, Channels, Directions) -> (H, W, 2, 2)
#     p = torch.zeros((h, w, c, 2), device=device, dtype=torch.float32)
    
#     # 算法参数
#     L = 8.0
#     tau = 1.0 / (L ** 0.5)
#     sigma = 1.0 / (L ** 0.5)
#     theta = 1.0
    
#     current_lambda = lambda_tv
#     current_sparsity_threshold = 0.0

#     # ==========================================
#     # 3. 定义严格算子 (Joint Version)
#     # ==========================================

#     def gradient_strict_joint(img):
#         """
#         计算梯度，强制边界为 0 (无需 Padding)。
#         Input: (H, W, C)
#         Output: (H, W, C, 2)
#         """
#         grad = torch.zeros((h, w, c, 2), device=device, dtype=torch.float32)
        
#         # X 方向: u[x+1] - u[x]
#         grad[:, :-1, :, 0] = img[:, 1:, :] - img[:, :-1, :]
#         grad[:, -1, :, 0] = 0 # 严格约束
        
#         # Y 方向: u[y+1] - u[y]
#         grad[:-1, :, :, 1] = img[1:, :, :] - img[:-1, :, :]
#         grad[-1, :, :, 1] = 0 # 严格约束
        
#         return grad

#     def divergence_strict_joint(dual_p):
#         """
#         梯度的负转置 (Negative Adjoint)。
#         Input: (H, W, C, 2)
#         Output: (H, W, C)
#         """
#         p_x = dual_p[..., 0]
#         p_y = dual_p[..., 1]
        
#         div = torch.zeros((h, w, c), device=device, dtype=torch.float32)
        
#         # X Divergence
#         div[:, 1:-1, :] += p_x[:, 1:-1, :] - p_x[:, :-2, :]
#         div[:, 0, :]    += p_x[:, 0, :]
#         div[:, -1, :]   -= p_x[:, -2, :] # 对应 gradient 右边界截断
        
#         # Y Divergence
#         div[1:-1, :, :] += p_y[1:-1, :, :] - p_y[:-2, :, :]
#         div[0, :, :]    += p_y[0, :, :]
#         div[-1, :, :]   -= p_y[-2, :, :] # 对应 gradient 下边界截断
        
#         return div
    
#     # [新增] 矢量软阈值算子 (Vectorial Soft Thresholding)
#     def proximal_group_sparsity(u_in, threshold):
#         # 计算 c1, c2 的联合模长: sqrt(c1^2 + c2^2)
#         # u_in shape: (H, W, 2) -> dim=-1 是通道
#         magnitude = torch.sqrt(torch.sum(u_in ** 2, dim=-1, keepdim=True)) # (H, W, 1)
        
#         # 计算收缩比例: max(0, 1 - threshold / |u|)
#         # 如果模长小于 threshold，这一项变成 0，颜色被彻底抹除
#         scale = torch.maximum(
#             torch.zeros_like(magnitude), 
#             1.0 - threshold / (magnitude + 1e-8)
#         )
        
#         # 应用收缩 (c1 和 c2 同时变小或归零)
#         return u_in * scale
    
#     def proximal_group_sigmoid(u_in, threshold, sharpness=100.0):
#         """
#         Sigmoid Gating: 平滑的硬截断。
#         Args:
#             sharpness: 控制 Sigmoid 的陡峭程度。
#                     值越大越像 Hard Threshold，值越小过渡越平缓。
#         """
#         # 1. 计算联合模长
#         magnitude = torch.sqrt(torch.sum(u_in ** 2, dim=-1, keepdim=True))
        
#         # 2. 计算 Sigmoid 增益系数 (Gain)
#         # (magnitude - threshold) > 0 时，exp 为负，分母接近 1，gain -> 1
#         # (magnitude - threshold) < 0 时，exp 为正大数，分母很大，gain -> 0
#         # sharpness 用于放大这个差异，让过渡带变窄
#         gain = torch.sigmoid(sharpness * (magnitude - threshold))
        
#         # 3. 应用增益
#         return u_in * gain

#     # ==========================================
#     # 4. 优化循环
#     # ==========================================
    
#     print(f"Starting Vectorial TV-L1 (Joint C1/C2 optimization)...")
    
#     for i in range(n_iters):
#         # --- Step 1: Dual Update ---
#         grad_u_bar = gradient_strict_joint(u_bar)
        
#         # 梯度上升: p = p + sigma * (Gradient(u) - TargetGradient)
#         p = p + sigma * (grad_u_bar - g_target)
        
#         # --- Step 2: Joint Projection (Vectorial TV 核心) ---
#         # 我们要计算 (dx_c1, dy_c1, dx_c2, dy_c2) 这个 4D 向量的模长
#         # p shape: (H, W, 2, 2) -> dim=2 是通道, dim=3 是方向
        
#         # 求和维度 (2, 3) 意味着把 C1, C2 和 X, Y 全部平方加在一起
#         # norm_p shape: (H, W)
#         norm_p = torch.sqrt(torch.sum(p ** 2, dim=(2, 3)))
        
#         # 避免除以 0
#         norm_p = torch.maximum(norm_p, torch.tensor(1e-8, device=device))
        
#         # 投影系数
#         scale = torch.maximum(torch.ones_like(norm_p), norm_p / current_lambda)
        
#         # 广播回 (H, W, 2, 2) 并进行除法
#         # scale 是 (H, W)，需要扩展后两个维度
#         p = p / scale.unsqueeze(-1).unsqueeze(-1)
        
#         # --- Step 3: Primal Update ---
#         u_old = u.clone()
        
#         div_p = divergence_strict_joint(p)
#         u = u + tau * div_p

#         # [新增] 再应用 Soft Thresholding (Sparsity)
#         # 注意阈值需要乘以步长 tau
#         # if lambda_sparsity > 0 and i % 100 == 0:
#         #     u = proximal_group_sparsity(u, threshold=tau * lambda_sparsity)
#         #     print(f"Applied Sparsity (lambda_sparsity={lambda_sparsity})")

#         # --- [关键修改] 动态调整阈值 ---
#         # 每 200 轮重新评估一次阈值，防止震荡
#         if i > 50 and i % 200 == 0:
#             # 计算这一刻图像的"底噪"
#             # new_thresh = get_dynamic_threshold(u, method="mad", ratio=2.0)
#             new_thresh = calc_noise_min_energy(u)
            
#             # 使用动量更新 (Exponential Moving Average)，让阈值变化平滑
#             # threshold = 0.7 * old + 0.3 * new
#             current_sparsity_threshold = 0.7 * current_sparsity_threshold + 0.3 * new_thresh
#             print(f"Iter {i}: Auto Threshold updated to {current_sparsity_threshold:.5f}")

#             # 应用 Soft Thresholding
#             if current_sparsity_threshold > 1e-6:
#                 # 注意：Primal-Dual 中的 Proximal 算子阈值需要乘步长 tau
#                 # 但这里我们算出来的 threshold 是针对像素值的绝对量级
#                 # 所以直接传进去即可，不需要乘 tau，或者根据你的 proximal 实现调整
#                 # 这里的逻辑是：如果 |u| < threshold，则归零
#                 u = proximal_group_sigmoid(u, threshold=current_sparsity_threshold)

#         # [新增]: 强制 Dirichlet 边界 (黑边框)
#         u[0, :, :] = 0   # 上边缘
#         u[-1, :, :] = 0  # 下边缘
#         u[:, 0, :] = 0   # 左边缘
#         u[:, -1, :] = 0  # 右边缘
        
#         # --- Step 4: Relaxation ---
#         u_bar = u + theta * (u - u_old)
        
#         # 监控收敛
#         if i % 100 == 0:
#             diff = torch.mean(torch.abs(u - u_old)).item()
            
#             # --- 计算 C1 的 Loss (Channel 0) ---
#             # 取出 Channel 0 的所有方向 (X和Y)，计算与目标梯度的 L1 距离
#             loss_c1 = torch.mean(torch.abs(grad_u_bar[:, :, 0, :] - g_target[:, :, 0, :])).item()
            
#             # --- 计算 C2 的 Loss (Channel 1) ---
#             # 取出 Channel 1 的所有方向 (X和Y)，计算与目标梯度的 L1 距离
#             loss_c2 = torch.mean(torch.abs(grad_u_bar[:, :, 1, :] - g_target[:, :, 1, :])).item()
            
#             print(f"Iter {i}, Update Diff: {diff:.2e}, Loss_c1: {loss_c1:.6f}, Loss_c2: {loss_c2:.6f}")
            
#         # Lambda 退火 (如你 sample 代码所示)
#         if i > 0 and i % adaptive_iters == 0:
#             current_lambda = current_lambda * 0.5
#             print(f"Refining lambda -> {current_lambda:.4f}")
#             plt.imsave(f"../results/simulation/rgb_test/debug/c1_{i}.png", u[..., 0].cpu().numpy())
#             plt.imsave(f"../results/simulation/rgb_test/debug/c2_{i}.png", u[..., 1].cpu().numpy())

#     # 返回分离的通道
#     return u[..., 0], u[..., 1]

def get_top_dominant_hue_axes(g_target, device='cuda', max_samples=200000, k_clusters=8, top_n_axes=2):
    """
    找出图像中最重要的几个色调轴（严格锁定上半球）。
    
    逻辑：
    1. 因为梯度是双向的 (v 和 -v 代表同一颜色)，我们强制把所有向量折叠到上半球 (c2 >= 0)。
    2. 进行 K-Means 聚类。
    3. 筛选出权重最大的几个轴。
    """
    print(f"Analyzing dominant hues (Upper Hemisphere Only) with K={k_clusters}...")
    
    # 1. 准备数据 & 下采样
    # Flatten -> (N, 2)
    grad_samples = g_target.permute(0, 1, 3, 2).reshape(-1, 2)
    num_total = grad_samples.shape[0]
    
    if num_total > max_samples:
        perm = torch.randperm(num_total, device=device)[:max_samples]
        grad_subset = grad_samples[perm]
    else:
        grad_subset = grad_samples
        
    # 2. 过滤噪声
    magnitudes = torch.norm(grad_subset, dim=1)
    threshold = torch.quantile(magnitudes, 0.8)
    strong_grads = grad_subset[magnitudes > threshold]
    
    if strong_grads.shape[0] < k_clusters * 2:
        return [torch.tensor([1.0, 0.0], device=device), torch.tensor([0.0, 1.0], device=device)]

    # 3. 归一化
    strong_grads_norm = F.normalize(strong_grads, dim=1)
    
    # =========================================================
    # [核心修改] 强制折叠到上半球 (Upper Hemisphere Folding)
    # =========================================================
    # 规则：
    # 1. 如果 c2 < 0，必须翻转 (-v)。
    # 2. 如果 c2 == 0 且 c1 < 0 (水平向左)，必须翻转为水平向右 (-v)。
    # 这样保证所有向量都在 [0, 180) 度之间，消除了双向歧义。
    
    c1 = strong_grads_norm[:, 0]
    c2 = strong_grads_norm[:, 1]
    
    # 创建翻转掩膜 (需要翻转的地方设为 True)
    # 逻辑：c2 为负，或者 c2为0且c1为负
    flip_condition = (c2 < 0) | ((torch.abs(c2) < 1e-6) & (c1 < 0))
    
    # 制作乘法因子：需要翻转的乘 -1，不需要的乘 1
    flip_factor = torch.where(flip_condition, 
                              torch.tensor(-1.0, device=device), 
                              torch.tensor(1.0, device=device)).unsqueeze(1)
    
    # 应用折叠
    folded_grads = strong_grads_norm * flip_factor
    
    # =========================================================
    
    # 4. K-Means 聚类 (对折叠后的数据)
    # 随机初始化
    centers = folded_grads[torch.randperm(folded_grads.size(0))[:k_clusters]]
    
    final_labels = None
    for _ in range(15): 
        scores = torch.mm(folded_grads, centers.t())
        final_labels = torch.argmax(scores, dim=1)
        
        new_centers = []
        for k in range(k_clusters):
            mask = (final_labels == k)
            if mask.sum() > 0:
                cluster_mean = folded_grads[mask].mean(dim=0)
                new_centers.append(F.normalize(cluster_mean, dim=0))
            else:
                # 重新随机采样
                random_idx = torch.randint(0, folded_grads.size(0), (1,))
                new_centers.append(folded_grads[random_idx].squeeze(0))
        centers = torch.stack(new_centers)

    # 5. 统计权重并排序
    cluster_weights = []
    for k in range(k_clusters):
        weight = (final_labels == k).sum().item()
        cluster_weights.append(weight)
    
    sorted_indices = np.argsort(cluster_weights)[::-1]
    sorted_centers = centers[sorted_indices.copy()]
    
    # 6. 筛选主轴 (Dot Product Check)
    selected_axes = []
    
    for i in range(k_clusters):
        candidate = sorted_centers[i]
        is_unique = True
        
        for selected in selected_axes:
            # 这里的 candidate 已经在上半球了，所以只需检查它们是否靠太近
            # 不需要检查 dot < -0.9，因为反向向量已经被翻转过来了
            dot = torch.sum(candidate * selected)
            
            # 如果夹角小于约 25度 (cos(25) ≈ 0.9)，认为重复
            if dot > 0.9: 
                is_unique = False
                break
        
        if is_unique:
            selected_axes.append(candidate)
            if len(selected_axes) >= top_n_axes:
                break
                
    print(f"Found {len(selected_axes)} distinct axes (Hemisphere Folded).")
    for idx, ax in enumerate(selected_axes):
        # 为了方便看，如果是 c2 < 0 的微小误差，强制显示为正
        print(f"  Axis {idx+1}: ({ax[0]:.2f}, {ax[1]:.2f})")
        
    return selected_axes


def plot_gradient_scatter(g_target, detected_axes=None, save_path="../results/debug/grad_scatter.png"):
    """
    绘制梯度向量分布图 (2D Histogram)。
    X轴: C1 Gradient
    Y轴: C2 Gradient
    增加功能: 过滤掉坐标轴上的伪影点。
    """
    print("Plotting gradient scatter...")
    
    # 1. 数据准备
    # g_target shape: (H, W, 2, 2) -> (H, W, C=2, Dir=2)
    # Flatten -> (N, 2)
    grads = g_target.detach().permute(0, 1, 3, 2).reshape(-1, 2).cpu().numpy()
    
    # 2. 过滤噪声 (只看强梯度)
    # 计算模长
    norms = np.linalg.norm(grads, axis=1)
    
    # 设定能量阈值 (去除中心空洞)
    thresh = np.percentile(norms, 80)
    
    # =========================================================
    # [新增] 3. 联合过滤: 能量必须够大 AND 不能在轴上
    # =========================================================
    axis_epsilon = 1e-6  # 判定是否在轴上的阈值 (非常接近0就算在轴上)
    
    # 条件1: 能量足够大 (原有逻辑)
    mask_strong = norms > thresh
    
    # 条件2: 不在 X 轴上 (abs(y) > eps) 且 不在 Y 轴上 (abs(x) > eps)
    # 注意：grads[:, 0] 是 x坐标, grads[:, 1] 是 y坐标
    mask_not_on_axis = (np.abs(grads[:, 0]) > axis_epsilon) & \
                       (np.abs(grads[:, 1]) > axis_epsilon)
                       
    # 组合掩膜
    final_mask = mask_strong 
    # final_mask = mask_strong & mask_not_on_axis
    
    strong_grads = grads[final_mask]
    
    if len(strong_grads) == 0:
        print("No strong gradients found to plot (after filtering axes).")
        return

    # 4. 绘图
    plt.figure(figsize=(10, 8))
    
    # 使用 2D 直方图
    # bins 可以适当调大一点看细节
    plt.hist2d(strong_grads[:, 0], strong_grads[:, 1], bins=150, cmap='inferno', density=True)
    plt.colorbar(label='Density (Pixel Count)')
    
    # 绘制参考线 (此时轴上应该没有点了，变黑了)
    plt.axhline(0, color='white', alpha=0.3, linestyle='--')
    plt.axvline(0, color='white', alpha=0.3, linestyle='--')
    
    # 5. 叠加显示检测到的主轴
    if detected_axes is not None:
        limit = np.max(np.abs(strong_grads)) * 0.8 if len(strong_grads) > 0 else 1.0
        
        for i, ax_vec in enumerate(detected_axes):
            vec = ax_vec.cpu().numpy()
            
            # 正向
            plt.arrow(0, 0, vec[0]*limit, vec[1]*limit, 
                      head_width=limit*0.05, head_length=limit*0.05, 
                      fc='cyan', ec='cyan', linewidth=2, label=f'Axis {i+1}')
            
            # 反向
            plt.plot([0, -vec[0]*limit], [0, -vec[1]*limit], 'c--', alpha=0.5)
            
            # 标注
            plt.text(vec[0]*limit*1.1, vec[1]*limit*1.1, f'Axis {i+1}', 
                     color='cyan', fontsize=12, fontweight='bold')

    plt.title(f'Gradient Vector Distribution (Off-Axis)\nTotal points: {len(strong_grads)}')
    plt.xlabel('Gradient C2')
    plt.ylabel('Gradient C1')
    plt.grid(False)
    plt.axis('equal')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {save_path}")


def solve_coupled_chroma_strict_neumann(gx_c1, gy_c1, gx_c2, gy_c2, 
                                        lambda_tv=0.2,
                                        lambda_sparsity=0.05, 
                                        hue_locking_strength=0.95, # [新增] 色调锁定强度 (0.0~1.0)
                                        n_iters=2000, 
                                        adaptive_iters=2000,
                                        device='cuda'):
    """
    Coupled TV-L1 Reconstruction with Hue Locking.
    融合版本：
    1. Strict Neumann (强制边界梯度为0)。
    2. Vectorial TV (强制 C1, C2 边缘对齐)。
    3. Hue Locking (强制色调统一，防止蓝绿斑块)。
    """
    # ==========================================
    # 1. 数据准备 (Stacking & Masking)
    # ==========================================
    
    # 堆叠 -> (H, W, 2)
    g_x = torch.stack([gx_c1, gx_c2], dim=-1).to(device, dtype=torch.float32).detach()
    g_y = torch.stack([gy_c1, gy_c2], dim=-1).to(device, dtype=torch.float32).detach()
    
    h, w, c = g_x.shape # c = 2
    
    # --- Input Masking ---
    g_x[:, 0, :] = 0; g_x[:, -1, :] = 0
    g_x[0, :, :] = 0; g_x[-1, :, :] = 0
    g_y[:, 0, :] = 0; g_y[:, -1, :] = 0
    g_y[0, :, :] = 0; g_y[-1, :, :] = 0
    
    # g_target shape: (H, W, 2, 2)
    g_target = torch.stack([g_x, g_y], dim=-1)
    g_target[:, 0, :, :] = 0  # 左边缘
    g_target[:, -1, :, :] = 0 # 右边缘
    g_target[0, :, :, :] = 0  # 上边缘
    g_target[-1, :, :, :] = 0 # 下边缘

    # ==========================================
    # [新增] 2. 计算全局主色调向量 (Global Dominant Hue)
    # ==========================================
    # 统计全图梯度的总方向。因为梯度代表了颜色的存在，
    # 它们的矢量和指示了"主要的颜色方向"。
    
    # 对 H, W, XY 方向求和，只保留 Channel 维度 (2,)
    total_grad = torch.sum(g_target, dim=(0, 1, 3)) 
    grad_norm = torch.norm(total_grad)
    
    if grad_norm < 1e-6:
        # 如果没有梯度，默认设为对角线方向
        dominant_hue_vec = torch.tensor([0.7071, 0.7071], device=device)
    else:
        # 归一化得到单位向量
        dominant_hue_vec = total_grad / grad_norm
        
    print(f"Detected Dominant Hue: c1={dominant_hue_vec[0]:.2f}, c2={dominant_hue_vec[1]:.2f}")

    # 1. 定义角度（弧度制）
    # 30度 = 30 * pi / 180
    theta = torch.tensor(math.radians(30), device=device) 

    # 2. 构建旋转矩阵 (针对逆时针旋转)
    # 如果需要顺时针旋转，请将 theta 设为 -math.radians(30)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rot_mat = torch.tensor([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ], device=device)

    top_axes = get_top_dominant_hue_axes(g_target, device=device, top_n_axes=3)
    v = top_axes[0]
    top_axes[1] = torch.matmul(rot_mat, v)

    theta = torch.tensor(-math.radians(30), device=device) 

    # 2. 构建旋转矩阵 (针对逆时针旋转)
    # 如果需要顺时针旋转，请将 theta 设为 -math.radians(30)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rot_mat = torch.tensor([
        [cos_t, -sin_t],
        [sin_t,  cos_t]
    ], device=device)

    top_axes[2] = torch.matmul(rot_mat, v)

    plot_gradient_scatter(g_target, detected_axes=top_axes, save_path="../results/debug/gradient_distribution.png")
    # dominant_hue_vec = top_axes[1]  # 选择最重要的轴
    # dominant_hue_vec = torch.flip(dominant_hue_vec, dims=[0])  # 选择最重要的轴

    # ==========================================
    # 3. 初始化变量
    # ==========================================
    
    u = torch.zeros((h, w, c), device=device, dtype=torch.float32)
    u_bar = torch.zeros_like(u)
    p = torch.zeros((h, w, c, 2), device=device, dtype=torch.float32)
    
    L = 8.0
    tau = 1.0 / (L ** 0.5)
    sigma = 1.0 / (L ** 0.5)
    theta = 1.0
    
    current_lambda = lambda_tv
    current_sparsity_threshold = 0.0
    hue_locking_weight = 0.015
    sparsity_weight = 0.1

    # ==========================================
    # 4. 定义算子
    # ==========================================

    def gradient_strict_joint(img):
        grad = torch.zeros((h, w, c, 2), device=device, dtype=torch.float32)
        grad[:, :-1, :, 0] = img[:, 1:, :] - img[:, :-1, :]
        grad[:, -1, :, 0] = 0 
        grad[:-1, :, :, 1] = img[1:, :, :] - img[:-1, :, :]
        grad[-1, :, :, 1] = 0 
        return grad

    def divergence_strict_joint(dual_p):
        p_x = dual_p[..., 0]; p_y = dual_p[..., 1]
        div = torch.zeros((h, w, c), device=device, dtype=torch.float32)
        div[:, 1:-1, :] += p_x[:, 1:-1, :] - p_x[:, :-2, :]
        div[:, 0, :]    += p_x[:, 0, :]
        div[:, -1, :]   -= p_x[:, -2, :] 
        div[1:-1, :, :] += p_y[1:-1, :, :] - p_y[:-2, :, :]
        div[0, :, :]    += p_y[0, :, :]
        div[-1, :, :]   -= p_y[-2, :, :] 
        return div
    
    # [新增] 色调锁定算子
    def proximal_hue_lock(u_in, dominant_vec, strength):
        """
        Hue Locking (Preserving Saturation): 
        旋转 u_in 的方向使其靠近 dominant_vec，但严格保持模长不变。
        """
        if strength <= 1e-4: return u_in
        
        # dominant_vec: (2,) -> (1, 1, 2)
        v = dominant_vec.view(1, 1, 2)
        
        # 1. 计算原始模长 (Saturation)
        # 这是我们必须守护的数值，绝对不能变小
        original_mag = torch.sqrt(torch.sum(u_in ** 2, dim=-1, keepdim=True))
        
        # 2. 构建目标向量 (Target)
        # 方向是 v，长度是原始模长
        u_target = original_mag * v
        
        # 3. 混合方向 (Linear Mix)
        # 注意：两个等长向量线性混合，结果会变短 (弦长 < 弧长)
        u_mixed = (1.0 - strength) * u_in + strength * u_target
        
        # 4. [关键] 恢复模长 (Renormalize)
        # 计算混合后的模长
        mixed_mag = torch.sqrt(torch.sum(u_mixed ** 2, dim=-1, keepdim=True))
        
        # 计算拉伸系数: 原始模长 / 混合后模长
        # 加 1e-8 防止除以 0 (黑色背景区域)
        scale = original_mag / (mixed_mag + 1e-8)
        
        # 5. 输出
        # 此时：方向 = 混合方向， 模长 = 原始模长
        return u_mixed * scale
    

    def proximal_hue_lock_dual_target(u_in, vec1, vec2, strength):
        """
        双色调锁定 (Dual Hue Locking) - 保持模长版
        
        Args:
            u_in: 当前图像 (H, W, 2)
            vec1: 主色调向量 1 (2,)
            vec2: 主色调向量 2 (2,)
            strength: 锁定强度 0~1
        """
        if strength <= 1e-4: return u_in
        
        # ==========================================
        # 1. 制作逐像素的目标向量图 (Target Map)
        # ==========================================
        
        # 归一化 u 以计算方向相似度
        u_norm = torch.sqrt(torch.sum(u_in ** 2, dim=-1, keepdim=True))
        u_dir = u_in / (u_norm + 1e-8)
        
        # 准备 vec1, vec2
        v1 = vec1.view(1, 1, 2)
        v2 = vec2.view(1, 1, 2)
        
        # 计算点积 (Cosine Similarity)
        dot1 = torch.sum(u_dir * v1, dim=-1, keepdim=True) # (H, W, 1)
        dot2 = torch.sum(u_dir * v2, dim=-1, keepdim=True) # (H, W, 1)
        
        # 制作掩膜: 如果离 vec1 更近，mask=1，否则 mask=0
        # 注意：这里我们比较 abs(dot)，因为我们不希望把反向的颜色强行拉过来
        # 或者直接比较 dot (如果之前已经做了 K-Means 区分正反)
        # 鉴于之前的分析，vec1和vec2方向差异很大，直接比 dot 即可
        mask_v1 = (torch.abs(dot1) > torch.abs(dot2)).float()
        
        # 组合出每个像素的 target_v
        # target_v shape: (H, W, 2)
        target_v = v1 * mask_v1 + v2 * (1.0 - mask_v1)

        # ==========================================
        # [新增] 统计并打印比例
        # ==========================================
        total_pixels = u_in.shape[0] * u_in.shape[1]
        count_v1 = torch.sum(mask_v1).item()
        ratio_v1 = count_v1 / total_pixels
        ratio_v2 = 1.0 - ratio_v1
        
        # 打印当前比例 (比如: Vec1: 60.5%, Vec2: 39.5%)
        # 建议加上只在特定迭代打印的逻辑，否则刷屏太快
        print(f"[Hue Lock Stats] Vec1: {ratio_v1*100:.1f}% | Vec2: {ratio_v2*100:.1f}%")
        
        # ==========================================
        
        # ==========================================
        # 2. 应用你的算法 (Linear Mix + Renormalize)
        # ==========================================
        
        # original_mag 已经在上面算过了 (u_norm)
        original_mag = u_norm
        
        # 构建目标向量 (保持原始模长)
        u_target = original_mag * target_v
        
        # 混合方向
        u_mixed = (1.0 - strength) * u_in + strength * u_target
        
        # 恢复模长
        mixed_mag = torch.sqrt(torch.sum(u_mixed ** 2, dim=-1, keepdim=True))
        scale = original_mag / (mixed_mag + 1e-8)

        return u_mixed * scale
        

    def proximal_hue_lock_smart_fusion(u_in, vec1, vec2, strength, sharpness=20.0, print_stats=False):
        """
        智能双色调锁定 (Smart Hue Lock):
        1. 夹角内保护: 如果像素已经在 vec1 和 vec2 之间，保持方向不变 (Strength=0)。
        2. 夹角外修正: 如果像素跑偏，使用 Soft Fusion 柔和地拉回最近的边界。
        """
        if strength <= 1e-4: return u_in
        
        # ==========================================
        # 1. 准备数据 & 几何计算
        # ==========================================
        u_norm = torch.sqrt(torch.sum(u_in ** 2, dim=-1, keepdim=True))
        u_dir = u_in / (u_norm + 1e-8)
        
        # 确保 vec1, vec2 是单位向量并 reshape 为 (1, 1, 2)
        v1 = F.normalize(vec1, dim=0).view(1, 1, 2)
        v2 = F.normalize(vec2, dim=0).view(1, 1, 2)
        
        # ------------------------------------------
        # 判断 "Inside" (利用 2D 叉乘)
        # ------------------------------------------
        def cross_2d(a, b):
            # shape: (H, W, 2) or (1, 1, 2)
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        
        # 1. 确保 v1 -> v2 是逆时针方向 (保证我们算的是夹角内侧)
        # 计算 v1 到 v2 的叉乘
        cp_v1_v2 = v1[0,0,0] * v2[0,0,1] - v1[0,0,1] * v2[0,0,0]
        
        # 如果 v1 在 v2 右边 (顺时针)，交换它们用于检测 (物理上的 target 不用换)
        v_start, v_end = (v1, v2) if cp_v1_v2 >= 0 else (v2, v1)
        
        # 2. 判断 u 是否在 v_start 和 v_end 之间
        # 条件: u 在 start 左边 (cross > 0) AND u 在 end 右边 (cross < 0)
        # 注意: cross_2d 广播计算 (1,1,2) vs (H,W,2)
        cp_start_u = cross_2d(v_start, u_dir)
        cp_u_end   = cross_2d(u_dir, v_end) # 也就是 cross(end, u) < 0
        
        # mask_inside: (H, W) 布尔值
        # 这里的逻辑是: v_start X u > 0 (逆时针) 且 u X v_end > 0 (逆时针)
        # 只要 u 在 v_start->v_end 的扇形扫描路径上，两个叉乘应该同号(且为正)
        mask_inside = (cp_start_u >= -1e-4) & (cp_u_end >= -1e-4)
        
        # ==========================================
        # 2. Soft Fusion 目标计算 (对所有像素计算，备用)
        # ==========================================
        targets = torch.stack([vec1, vec2], dim=0).view(1, 1, 2, 2) 
        u_dir_expanded = u_dir.unsqueeze(2)
        similarity = torch.sum(u_dir_expanded * targets, dim=-1) # (H, W, 2)
        
        # 统计逻辑 (如果需要)
        if print_stats:
            # 统计所有像素的分布 (不管是否在夹角内)
            mask_v1_winner = (similarity[..., 0] > similarity[..., 1])
            count_v1 = torch.sum(mask_v1_winner).item()
            total = u_in.shape[0] * u_in.shape[1]
            
            # 也可以统计一下有多少在夹角内
            count_inside = torch.sum(mask_inside).item()
            
            print(f"[Stats] Inside Cone: {count_inside/total*100:.1f}% | Of Outside -> V1:{count_v1/total*100:.1f}%")

        # 计算 Softmax 权重
        weights = F.softmax(similarity * sharpness, dim=-1)
        
        # 合成目标方向
        target_v = torch.sum(weights.unsqueeze(-1) * targets, dim=2)
        target_v = target_v / (torch.norm(target_v, dim=-1, keepdim=True) + 1e-8)
        
        # ==========================================
        # 3. 混合应用 (关键修改: 动态 Strength)
        # ==========================================
        
        u_target_vector = u_norm * target_v
        
        # [关键] 构建强度遮罩
        # 如果 mask_inside 为 True，effective_strength = 0
        # 如果 mask_inside 为 False，effective_strength = strength
        mask_inside_float = mask_inside.float().unsqueeze(-1) # (H, W, 1)
        effective_strength = strength * (1.0 - mask_inside_float*0.995)
        
        # 混合
        # u = (1 - alpha) * u_in + alpha * u_target
        u_mixed = (1.0 - effective_strength) * u_in + effective_strength * u_target_vector
        
        # 恢复模长 (防止混合导致的变短)
        mixed_mag = torch.sqrt(torch.sum(u_mixed ** 2, dim=-1, keepdim=True))
        scale = u_norm / (mixed_mag + 1e-8)
        
        return u_mixed * scale

    def proximal_group_sigmoid(u_in, threshold, sharpness=20.0):
        magnitude = torch.sqrt(torch.sum(u_in ** 2, dim=-1, keepdim=True))
        gain = torch.sigmoid(sharpness * (magnitude - threshold))
        return u_in * gain
    
    def proximal_smooth_contrast(u_in, threshold, sharpness=50.0, limit=1.0):
        """
        更平滑的“拉开”算子，利用 Tanh 函数制造极陡峭的 S 曲线。
        """
        magnitude = torch.sqrt(torch.sum(u_in ** 2, dim=-1, keepdim=True))
        
        # 使用 Tanh 制造一个接近垂直的 S 曲线
        # 当 mag < threshold 时，activ -> 0
        # 当 mag > threshold 时，activ -> 1 (非常快)
        # (magnitude - threshold) 做了中心偏移
        activation = 0.5 * (torch.tanh(sharpness * (magnitude - threshold)) + 1.0)
        
        # 这里我们不做简单的乘法，而是做一个非线性映射
        # 把 activation (0~1) 映射到 (0 ~ limit)
        target_magnitude = activation * limit
        
        scale = target_magnitude / (magnitude + 1e-8)
        
        return u_in * scale

    # ==========================================
    # 5. 优化循环
    # ==========================================
    
    print(f"Starting Hue-Locked TV-L1 (Lock Strength={hue_locking_strength})...")
    
    for i in range(n_iters):
        # --- Step 1: Dual Update ---
        grad_u_bar = gradient_strict_joint(u_bar)
        p = p + sigma * (grad_u_bar - g_target)
        
        # Joint Projection (Vectorial TV)
        norm_p = torch.sqrt(torch.sum(p ** 2, dim=(2, 3))) # dim=(2,3) = sum over channels & xy
        norm_p = torch.maximum(norm_p, torch.tensor(1e-8, device=device))
        scale = torch.maximum(torch.ones_like(norm_p), norm_p / current_lambda)
        p = p / scale.unsqueeze(-1).unsqueeze(-1)
        
        # --- Step 2: Primal Update ---
        u_old = u.clone()
        div_p = divergence_strict_joint(p)
        u = u + tau * div_p

        # --- [关键修改] 插入 Hue Locking ---
        # 在去噪之前，先修正色调方向
        if hue_locking_strength > 0 and i > 1000:
            # u = u * 0.7 + 0.3 * proximal_hue_lock(u, dominant_hue_vec, strength=hue_locking_strength)
            # u = u * (1.0 - hue_locking_weight) + hue_locking_weight * proximal_hue_lock_smart_fusion(u, torch.flip(top_axes[0], dims=[0]), torch.flip(top_axes[1], dims=[0]), strength=hue_locking_strength, sharpness=20.0, print_stats=True)
            u = u * (1.0 - hue_locking_weight) + hue_locking_weight * proximal_hue_lock_smart_fusion(u, torch.flip(top_axes[1], dims=[0]), torch.flip(top_axes[2], dims=[0]), strength=hue_locking_strength, sharpness=0.1, print_stats=False)
            # hue_locking_weight = hue_locking_weight * 1.1
            # hue_locking_strength = np.clip(hue_locking_strength, 0.0, 0.5)

        # if i == n_iters - 1:
        #     u = proximal_hue_lock_dual_target(u, torch.flip(top_axes[0], dims=[0]), torch.flip(top_axes[1], dims=[0]), strength=1.0)
        

        # # # --- 动态调整阈值 & Sparsity ---
        # if i > 100:
        #     # 假设 calc_noise_min_energy 已经在外部定义或 import
        #     new_thresh = calc_noise_min_energy(u) 
        #     current_sparsity_threshold = 0.7 * current_sparsity_threshold + 0.3 * new_thresh
        #     print(f"Iter {i}: Auto Threshold updated to {current_sparsity_threshold:.5f}")

        #     if current_sparsity_threshold > 1e-6:
        #         # u = u * (1.0 - sparsity_weight) + sparsity_weight * proximal_smooth_contrast(u, threshold=current_sparsity_threshold, sharpness=1.0)
        #         u = u * (1.0 - sparsity_weight) + sparsity_weight * proximal_group_sigmoid(u, threshold=current_sparsity_threshold, sharpness=20.0)

        # if i == n_iters - 1:
        #     u = proximal_smooth_contrast(u, threshold=current_sparsity_threshold, sharpness=20.0)

        # 强制 Dirichlet 边界
        u[0, :, :] = 0; u[-1, :, :] = 0
        u[:, 0, :] = 0; u[:, -1, :] = 0

        # u[..., 1] = torch.abs(u[..., 1])
        # u[..., 0] = torch.abs(u[..., 0])

        # --- Step 3: Relaxation ---
        u_bar = u + theta * (u - u_old)
        
        # 监控收敛
        if i % 100 == 0:
            diff = torch.mean(torch.abs(u - u_old)).item()
            loss_c1 = torch.mean(torch.abs(grad_u_bar[:, :, 0, :] - g_target[:, :, 0, :])).item()
            loss_c2 = torch.mean(torch.abs(grad_u_bar[:, :, 1, :] - g_target[:, :, 1, :])).item()
            print(f"Iter {i}, Diff: {diff:.2e}, Loss_c1: {loss_c1:.6f}, Loss_c2: {loss_c2:.6f}")
            # print("max u_c1: {:.4f}, min u_c1: {:.4f}".format(torch.max(u[...,0]).item(), torch.min(u[...,0]).item())
            #       + ", max u_c2: {:.4f}, min u_c2: {:.4f}".format(torch.max(u[...,1]).item(), torch.min(u[...,1]).item()))
            
        # Lambda 退火
        if i > 0 and i % adaptive_iters == 0:
            current_lambda = current_lambda * 0.5
            print(f"Refining lambda -> {current_lambda:.4f}")
            # 保存中间结果调试
            # plt.imsave(f"../results/simulation/rgb_test/debug/c1_{i}.png", u[..., 0].cpu().numpy())
            # save mask == 0
            mask = (u[..., 0] < 0.1) & (u[..., 1] < 0.1)
            # plt.imsave(f"../results/simulation/rgb_test/debug/c1_mask_{i}.png", mask.cpu().numpy())
            # plt.imsave(f"../results/simulation/rgb_test/debug/c2_{i}.png", u[..., 1].cpu().numpy())

    u_max = torch.max(torch.abs(u))
    u = u / u_max

    top_axes = get_top_dominant_hue_axes(gradient_strict_joint(u), device=device, top_n_axes=2)
    plot_gradient_scatter(gradient_strict_joint(u), detected_axes=top_axes, save_path="../results/debug/gradient_distribution_reconstructed.png")


    return u[..., 0], u[..., 1]


# --- 1. 处理 L (MinMax) ---
    # 这一步是为了把亮度拉满到 0~1
    l_min = img_l_raw.min()
    l_max = img_l_raw.max()
    img_l = (img_l_raw - l_min) / (l_max - l_min + 1e-8)

    # --- 2. 处理 C1, C2 (MaxAbs 归一化) ---
    
    # A. 白平衡 (去均值)：必须做，确保 0 是无色
    img_c1 = img_c1_raw - img_c1_raw.mean()
    img_c2 = img_c2_raw - img_c2_raw.mean()
    
    # B. 找最大模长 (关键修改)
    # 我们不看 L 缩放了多少倍，我们要看 C 自己有多大
    # 找出当前画面里最鲜艳的那个像素的强度
    max_c1 = torch.max(torch.abs(img_c1))
    max_c2 = torch.max(torch.abs(img_c2))
    global_max_chroma = torch.maximum(max_c1, max_c2)
    
    # 防止除以 0 (如果是纯黑白图)
    if global_max_chroma < 1e-6:
        scale_factor = 1.0
    else:
        # C. 归一化并应用增益
        # 逻辑：把画面中最鲜艳的颜色映射到 saturation_gain (比如 0.8)
        # 这样保证了无论原始梯度数值多小，颜色都会被拉伸出来
        scale_factor = saturation_gain / global_max_chroma
    
    img_c1 = img_c1 * scale_factor
    img_c2 = img_c2 * scale_factor

    return img_l, img_c1, img_c2


def reconstruct_color_consistent(grad_x_rgb, grad_y_rgb, 
                                 lambda_lum=0.1, 
                                 lambda_chrom=2.0,
                                 lambda_sparsity=0.05,
                                 n_iters_lum=10000,
                                 n_iters_chrom=2000,
                                 adaptive_iters=2000,
                                 device='cuda'):
    """
    Color-Consistent Reconstruction (Luminance-Chrominance Decomposition).
    解决 RGB 独立重建导致的色偏和杂色问题。
    
    Args:
        grad_x_rgb: (H, W, 3) tensor, RGB 三通道的 X 梯度
        grad_y_rgb: (H, W, 3) tensor, RGB 三通道的 Y 梯度
        solver_func: 之前写好的 TV-L1 求解器函数 (接受 gx, gy, lambda)
        lambda_lum: 亮度通道的正则化 (保留细节，建议 0.1 - 0.2)
        lambda_chrom: 色差通道的正则化 (强力去噪，建议 1.0 - 5.0)
    """
    # 1. 拆分通道
    # 假设输入是 (H, W, 3)
    gx_r, gx_g, gx_b = grad_x_rgb[..., 0], grad_x_rgb[..., 1], grad_x_rgb[..., 2]
    gy_r, gy_g, gy_b = grad_y_rgb[..., 0], grad_y_rgb[..., 1], grad_y_rgb[..., 2]
    
    print("--- Phase 1: Gradient Transformation ---")
    # 2. 变换梯度到 L-C1-C2 空间
    # L (Luminance): 亮度 = (R+G+B)/3
    # C1 (Chrominance 1): 红绿差 = R - G
    # C2 (Chrominance 2): 蓝绿差 = B - G
    
    # 利用线性性: grad(L) = (grad(R) + grad(G) + grad(B)) / 3
    gx_l = (gx_r + gx_g + gx_b) / 3.0
    gy_l = (gy_r + gy_g + gy_b) / 3.0
    
    gx_c1 = gx_r - gx_g
    gy_c1 = gy_r - gy_g
    
    gx_c2 = gx_b - gx_g
    gy_c2 = gy_b - gy_g
    
    print("--- Phase 2: Solving Independent Channels ---")
    
    # 3. 分别重建 (关键在于 Lambda 的不同)
    
    # (A) 重建亮度 L: 使用小 Lambda，保留所有纹理细节
    print(f"Reconstructing Luminance (lambda={lambda_lum})...")
    # 注意：solver_func 返回的是 numpy 还是 tensor，这里统一转回 tensor 处理
    img_l = tv_l1_reconstruction_strict_neumann(gx_l, gy_l, lambda_tv=lambda_lum, n_iters=n_iters_lum, adaptive_iters=adaptive_iters, device=device)
    if isinstance(img_l, np.ndarray): img_l = torch.from_numpy(img_l).to(device)
    
    # # (B) 重建色差 C1 (R-G): 使用大 Lambda，强迫平滑
    # # 这步就是你想要的 "minimize TV(R-G)"
    # print(f"Reconstructing Chrominance R-G (lambda={lambda_chrom})...")
    # img_c1 = tv_l1_reconstruction_strict_neumann(gx_c1, gy_c1, lambda_tv=lambda_chrom, n_iters=n_iters, device=device)
    # if isinstance(img_c1, np.ndarray): img_c1 = torch.from_numpy(img_c1).to(device)

    # # (C) 重建色差 C2 (B-G): 使用大 Lambda
    # print(f"Reconstructing Chrominance B-G (lambda={lambda_chrom})...")
    # img_c2 = tv_l1_reconstruction_strict_neumann(gx_c2, gy_c2, lambda_tv=lambda_chrom, n_iters=n_iters, device=device)
    # if isinstance(img_c2, np.ndarray): img_c2 = torch.from_numpy(img_c2).to(device)

    # (B) & (C) 联合重建色度 (REPLACED)
    # 不再单独解 C1, C2，而是联合解，强制 Group Sparsity
    print(f"Reconstructing Coupled Chrominance (lambda={lambda_chrom})...")
    
    # img_c1, img_c2 = solve_coupled_chroma_strict_neumann(gx_c1, gy_c1, gx_c2, gy_c2, lambda_tv=lambda_chrom, lambda_sparsity=lambda_sparsity, n_iters=n_iters_chrom, adaptive_iters=adaptive_iters, device=device)
    img_c1, img_c2 = solve_coupled_chroma_strict_neumann(gx_c1, gy_c1, gx_c2, gy_c2, lambda_tv=lambda_chrom, lambda_sparsity=lambda_sparsity, n_iters=n_iters_chrom, adaptive_iters=adaptive_iters, device=device)
    if isinstance(img_c1, np.ndarray): img_c1 = torch.from_numpy(img_c1).to(device)
    if isinstance(img_c2, np.ndarray): img_c2 = torch.from_numpy(img_c2).to(device)

    l_max = torch.max(torch.abs(img_l))
    img_l = img_l
    img_c1 = img_c1 * l_max
    img_c2 = img_c2 * l_max

    # 4. 绘制 YCrCb Vector 二维直方图 (Cb vs Cr)
    plt.figure(figsize=(8, 6))
    
    # flatten() 将二维图像数据展平为一维数组以进行统计
    # range 设置为 [0, 1] 覆盖整个可能的色度范围
    plt.hist2d(img_c2.cpu().numpy().flatten(), img_c1.cpu().numpy().flatten(), bins=100, range=[[-8, 8], [-8, 8]], cmap='inferno')
    
    plt.colorbar(label='Pixel Count')
    plt.title(f'2D Histogram of YCrCb Vector (Cb vs Cr)\nSource: Reconstructed')
    plt.xlabel('Cb (Blue-Difference)')
    plt.ylabel('Cr (Red-Difference)')
    
    # 绘制中心十字线 (无色点)
    plt.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('../results/debug/ycrcb_vector_hist_reconstructed_before_inverse.png')
    
    print("--- Phase 3: Inverse Transformation ---")
    # 4. 逆变换回 RGB
    # 解方程组:
    # L = (R+G+B)/3
    # C1 = R - G  => R = C1 + G
    # C2 = B - G  => B = C2 + G
    # 代入第一式: 3L = (C1+G) + G + (C2+G) = C1 + C2 + 3G
    # => 3G = 3L - C1 - C2
    # => G = L - (C1 + C2) / 3

    img_g = img_l - (img_c1 + img_c2) / 3.0
    img_r = img_c1 + img_g
    img_b = img_c2 + img_g
    
    # 5. 堆叠并返回
    img_rgb = torch.stack([img_b, img_g, img_r], dim=-1)
    
    # 可选：简单的白平衡/归一化 (因为积分会有常数漂移)
    # 将每个通道的最小值对齐到 0 (或者根据背景色对齐)
    # img_rgb -= img_rgb.min() 
    
    return img_rgb.cpu().numpy(), img_l.cpu().numpy()


# def reconstruct_color_consistent(grad_x_rgb, grad_y_rgb, 
#                                  lambda_lum=0.1, 
#                                  lambda_chrom=2.0,
#                                  lambda_sparsity=0.05,
#                                  n_iters_lum=10000,
#                                  n_iters_chrom=2000,
#                                  adaptive_iters=2000,
#                                  device='cuda'):
#     """
#     Color-Consistent Reconstruction (YCbCr Decomposition).
#     使用 YCbCr 空间分离亮度 (Y) 和色度 (Cb, Cr)，解决 RGB 独立重建导致的色偏。
    
#     Forward Transform (Rec. 601):
#         Y  =  0.299*R + 0.587*G + 0.114*B
#         Cb = -0.168736*R - 0.331264*G + 0.5*B      (+ 0.5 偏移量在梯度中消失)
#         Cr =  0.5*R - 0.418688*G - 0.081312*B      (+ 0.5 偏移量在梯度中消失)
    
#     Args:
#         lambda_lum: 亮度 Y 的正则化权重 (建议小，如 0.1-0.2，保留纹理)
#         lambda_chrom: 色度 CbCr 的正则化权重 (建议大，如 1.0-5.0，强力去噪)
#     """
#     # 1. 拆分通道
#     # 假设输入是 (H, W, 3)
#     gx_r, gx_g, gx_b = grad_x_rgb[..., 0], grad_x_rgb[..., 1], grad_x_rgb[..., 2]
#     gy_r, gy_g, gy_b = grad_y_rgb[..., 0], grad_y_rgb[..., 1], grad_y_rgb[..., 2]
    
#     print("--- Phase 1: Gradient Transformation (RGB -> YCbCr) ---")
#     # 2. 变换梯度到 Y-Cb-Cr 空间
#     # 注意：常数项 (+0.5) 在求梯度时为 0，所以这里不需要加
    
#     # --- Y Channel (Luminance) ---
#     gx_y =  0.299000 * gx_r + 0.587000 * gx_g + 0.114000 * gx_b
#     gy_y =  0.299000 * gy_r + 0.587000 * gy_g + 0.114000 * gy_b
    
#     # --- Cb Channel (Blue-difference) ---
#     gx_cb = -0.168736 * gx_r - 0.331264 * gx_g + 0.500000 * gx_b
#     gy_cb = -0.168736 * gy_r - 0.331264 * gy_g + 0.500000 * gy_b
    
#     # --- Cr Channel (Red-difference) ---
#     gx_cr =  0.500000 * gx_r - 0.418688 * gx_g - 0.081312 * gx_b
#     gy_cr =  0.500000 * gy_r - 0.418688 * gy_g - 0.081312 * gy_b
    
#     print("--- Phase 2: Solving Independent Channels ---")
    
#     # 3. 分别重建
    
#     # (A) 重建亮度 Y: 使用小 Lambda，保留所有纹理细节
#     print(f"Reconstructing Luminance Y (lambda={lambda_lum})...")
#     img_y = tv_l1_reconstruction_strict_neumann(gx_y, gy_y, 
#                                                 lambda_tv=lambda_lum, 
#                                                 n_iters=n_iters_lum, 
#                                                 adaptive_iters=adaptive_iters, 
#                                                 device=device)
#     if isinstance(img_y, np.ndarray): img_y = torch.from_numpy(img_y).to(device)
    
#     # (B) 联合重建色度 Cb, Cr: 使用大 Lambda，强制平滑和边缘对齐
#     # 这里的 img_cb 和 img_cr 代表的是以 0 为中心的色差波动值 (delta from gray)
#     print(f"Reconstructing Coupled Chrominance Cb&Cr (lambda={lambda_chrom})...")
    
#     img_cb, img_cr = solve_coupled_chroma_strict_neumann(gx_cb, gy_cb, gx_cr, gy_cr, 
#                                                          lambda_tv=lambda_chrom, 
#                                                          lambda_sparsity=lambda_sparsity, 
#                                                          n_iters=n_iters_chrom, 
#                                                          adaptive_iters=adaptive_iters, 
#                                                          device=device)
    
#     img_cr = torch.abs(img_cr)
#     img_cb = torch.abs(img_cb)
#     if isinstance(img_cb, np.ndarray): img_cb = torch.from_numpy(img_cb).to(device)
#     if isinstance(img_cr, np.ndarray): img_cr = torch.from_numpy(img_cr).to(device)
    
#     print("--- Phase 3: Inverse Transformation (YCbCr -> RGB) ---")
#     # 4. 逆变换回 RGB
#     # 逆矩阵系数推导自你提供的正向系数 (标准 Rec.601 逆变换)
#     # R = Y + 1.402 * (Cr - 0.5)
#     # G = Y - 0.344136 * (Cb - 0.5) - 0.714136 * (Cr - 0.5)
#     # B = Y + 1.772 * (Cb - 0.5)
#     # 
#     # **关键点**：我们的 img_cb 和 img_cr 是通过梯度积分出来的，
#     # 它们本身就是相对于均值的偏差 (已经隐含了 -0.5 的效果)，
#     # 所以这里直接乘系数即可，不需要再减 0.5。
    
#     img_r = img_y + 1.402000 * img_cr
#     img_b = img_y + 1.772000 * img_cb
#     img_g = img_y - 0.344136 * img_cb - 0.714136 * img_cr
    
#     # 5. 堆叠并返回
#     img_rgb = torch.stack([img_r, img_g, img_b], dim=-1)
    
#     # 简单的归一化: 将暗部对齐到 0 (防止负值)
#     # 注意：梯度重建丢失了直流分量(DC)，所以整个图像的亮度基准可能是飘的
#     # 通常做法是减去最小值，或者基于先验知识(比如边缘像素应该是黑色的)来校正
#     # img_rgb = img_rgb - torch.min(img_rgb) 
    
#     # Clip 到合法范围
#     # img_rgb = torch.clamp(img_rgb, 0.0, 1.0)
    
#     return img_rgb.cpu().numpy()
