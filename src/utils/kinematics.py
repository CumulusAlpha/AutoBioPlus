
import torch

def qpos_linear_interpolate(
    start_qpos,
    target_qpos,
    num_steps,
):
    """
    start_qpos:  (B, D)
    target_qpos: (B, D)

    return:
        qpos_traj: (B, T, D), T = num_steps + 1
    """

    device = start_qpos.device
    dtype = start_qpos.dtype

    t = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    t = t[None, :, None]

    return (1.0 - t) * start_qpos[:, None, :] + t * target_qpos[:, None, :]

def batched_slerp(q0, q1, t):
    """
    q0, q1: (B, 4)
    t: (T,)
    return: (B, T, 4)
    """

    device = q0.device
    dtype = q0.dtype

    # --------------------------------------
    # Compute the dot product
    # --------------------------------------
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)  # (B,1)

    # Use the shorter arc
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot)

    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)  # (B,1)
    sin_theta = torch.sin(theta)

    # reshape for broadcasting with t: (B,1,1) vs (1,T,1)
    theta = theta[:, None, :]
    sin_theta = sin_theta[:, None, :]

    # --------------------------------------
    # Broadcast the time parameter
    # --------------------------------------
    t = t.to(device=device, dtype=dtype)
    t = t[None, :, None]  # (1,T,1)

    # --------------------------------------
    # Avoid division by zero
    # --------------------------------------
    small = sin_theta < 1e-6

    w1 = torch.sin((1.0 - t) * theta) / (sin_theta + 1e-8)
    w2 = torch.sin(t * theta) / (sin_theta + 1e-8)

    out = w1 * q0[:, None, :] + w2 * q1[:, None, :]

    # --------------------------------------
    # Fall back to linear interpolation for small angles
    # --------------------------------------
    out = torch.where(
        small,
        (1 - t) * q0[:, None, :] + t * q1[:, None, :],
        out,
    )

    # --------------------------------------
    # Normalize
    # --------------------------------------
    out = out / torch.linalg.norm(out, dim=-1, keepdim=True)

    return out

def batched_interpolate2(
    start_pos,
    end_pos,
    start_quat,
    end_quat,
    num_steps,
    height=None,
):
    """
    start_pos:  (B, 3)
    end_pos:    (B, 3)
    start_quat: (B, 4)
    end_quat:   (B, 4)

    return:
        pos_traj:  (B, T, 3)
        quat_traj: (B, T, 4)
    """

    B = start_pos.shape[0]
    T = num_steps + 1
    device = start_pos.device
    dtype = start_pos.dtype

    # --------------------------------------
    # 1️⃣ Compute the horizontal basis vector
    # --------------------------------------
    horizon_vec = end_pos[:, :2] - start_pos[:, :2]  # (B,2)
    horizon_dis = torch.linalg.norm(horizon_vec, dim=1, keepdim=True)  # (B,1)

    eps = 1e-8
    basis1_xy = horizon_vec / (horizon_dis + eps)

    basis1 = torch.cat(
        [basis1_xy, torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )  # (B,3)

    basis2 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).repeat(B, 1)

    # --------------------------------------
    # 2️⃣ Parabolic arc height
    # --------------------------------------
    if height is None:
        height = horizon_dis / 4.0
    else:
        height = torch.ones_like(horizon_dis) * height

    z1 = start_pos[:, 2]
    z2 = end_pos[:, 2]
    d = horizon_dis[:, 0]

    # Quadratic coefficients (analytical solution)
    a = -4.0 * height[:, 0] / (d**2 + eps)
    b = (z2 - z1 + 4.0 * height[:, 0]) / (d + eps)
    c = z1

    # --------------------------------------
    # 3️⃣ Time parameter
    # --------------------------------------
    s = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    x_eval = s.unsqueeze(0) * d.unsqueeze(1)  # (B,T)

    z_eval = (
        a.unsqueeze(1) * x_eval**2
        + b.unsqueeze(1) * x_eval
        + c.unsqueeze(1)
    )

    # --------------------------------------
    # 4️⃣ Combine the components into 3D coordinates
    # --------------------------------------
    origin = start_pos.clone()
    origin[:, 2] = 0.0

    pos_traj = (
        x_eval.unsqueeze(-1) * basis1.unsqueeze(1)
        + z_eval.unsqueeze(-1) * basis2.unsqueeze(1)
        + origin.unsqueeze(1)
    )

    # --------------------------------------
    # 5️⃣ Orientation interpolation
    # --------------------------------------
    quat_traj = batched_slerp(start_quat, end_quat, s)

    return pos_traj, quat_traj

def batched_interpolate_linear(
    start_pos,
    end_pos,
    start_quat,
    end_quat,
    num_steps,
):
    """
    Vectorized interpolation (pos + quaternion slerp)
    """

    device = start_pos.device
    dtype = start_pos.dtype

    # (T+1,)
    t = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)

    # -----------------------------
    # Position interpolation
    # -----------------------------
    # reshape for broadcasting
    t_pos = t[None, :, None]  # (1, T+1, 1)

    start_pos = start_pos[:, None, :]  # (B,1,3)
    end_pos = end_pos[:, None, :]      # (B,1,3)

    pos_traj = (1 - t_pos) * start_pos + t_pos * end_pos  # (B,T+1,3)

    # -----------------------------
    # Quaternion slerp (batched)
    # -----------------------------
    quat_traj = batched_slerp(start_quat, end_quat, t)  # (B,T+1,4)

    return pos_traj, quat_traj
