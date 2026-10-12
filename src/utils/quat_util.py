import torch
import torch.nn.functional as F

# ------------------------------
# quaternion multiply
# ------------------------------
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

# ------------------------------
# quaternion -> rotation matrix
# ------------------------------
def quat_to_mat(q):
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),
        2*(x*y - z*w),
        2*(x*z + y*w),

        2*(x*y + z*w),
        1 - 2*(x*x + z*z),
        2*(y*z - x*w),

        2*(x*z - y*w),
        2*(y*z + x*w),
        1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1,3,3)

    return R

# ------------------------------
# quaternion distance
# ------------------------------
def quat_distance(q1, q2):

    dot = torch.abs((q1 * q2).sum(-1))
    dot = torch.clamp(dot, -1.0, 1.0)

    return 2 * torch.acos(dot)

def quat_conjugate(q):
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out

def quat_rotate_vec(q, v):
    q = F.normalize(q, dim=-1)
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q_xyz, t, dim=-1)

def pose_inv_wxyz(pose):
    quat_inv = quat_conjugate(F.normalize(pose[..., 3:7], dim=-1))
    pos_inv = quat_rotate_vec(quat_inv, -pose[..., :3])
    return torch.cat((pos_inv, quat_inv), dim=-1)

def compose_pose_wxyz(lhs, rhs):
    lhs_pos = lhs[..., :3]
    lhs_quat = F.normalize(lhs[..., 3:7], dim=-1)
    rhs_pos = rhs[..., :3]
    rhs_quat = F.normalize(rhs[..., 3:7], dim=-1)

    pos = lhs_pos + quat_rotate_vec(lhs_quat, rhs_pos)
    quat = F.normalize(quat_mul(lhs_quat, rhs_quat), dim=-1)
    return torch.cat((pos, quat), dim=-1)

def axis_angle_to_quat(axis, angle, B=None):
    angle = torch.as_tensor(angle, device=axis.device, dtype=axis.dtype)
    if B is None:
        B = 1 if angle.ndim == 0 else int(angle.numel())

    axis = F.normalize(axis, dim=-1)
    if axis.ndim == 1:
        axis = axis.expand(B, 3)
    elif axis.shape[0] != B:
        raise ValueError(f"axis batch size {axis.shape[0]} does not match B={B}.")

    angle = angle.reshape(-1)
    if angle.numel() == 1:
        angle = angle.expand(B)
    elif angle.numel() != B:
        raise ValueError(f"angle size {angle.numel()} does not match B={B}.")

    half = angle / 2
    w = torch.cos(half).unsqueeze(-1)
    xyz = axis * torch.sin(half).unsqueeze(-1)

    quat = torch.cat([w, xyz], dim=-1)
    return F.normalize(quat, dim=-1)
