"""Internal Gaussian surface evidence; never an absolute geometry certificate."""
import torch


def surface_row_evidence(expected_depth, median_depth, alpha, primitive_normal, depth_normal):
    z,m,a=[torch.as_tensor(x).float() for x in [expected_depth,median_depth,alpha]]
    n,s=[torch.as_tensor(x).float() for x in [primitive_normal,depth_normal]]
    if z.ndim!=1 or m.shape!=z.shape or a.shape!=z.shape or n.shape!=(len(z),3) or s.shape!=n.shape:raise ValueError('surface evidence row shapes differ')
    norm_n=torch.linalg.vector_norm(n,dim=1);norm_s=torch.linalg.vector_norm(s,dim=1)
    valid=torch.isfinite(z)&torch.isfinite(m)&torch.isfinite(a)&torch.isfinite(n).all(1)&torch.isfinite(s).all(1)&(z>0)&(m>0)&(a>=.5)&(norm_n>.1)&(norm_s>.1)
    relative=(z-m).abs()/z.clamp_min(1e-8)
    # Normals are both world-frame in the installed gsplat implementation.
    # Compare unoriented axes; signs are not a physical quality label.
    cosine=(n*s).sum(1).abs()/(norm_n*norm_s).clamp_min(1e-8)
    angle=torch.rad2deg(torch.acos(cosine.clamp(0,1)))
    conflict=valid&(relative>.02)&(angle>45.)
    return {'valid':valid,'conflict':conflict,'relative_depth_disagreement':relative,
            'normal_axis_disagreement_deg':angle}
