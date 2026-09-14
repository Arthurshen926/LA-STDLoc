import numpy as np
import pytest
from plyfile import PlyData, PlyElement
from priors.models import GaussianModel3D


@pytest.mark.parametrize('rows',[1,2])
def test_unaligned_byte_normal_ply_loads_on_cpu(tmp_path,rows):
    # Vectorized Long-LRM outputs can contain byte-valued normals. Their
    # structured-record stride is not necessarily divisible by four.
    fields=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity',
            'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    vertices=np.zeros(rows,dtype=[(n,'u1' if n in ['nx','ny','nz'] else '<f4') for n in fields])
    vertices['opacity']=np.arange(1,rows+1);vertices['rot_0']=1
    path=tmp_path/'unaligned.ply';PlyData([PlyElement.describe(vertices,'vertex')]).write(path)
    model=GaussianModel3D(0,device='cpu');model.load_ply(path,loc_feature_dim=0)
    np.testing.assert_array_equal(model._opacity.numpy().ravel(),np.arange(1,rows+1))
