"""CPU numerical regression against the exact historical vanilla producer source."""
from __future__ import annotations
import ast, copy, importlib
from pathlib import Path
import torch
from anygsloc.artifacts import sha256
from .driver import OLD

PAIRS = {
 "projective_association":"projective_association",
 "projective_reconstruction":"projective_reconstruction",
 "projective_completion":"projective_completion",
 "triangulation":"triangulation", "tracks":"tracks",
 "observation_provider":"observation_provider", "render_certificate":"v7_render_certificate",
}
CHANGED = {
 "projective_association":{"build_projective_association_graph"},
 "projective_reconstruction":{"reconstruct_projective_anchors"},
 "projective_completion":{"build_projective_completion"},
 "triangulation":{"robust_triangulate_associations"},
}
def functions(path):
    return {x.name:x for x in ast.parse(Path(path).read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}

def source_audit():
    result={}
    for current,legacy in PAIRS.items():
        module=importlib.import_module("anygsloc.mapping."+current)
        a=functions(OLD/"code/evidence"/(legacy+".py"));b=functions(module.__file__)
        changed={k for k in a.keys()&b.keys() if ast.dump(a[k],include_attributes=False)!=ast.dump(b[k],include_attributes=False)}
        if changed != CHANGED.get(current,set()):
            raise ValueError("unexpected historical kernel drift: "+current+" "+str(changed))
        result[current]={"legacy_sha":sha256(OLD/"code/evidence"/(legacy+".py")),
                         "current_sha":sha256(module.__file__),"changed_functions":sorted(changed)}
    return result

def historic_function(module_name, function_name, overrides=None):
    """Execute only the historical function AST, with verified unchanged helper functions."""
    module=importlib.import_module("anygsloc.mapping."+module_name)
    node=functions(OLD/"code/evidence"/(PAIRS[module_name]+".py"))[function_name]
    namespace=dict(vars(module));namespace.update(overrides or {})
    exec(compile(ast.Module(body=[node],type_ignores=[]),"<frozen-history-regression>","exec"),namespace)
    return namespace[function_name]

def assert_same(a,b):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype or not torch.allclose(a,b,atol=0,rtol=0,equal_nan=True):
            raise AssertionError("tensor regression")
    elif isinstance(a,dict):
        if set(a)!=set(b): raise AssertionError("dictionary keys differ")
        for k in a: assert_same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError("length differs")
        for x,y in zip(a,b):assert_same(x,y)
    else:
        if a!=b: raise AssertionError("scalar differs")

def compare_candidates(a,b):
    for key in ("anchor_xyz","anchor_features","anchor_position_covariance","identity_reliability",
                "geometry_reliability","projective_anchor_observations"):
        assert_same(a[key],b[key])

def numerical_regression(payload):
    source_audit()
    from anygsloc.mapping.observation_provider import GaussianRenderObservationProvider
    from anygsloc.mapping import projective_association as ca, projective_reconstruction as cr
    from anygsloc.mapping import projective_completion as cc
    from anygsloc.mapping import triangulation as ct
    from experiments.cambridge_input_matrix.contracts import MatrixObservationProvider
    from experiments.cambridge_input_matrix import projective_association as ea
    from experiments.cambridge_input_matrix import projective_reconstruction as er
    from experiments.cambridge_input_matrix import projective_completion as ec
    old_tri=historic_function("triangulation","robust_triangulate_associations")
    old_recon=historic_function("projective_reconstruction","reconstruct_projective_anchors",
                                {"robust_triangulate_associations":old_tri})
    old_assoc=historic_function("projective_association","build_projective_association_graph")
    old_completion=historic_function("projective_completion","build_projective_completion",
                                    {"reconstruct_projective_anchors":old_recon})
    provider=GaussianRenderObservationProvider(payload); adapted=MatrixObservationProvider(payload)
    old=old_assoc(provider,device="cpu");new=ca.build_projective_association_graph(provider,device="cpu")
    exp=ea.build_projective_association_graph(adapted,device="cpu")
    for key in ("tracks","component_statistics","query_bins","pair_sidecar"):
        # Runtime timing diagnostics are excluded; all actual numerical sidecar columns retained.
        aa=copy.deepcopy(old[key]);bb=copy.deepcopy(new[key]);ee=copy.deepcopy(exp[key])
        if isinstance(aa,dict):
            for val in (aa,bb,ee):
                for k in list(val):
                    if "timing" in k:val.pop(k)
        assert_same(aa,bb);assert_same(aa,ee)
    a=old_recon(provider,old,parallel_workers=1)
    b=cr.reconstruct_projective_anchors(provider,new,parallel_workers=1)
    e=er.reconstruct_projective_anchors(adapted,exp,parallel_workers=1)
    compare_candidates(a,b);compare_candidates(a,e)
    for graph in (old,new,exp):
        graph["tracks"]["query_index"]=torch.empty(0,dtype=torch.long)
        graph["tracks"]["keypoint_index"]=torch.empty(0,dtype=torch.long)
    opts=dict(voxel_size_m=.05,alpha_minimum=.05,minimum_similarity=.7,minimum_camera_families=1,device="cpu")
    a=old_completion(provider,old,**opts);b=cc.build_projective_completion(provider,new,**opts)
    e=ec.build_projective_completion(adapted,exp,**opts)
    compare_candidates(a,b);compare_candidates(a,e)
    if torch.cuda.is_initialized():raise AssertionError("CPU regression initialized CUDA")
    return {"exact_tensor_equality":True,"association_reconstruction_completion":True,"cuda_initialized":False}

