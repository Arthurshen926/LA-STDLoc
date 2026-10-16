"""Small stage identity and publication checks; no numerical work."""
import hashlib,json
from pathlib import Path
MODE="real_gs_assisted"
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def identity(root,scene,stage,selection=None):
    return dict(stage=stage,scene=scene,input_mode=MODE,selection=selection,
        protocol_sha256=sha(Path(root)/"protocol.json"))
def check_identity(value,root,scene,stage,selection=None):
    for key,expected in identity(root,scene,stage,selection).items():
        if value.get(key)!=expected:raise ValueError("artifact identity differs: "+key)
def stage_path(root,scene,stage,selection=None):
    root=Path(root)
    return {
        "prepare":root/"protocol.json","audit":root/"audit"/(str(scene)+".json"),
        "sample":root/"auxiliary"/str(scene)/"summary.json",
        "build":root/"maps"/str(scene)/"full/summary.json",
        "select":root/"maps"/str(scene)/"native_compact/summary.json",
        "evaluate":root/"cells"/str(scene)/(str(selection)+".json"),
        "report":root/"report.json"}[stage]
def stage_name(scene,stage,selection=None):
    return "_".join(str(x) for x in (stage,scene,selection) if x is not None)
def exact_ref(value,path,rehash=False):
    path=Path(path).resolve()
    if value.get("path")!=str(path):raise ValueError("noncanonical artifact path")
    st=path.stat()
    if any(value.get(k)!=v for k,v in dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino).items()):
        raise ValueError("artifact stat differs")
    if rehash and value.get("sha256")!=sha(path):raise ValueError("artifact hash differs")
    return path
def checked_stage(root,scene,stage,selection=None):
    path=stage_path(root,scene,stage,selection)
    value=read(path);check_identity(value,root,scene,stage,selection)
    if value.get("complete") is not True:raise ValueError("stage incomplete")
    receipt=read(Path(root)/"execution"/(stage_name(scene,stage,selection)+".json"))
    check_identity(receipt,root,scene,stage,selection)
    if receipt.get("success") is not True or receipt.get("returncode")!=0:
        raise ValueError("stage predecessor not successful")
    exact_ref(receipt["summary"],path,rehash=True)
    return value
def checked_map(root,scene,selection):
    stage="build" if selection=="full" else "select"
    value=checked_stage(root,scene,stage)
    key="full" if selection=="full" else "map"
    exact_ref(value[key],Path(root)/"maps"/scene/selection/"deployment_map.pt")
    return value
def check_aux_lineage(aux,root,scene,source,prior_sha):
    check_identity(aux,root,scene,"sample")
    if aux.get("raw_cache_sha256")!=source["raw_cache"]["sha256"] or aux.get("prior_sha256")!=prior_sha:
        raise ValueError("auxiliary raw/prior lineage differs")
def check_state_lineage(state,root,scene,source,prior_sha,aux_sha):
    value=state["provenance"];check_identity(value,root,scene,"build")
    expected=dict(raw_cache_sha256=source["raw_cache"]["sha256"],
        raw_full_sha256=source["raw_full"]["sha256"],raw_state_sha256=source["raw_state"]["sha256"],
        raw_construction_sha256=source["raw_construction"]["sha256"],
        prior_sha256=prior_sha,auxiliary_sha256=aux_sha)
    for key,val in expected.items():
        if value.get(key)!=val:raise ValueError("state source lineage differs: "+key)
def checked_cell(root,scene,selection):
    value=checked_stage(root,scene,"evaluate",selection)
    summary=checked_map(root,scene,selection)
    mi=summary["full" if selection=="full" else "map"]
    if value["map"]!=mi:raise ValueError("cell map differs from own selection summary")
    for name in ("results.json","manifest.json","summary.json"):
        exact_ref(value["files"][name],Path(root)/"evaluations"/scene/selection/name,rehash=True)
    return value
