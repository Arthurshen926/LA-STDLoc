"""Query-semantic-on protocol adapter; native feature, solver and refinement stay unchanged."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()

def once(text, old, new):
    if text.count(old)!=1: raise ValueError("frozen source anchor changed: "+old[:70])
    return text.replace(old,new)

def mask_rgb(name, image, masks, query_names):
    """Same object/sky/geometry AND as native ULF query preprocessing."""
    import torch
    import torch.nn.functional as F
    if name not in query_names: raise ValueError("non-query image passed to semantic adapter")
    layers=masks[name]
    if len(layers)!=3: raise ValueError("three mask channels required")
    converted=[]
    for value in layers:
        value=torch.as_tensor(value)
        if value.ndim!=2: raise ValueError("mask must be two-dimensional")
        value=value.bool()
        if tuple(value.shape)!=tuple(image.shape[-2:]):
            value=F.interpolate(value.float()[None,None],size=image.shape[-2:],mode="nearest")[0,0]
        converted.append(value.bool().to(image.device))
    valid=converted[0]&converted[1]&converted[2]
    return image*valid[None]

def std_text(original):
    text=once(original,
        '        query_image = camera_info.original_image.to("cuda")\n',
        '        query_image = camera_info.original_image.to("cuda")\n'
        '        query_image = _QUERY_MASK_RGB(camera_info.image_name, query_image)\n')
    text=once(text,'    print("Output path:", output_path)\n',
        '    output_path = _CONTROLLED_OUTPUT\n    print("Output path:", output_path)\n')
    ast.parse(text)
    return text

def wrapper_text(original):
    text=once(original,
        "            if a.phase != 'train':\n"
        "                raise RuntimeError('query semantic file must not be read')\n"
        "            raw=original_load(stream,*args,**kwargs) if a.mapping_semantics=='on' else None\n",
        "            if a.phase != 'evaluate':\n"
        "                raise RuntimeError('query adapter only permits evaluation')\n"
        "            raw=original_load(stream,*args,**kwargs)\n")
    text=once(text,
        "        os.path.exists=lambda path: False if str(path).endswith('masks.pkl') else original_exists(path)\n",
        "        query_masks=None\n"
        "        if a.method=='STDLoc':\n"
        "            with open(a.source/'processed/masks.pkl','rb') as query_stream:\n"
        "                query_masks=TrainingMasks(original_load(query_stream))\n")
    text=once(text,
        "        namespace=runpy.run_path(str(a.repo/entry),run_name='__main__')\n",
        "        namespace=_CONTROLLED_NATIVE_RUN(str(a.repo/entry),a.method,query_masks)\n")
    text=once(text,"            mapping_semantics=a.mapping_semantics,query_semantics=False,seed=2026,\n",
        "            mapping_semantics=a.mapping_semantics,query_semantics=True,seed=2026,\n")
    text=once(text,"        value['build_stages']=build_stages\n",
        "        value['build_stages']=build_stages\n"
        "        value['query_semantic_variant']='STDLoc external RGB mask preprocessing; ULFLoc native query-mask branch enabled'\n"
        "        value['query_semantic_timing_boundary']='precomputed mask archive/preprocessing outside native localize timing; no segmentation network; native GT diagnostic inside core retained'\n")
    ast.parse(text)
    return text

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract",type=Path,required=True)
    a=ap.parse_args()
    c=json.loads(a.contract.read_text())
    if c.get("schema")!="native_baseline_query_semantic_on_cell_v1" or c.get("query_semantics") is not True:
        raise ValueError("unfrozen query-on cell")
    if c["adapter_sha256"]!=sha(__file__):raise ValueError("query adapter drift")
    argv=c["wrapper_argv"]
    def arg(flag):
        if argv.count(flag)!=1:raise ValueError("ambiguous wrapper flag "+flag)
        return argv[argv.index(flag)+1]
    if arg("--phase")!="evaluate" or arg("--method")!=c["method"] or c["method"] not in ("STDLoc","ULFLoc"):
        raise ValueError("query adapter cannot train or switch methods")
    if arg("--source")!=c["source"] or arg("--receipt")!=c["receipt"]:
        raise ValueError("query identity detached")
    if len(set(c["query_names"]))!=len(c["query_names"]):raise ValueError("duplicate query names")
    for ref in (c["wrapper"],c["native_entry"]):
        if sha(ref["path"])!=ref["sha256"]:raise ValueError("native source drift")
    native_source=Path(c["native_entry"]["path"])
    def execute_native(path,method,masks):
        if Path(path).resolve()!=native_source.resolve():raise ValueError("unexpected native entry")
        if method=="ULFLoc":
            return runpy.run_path(path,run_name="__main__")
        out=Path(c["native_output"])
        if out.exists():raise FileExistsError(out)
        scope={"__name__":"__main__","__file__":path,"__package__":"",
               "_CONTROLLED_OUTPUT":str(out),
               "_QUERY_MASK_RGB":lambda name,image:mask_rgb(name,image,masks,set(c["query_names"]))}
        exec(compile(std_text(native_source.read_text()),path,"exec"),scope)
        return scope
    scope={"__name__":"__main__","__file__":c["wrapper"]["path"],"__package__":"",
           "_CONTROLLED_NATIVE_RUN":execute_native}
    sys.argv=[c["wrapper"]["path"]]+argv
    exec(compile(wrapper_text(Path(c["wrapper"]["path"]).read_text()),c["wrapper"]["path"],"exec"),scope)

if __name__=="__main__":main()
