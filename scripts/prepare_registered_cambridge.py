"""Explicit GreatCourt exception and mapping-only Vanilla SfM initialization."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
from common.hashing import sha256_file
from data.colmap import read_extrinsics_binary, read_intrinsics_binary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    for scene in ['GreatCourt','ShopFacade','StMarysChurch']:
        source=Path('/mnt/pool/sqy/Cambridge_stdloc')/scene
        ims=read_extrinsics_binary(str(source/'sparse/0/images.bin'))
        cams=read_intrinsics_binary(str(source/'sparse/0/cameras.bin'))
        names={r.name:r for r in ims.values()}
        lines=(source/'dataset_train.txt').read_text().splitlines()
        train=[x.split()[0] for x in lines if x.strip() and x.split()[0].lower().endswith(('.png','.jpg','.jpeg'))]
        missing=[n for n in train if n not in names or not (source/'processed'/n).exists()]
        if missing != (['seq5/frame00297.png'] if scene=='GreatCourt' else []):
            raise ValueError(f'unexpected missing input: {scene} {missing}')
        valid=[n for n in train if n not in missing]
        if scene=='GreatCourt':
            dataset=args.root/'datasets'/scene
            dataset.mkdir(parents=True)
            for item in source.iterdir():
                if item.name!='dataset_train.txt': (dataset/item.name).symlink_to(item)
            (dataset/'dataset_train.txt').write_text('\n'.join(x for x in lines if not x.strip() or x.split()[0] not in missing)+'\n')
            (dataset/'registration_exception.json').write_text(json.dumps({'source':str(source),
                'excluded_mapping':missing,'reason':'absent COLMAP pose and processed image; raw preserved',
                'registered_mapping_count':len(valid),'test_modified':False,
                'source_train_sha256':sha256_file(source/'dataset_train.txt')},indent=2))
        output=args.root/'vanilla_inputs'/scene
        (output/'images').mkdir(parents=True)
        sparse=output/'sparse/0';sparse.mkdir(parents=True)
        image_lines=[]; camera_lines=[];records=[]
        for n in valid:
            r=names[n];c=cams[r.camera_id]
            if c.model!='SIMPLE_RADIAL':raise ValueError(c.model)
            f,cx,cy,k=c.params;flat=n.replace('/','__')
            (output/'images'/flat).symlink_to(source/'processed'/n)
            camera_lines.append(f'{c.id} PINHOLE {c.width} {c.height} {f:.17g} {f:.17g} {cx:.17g} {cy:.17g}')
            image_lines.extend([' '.join([str(r.id),*[f'{v:.17g}' for v in np.r_[r.qvec,r.tvec]],str(c.id),flat]),''])
            records.append({'source_name':n,'prepared_name':flat,'sha256':sha256_file(source/'processed'/n)})
        (sparse/'cameras.txt').write_text('\n'.join(dict.fromkeys(camera_lines))+'\n')
        (sparse/'images.txt').write_text('\n'.join(image_lines)+'\n')
        ids={names[n].id for n in valid};kept=[];dropped=0
        with (source/'sparse/0/points3D.bin').open('rb') as stream:
            count=struct.unpack('<Q',stream.read(8))[0]
            for _ in range(count):
                values=struct.unpack('<QdddBBBd',stream.read(43))
                length=struct.unpack('<Q',stream.read(8))[0]
                track=struct.unpack('<'+'ii'*length,stream.read(8*length))
                if length>=2 and set(track[::2])<=ids:
                    kept.append(' '.join(str(v) for v in values))
                else:dropped+=1
        if not kept:raise ValueError('empty mapping-only SfM initialization')
        (sparse/'points3D.txt').write_text('\n'.join(kept)+'\n')
        (output/'input_manifest.json').write_text(json.dumps({'scene':scene,'image_domain':'rectified',
            'images':records,'excluded_mapping':missing,'test_used':False,
            'point_policy':'retain only points whose complete original track is mapping-only, length >= 2',
            'retained_points':len(kept),'dropped_points':dropped,
            'source_sparse_sha256':{p.name:sha256_file(p) for p in (source/'sparse/0').glob('*.bin')}},indent=2))
        print(scene,len(valid),len(kept),'COMPLETE',flush=True)


if __name__=='__main__':main()
