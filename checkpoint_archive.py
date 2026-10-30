"""Lossless block-difference archives for duplicated immutable checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

BLOCK=65536


def verify_archive(archive,destination=None):
    archive=Path(archive).resolve()
    output=Path(destination) if destination is not None else None
    temporary=output.with_name(output.name+'.restore.tmp') if output else None
    if output and output.exists():raise FileExistsError(output)
    try:
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('manifest.json'))
            base=(archive.parent/manifest['base']).resolve()
            base_hash=hashlib.sha256();target_hash=hashlib.sha256();total=0
            with base.open('rb') as source:
                writer=temporary.open('wb') if temporary else None
                try:
                    for index in range((manifest['size']+BLOCK-1)//BLOCK):
                        chunk=source.read(BLOCK);base_hash.update(chunk)
                        data=z.read(f'blocks/{index}') if index in manifest['changed_blocks'] else chunk
                        target_hash.update(data);total+=len(data)
                        if writer:writer.write(data)
                    if source.read(1):raise ValueError('Base checkpoint size differs')
                finally:
                    if writer:writer.close()
            if base_hash.hexdigest()!=manifest['base_sha256']:raise ValueError('Base checkpoint hash differs')
            if total!=manifest['size'] or target_hash.hexdigest()!=manifest['target_sha256']:raise ValueError('Restored checkpoint hash differs')
        if output:os.replace(temporary,output)
        return manifest
    finally:
        if temporary:temporary.unlink(missing_ok=True)


def archive_checkpoint(base,target,replace=False):
    base,target=Path(base).resolve(),Path(target).resolve()
    if base==target or base.stat().st_size!=target.stat().st_size:raise ValueError('Use different equal-sized files')
    archive=target.with_name(target.name+'.delta.zip')
    if archive.exists():raise FileExistsError(archive)
    base_hash=hashlib.sha256();target_hash=hashlib.sha256();changed=[]
    with base.open('rb') as a,target.open('rb') as b,zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        index=0
        while True:
            left,right=a.read(BLOCK),b.read(BLOCK)
            if not right:break
            base_hash.update(left);target_hash.update(right)
            if left!=right:changed.append(index);z.writestr(f'blocks/{index}',right)
            index+=1
        manifest={'format':'checkpoint-block-delta-v1','block_size':BLOCK,'base':os.path.relpath(base,archive.parent),
                  'target':target.name,'size':target.stat().st_size,'base_sha256':base_hash.hexdigest(),
                  'target_sha256':target_hash.hexdigest(),'changed_blocks':changed}
        z.writestr('manifest.json',json.dumps(manifest))
    verify_archive(archive)
    if replace:target.unlink()
    return {'archive':str(archive),'original_bytes':manifest['size'],'archive_bytes':archive.stat().st_size,
            'verified_sha256':manifest['target_sha256'],'original_replaced':replace}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('archive');a.add_argument('base');a.add_argument('target');a.add_argument('--replace',action='store_true')
    r=sub.add_parser('restore');r.add_argument('archive')
    args=p.parse_args()
    if args.command=='archive':print(json.dumps(archive_checkpoint(args.base,args.target,args.replace),ensure_ascii=False))
    else:
        path=Path(args.archive);destination=path.with_name(path.name.removesuffix('.delta.zip'))
        verify_archive(path,destination);print(str(destination))
