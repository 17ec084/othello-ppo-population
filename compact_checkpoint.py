"""Lossless sparse actor/Adam checkpoints relative to an immutable initial pool.

Only opened actor coordinates can change. Store those coordinates, the critic,
and all control/RNG state. Stream gzip directly to the atomic temporary file so
an uncompressed full checkpoint is never required on disk.
"""
import copy
import gzip
import hashlib
import shutil
from pathlib import Path
import torch
from runtime_io import replace_with_retry

FORMAT='sparse-policy-adam-v1'
FULL_FORMAT='trained-models-adam-v1'


def cpu_copy(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_copy(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_copy(v) for v in value)
    return value


def encode(data, masks):
    if masks is None:
        packed={k:cpu_copy(v) for k,v in data.items() if k!='models'}
        packed['checkpoint_format']=FULL_FORMAT
        changed=data.get('trained_model_ids',range(len(data['models'])))
        packed['model_updates']={int(i):cpu_copy(data['models'][i]) for i in changed}
        return packed
    indices={n:m.flatten().nonzero().flatten().cpu() for n,m in masks.items()}
    packed={k:v for k,v in data.items() if k not in ('models','optimizers')}
    packed['checkpoint_format']=FORMAT;packed['active_indices']=indices
    packed['model_deltas']=[{n:v.detach().flatten()[indices[n].to(v.device)].cpu().clone()
                             for n,v in model.items()} for model in data['models']]
    names=list(indices)
    packed['optimizers']=[]
    for optimizer in data['optimizers']:
        result={'param_groups':copy.deepcopy(optimizer['param_groups']),'state':{}}
        param_ids=[i for group in optimizer['param_groups'] for i in group['params']]
        id_names=dict(zip(param_ids,names)) # Additional IDs are the fully trained critic.
        for pid,entry in optimizer['state'].items():
            values={}
            for key,value in entry.items():
                if isinstance(value,torch.Tensor):
                    if pid in id_names and value.ndim>0:
                        index=indices[id_names[pid]].to(value.device)
                        values[key]={'sparse':True,'shape':tuple(value.shape),
                                     'values':value.detach().flatten()[index].cpu().clone()}
                    else:values[key]=value.detach().cpu().clone()
                else:values[key]=value
            result['state'][pid]=values
        packed['optimizers'].append(result)
    if packed.get('critics'):
        packed['critics']=[{n:v.detach().cpu().clone() for n,v in c.items()} for c in packed['critics']]
    return packed


def decode(packed, initial):
    checkpoint_format=packed.get('checkpoint_format')
    if checkpoint_format not in (FORMAT,FULL_FORMAT):return packed
    if packed.get('initial_policy_sha256'):
        digest=hashlib.sha256()
        for model in initial:
            for name,value in model.items():digest.update(name.encode());digest.update(value.numpy().tobytes())
        if digest.hexdigest()!=packed['initial_policy_sha256']:raise ValueError('Initial checkpoint hash differs')
    if checkpoint_format==FULL_FORMAT:
        result={k:v for k,v in packed.items() if k!='model_updates'}
        result['models']=[{n:v.clone() for n,v in weights.items()} for weights in initial]
        for i,weights in packed['model_updates'].items():result['models'][i]=weights
        return result
    result={k:v for k,v in packed.items() if k not in ('model_deltas','active_indices')}
    indices=packed['active_indices'];names=list(indices)
    result['models']=[]
    if len(initial)!=len(packed['model_deltas']):raise ValueError('Initial pool size differs')
    for base,delta in zip(initial,packed['model_deltas']):
        model={n:v.clone() for n,v in base.items()}
        for n,values in delta.items():model[n].view(-1)[indices[n]]=values
        result['models'].append(model)
    result['optimizers']=[]
    for optimizer in packed['optimizers']:
        result_opt={'param_groups':optimizer['param_groups'],'state':{}}
        param_ids=[i for group in optimizer['param_groups'] for i in group['params']]
        id_names=dict(zip(param_ids,names))
        for pid,entry in optimizer['state'].items():
            values={}
            for key,value in entry.items():
                if isinstance(value,dict) and value.get('sparse') is True:
                    tensor=torch.zeros(value['shape'],dtype=value['values'].dtype)
                    tensor.view(-1)[indices[id_names[pid]]]=value['values'];values[key]=tensor
                else:values[key]=value
            result_opt['state'][pid]=values
        result['optimizers'].append(result_opt)
    return result


def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
    return 0


def save(path, data, masks, reserve_bytes=128*1024**2):
    path=Path(path);packed=encode(data,masks)
    # The bound allows the complete uncompressed payload plus metadata, in
    # addition to a reserve for progress/log writes and other applications.
    required=tensor_bytes(packed)+8*1024**2+reserve_bytes
    free=shutil.disk_usage(path.parent).free
    if free<required:raise OSError(28,f'Checkpoint needs {required} free bytes; available {free}')
    temporary=path.with_name(path.name+'.tmp')
    try:
        with gzip.open(temporary,'wb',compresslevel=1) as stream:torch.save(packed,stream)
        replace_with_retry(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, initial):
    with gzip.open(path,'rb') as stream:packed=torch.load(stream,map_location='cpu',weights_only=False)
    return decode(packed,initial)
