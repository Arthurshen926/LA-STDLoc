"""Read selected tensors from trusted local torch ZIP caches without full loading.

This is a figure-only reader for the audited tensor-cache format, not a generic
checkpoint loader. As with torch.load(weights_only=False), inputs must be trusted.
"""
import io,pickle,zipfile
import numpy as np

DTYPES={'FloatStorage':'<f4','DoubleStorage':'<f8','HalfStorage':'<f2',
        'LongStorage':'<i8','IntStorage':'<i4','ShortStorage':'<i2',
        'CharStorage':'i1','ByteStorage':'u1','BoolStorage':'?'}

class TensorSlice:
    def __init__(self,storage,offset,size,stride,*args):
        self.storage=storage;self.offset=int(offset);self.size=tuple(size);self.stride=tuple(stride)
    def numpy(self):
        archive,prefix,key,dtype,count=self.storage
        raw=archive.read(prefix+'data/'+key)
        flat=np.frombuffer(raw,dtype=dtype,count=count)
        # Cache tensors are read-only; preserve strided view semantics.
        if self.size:
            assert self.offset+sum((n-1)*s for n,s in zip(self.size,self.stride))<len(flat)
        view=np.lib.stride_tricks.as_strided(flat[self.offset:],shape=self.size,
              strides=tuple(s*flat.itemsize for s in self.stride),writeable=False)
        return view.copy()
    def __iter__(self):return iter(self.numpy())
    def __array__(self,dtype=None,copy=None):return np.asarray(self.numpy(),dtype=dtype)

class CacheUnpickler(pickle.Unpickler):
    def __init__(self,stream,archive,prefix):super().__init__(stream);self.archive=archive;self.prefix=prefix
    def find_class(self,module,name):
        if module=='torch' and name in DTYPES:return np.dtype(DTYPES[name])
        if module=='torch._utils' and name in ('_rebuild_tensor','_rebuild_tensor_v2'):return TensorSlice
        return super().find_class(module,name)
    def persistent_load(self,value):
        kind,dtype,key,location,count=value
        if kind!='storage' or not isinstance(dtype,np.dtype):raise ValueError('Unsupported storage record')
        return self.archive,self.prefix,str(key),dtype,int(count)

def read_cache(path):
    archive=zipfile.ZipFile(path)
    entries=[n for n in archive.namelist() if n.endswith('/data.pkl')]
    if len(entries)!=1:raise ValueError('Expected one torch data.pkl')
    entry=entries[0];prefix=entry[:-len('data.pkl')]
    if prefix+'byteorder' in archive.namelist() and archive.read(prefix+'byteorder')!=b'little':
        raise ValueError('Only little-endian archives are supported')
    return CacheUnpickler(io.BytesIO(archive.read(entry)),archive,prefix).load()
