"""Read contiguous stored rows before restoring the exact requested unit order."""
import numpy as np
class SlabAccess:
    def __init__(self,array):
        self.array=array
    @property
    def shape(self):
        return self.array.shape
    def __getitem__(self,key):
        split,half,units,columns=key
        units=np.asarray(units)
        lo=int(units.min());hi=int(units.max())+1
        # Copy traverses a single forward contiguous byte range, allowing file
        # readahead. Advanced indexing happens on RAM, never the remote mmap.
        slab=np.array(self.array[split,half,lo:hi,columns],copy=True)
        return slab[units-lo]
