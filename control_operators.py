from typing import Dict, Any, Tuple, List
import numpy as np
import torch

# Control Operator

class ControlOperator(torch.nn.Module): pass

# Typed Operators

class Vector(ControlOperator):
    
    def __init__(self, size : int) -> None:
        super().__init__()
        self.size = size
    
    def output_size(self) -> int:
        return self.size
        
    def forward(self, x : List[torch.FloatTensor]) -> torch.FloatTensor:
        assert all(len(xb.shape) == 1 for xb in x)
        assert all(xb.shape[0] == self.output_size() for xb in x)
        return torch.stack(x, dim=0)
        
        
class Location(Vector):
    def __init__(self) -> None: super().__init__(3)

class Direction(Vector):
    def __init__(self) -> None: super().__init__(3)

class Velocity(Vector):
    def __init__(self) -> None: super().__init__(3)


def quat_to_xform_xy(q):
    qw, qx, qy, qz = q[...,0:1], q[...,1:2], q[...,2:3], q[...,3:4]
    
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, wx = qx * x2, qy * y2, qw * x2
    xy, yz, wy = qx * y2, qy * z2, qw * y2
    xz, zz, wz = qx * z2, qz * z2, qw * z2
    
    return torch.cat([
        torch.cat([1.0 - (yy + zz), xy - wz], dim=-1)[...,None,:],
        torch.cat([xy + wz, 1.0 - (xx + zz)], dim=-1)[...,None,:],
        torch.cat([xz - wy, yz + wx], dim=-1)[...,None,:],
    ], dim=-2)

class Rotation(ControlOperator):
    
    def output_size(self) -> int:
        return 6
        
    def forward(self, x : List[torch.FloatTensor]) -> torch.FloatTensor:
        assert all(len(xb.shape) == 1 for xb in x)
        assert all(xb.shape[0] == 4 for xb in x)
        return quat_to_xform_xy(torch.stack(x, dim=0)).reshape([len(x), 6])

# And / Struct

class And(ControlOperator):
    
    def __init__(self, ops : Dict[str,ControlOperator]) -> None:
        super().__init__()
        self.ops = torch.nn.ModuleDict(ops)
        
    def output_size(self) -> int:
        return sum([v.output_size() for v in self.ops.values()])
        
    def forward(self, x : List[Dict[str,Any]]) -> torch.FloatTensor:
        assert all(all(k in xb for k in self.ops) for xb in x)
        return torch.cat([v([xb[k] for xb in x]) for k, v, in self.ops.items()], dim=-1)
        
class Struct(And): pass

# Or / Union

class Or(ControlOperator):
    
    def __init__(self, ops : Dict[str,ControlOperator], encoding_size=256) -> None:
        super().__init__()
        
        self.ops = torch.nn.ModuleDict(ops)
        self.Ws = torch.nn.ModuleDict({k: 
            torch.nn.Linear(v.output_size(), encoding_size) 
            for k, v in ops.items()})
        self.encoding_size = encoding_size
        
    def output_size(self) -> int:
        return self.encoding_size + len(self.ops)
        
    def forward(self, x : List[Tuple[str,Any]]) -> torch.FloatTensor:
        assert(all(xb[0] in self.ops for xb in x))
        
        # Create zero output
        out = torch.zeros([len(x), self.output_size()], dtype=torch.float32)
        
        # Loop over sub-operators
        for ki, (k, v) in enumerate(self.ops.items()):
            
            # Find batch indices for this sub operator
            indices = torch.as_tensor([xi for xi, xb in enumerate(x) if xb[0] == k])
            
            if len(indices) > 0:
                # Insert encoded
                out[indices,:-len(self.ops)] = self.Ws[k](v([xb[1] for xb in x if xb[0] == k]))
                
                # Insert one-hot
                out[indices,-len(self.ops) + ki] = 1.0
        
        return out
        
class Union(Or): pass

# Set

class Set(ControlOperator):
    
    def __init__(self, op : ControlOperator, head_num=8, query_size=256, encoding_size=256) -> None:
        super().__init__()
        self.op = op
        self.head_num = head_num
        self.query_size = query_size
        self.encoding_size = encoding_size
        self.Q = torch.nn.Linear(op.output_size(), head_num * query_size)
        self.K = torch.nn.Linear(op.output_size(), head_num * query_size)
        self.V = torch.nn.Linear(op.output_size(), head_num * encoding_size)

    def output_size(self) -> int:
        return self.head_num * self.encoding_size
        
    def forward(self, x : List[List[Any]]) -> torch.FloatTensor:
        
        encoded = self.op(sum(x, []))        
        queries = self.Q(encoded).reshape([len(encoded), self.head_num, self.query_size])
        keys = self.K(encoded).reshape([len(encoded), self.head_num, self.query_size])
        values = self.V(encoded).reshape([len(encoded), self.head_num, self.encoding_size])
        
        output = torch.zeros([len(x), self.head_num, self.encoding_size], dtype=torch.float32)
        
        offset = 0
        for xi, xb in enumerate(x):
            attn = torch.softmax((queries * keys)[offset:offset+len(xb)].sum(dim=-1) / np.sqrt(self.query_size), dim=0)
            output[xi] = (attn[...,None] * values[offset:offset+len(xb)]).sum(axis=0)
            offset += len(xb)
        
        return output.reshape([len(x), self.head_num * self.encoding_size])

# Fixed Array

class FixedArray(ControlOperator):

    def __init__(self, op : ControlOperator, num : int) -> None:
        super().__init__()
        self.op = op
        self.num = num
    
    def output_size(self) -> int:
        return self.op.output_size() * self.num
    
    def forward(self, x : List[List[Any]]) -> torch.FloatTensor:
        assert all(len(xb) == self.num for xb in x)
        return self.op(sum(x, [])).reshape([len(x), -1])

# Null

class Null(ControlOperator):
    
    def output_size(self) -> int:
        return 0
        
    def forward(self, x : List[None]) -> torch.FloatTensor:
        return torch.empty([len(x), 0], dtype=torch.float32)

# Index

class Index(ControlOperator):

    def __init__(self, max_index : int = 128) -> None:
        super().__init__()
        self.max_index = max_index
        
    def output_size(self) -> int:
        return 1

    def forward(self, x : List[int]) -> torch.FloatTensor:
        return torch.as_tensor(x, dtype=torch.float32)[...,None] / self.max_index
    
# One Of

class OneOf(ControlOperator):
    
    def __init__(self, choices : List[str]) -> None:
        super().__init__()
        self.choices = choices
        
    def output_size(self) -> int:
        return len(self.choices)
        
    def forward(self, x : List[str]) -> torch.FloatTensor:
        return torch.nn.functional.one_hot(
            torch.as_tensor([self.choices.index(xb) for xb in x]), len(self.choices))
    
class Enum(OneOf): pass

# Some Of

class SomeOf(ControlOperator):
    
    def __init__(self, choices : List[str]) -> None:
        super().__init__()
        self.choices = choices
        
    def output_size(self) -> int:
        return len(self.choices)
        
    def forward(self, x : List[List[str]]) -> torch.FloatTensor:
        out = torch.zeros([len(x), len(self.choices)])
        for xi, xb in enumerate(x):
            for c in xb:
                out[xi,self.choices.index(c)] = 1.0
        return out

class Flags(SomeOf): pass
    
# String

import clip

clip_model, _ = clip.load("ViT-B/32")

class String(ControlOperator):
    
    def output_size(self) -> int:
        return clip_model.token_embedding.weight.shape[1]

    def forward(self, x : List[str]) -> torch.FloatTensor:
        return clip_model.encode_text(clip.tokenize(x))

# Optional

class Optional(ControlOperator):

    def __init__(self, op : ControlOperator, **kw) -> None:
        super().__init__()
        self.op = Or({ 'null': Null(), 'valid': op }, **kw)
        
    def output_size(self) -> int:
        return self.op.output_size()

    def forward(self, x : List[Any]) -> torch.FloatTensor:
        return self.op([('null', None) if xb is None else ('valid', xb) for xb in x])
        

class Maybe(Optional): pass

# Array

class Array(ControlOperator):

    def __init__(self, op : ControlOperator, **kw) -> None:
        super().__init__()
        self.op = Set(And({'index': Index(), 'value': op}), **kw)
    
    def output_size(self) -> int:
        return self.op.output_size()
    
    def forward(self, x : List[List[Any]]) -> torch.FloatTensor:
        return self.op([[{'index': xbi, 'value': xbv} for xbi, xbv in enumerate(xb)] for xb in x])

# Dictionary

class Dictionary(ControlOperator):

    def __init__(self, key : ControlOperator, value : ControlOperator, **kw) -> None:
        super().__init__()
        self.op = Set(And({'key': key, 'value': value}), **kw)
    
    def output_size(self) -> int:
        return self.op.output_size()
    
    def forward(self, x : List[Dict[Any,Any]]) -> torch.FloatTensor:
        return self.op([[{'key': xbk, 'value': xbv} for xbk, xbv in xb.items()] for xb in x])


# Encoded

class Encoded(ControlOperator):

    def __init__(self, op : ControlOperator, encoding_size = 256, activation = torch.nn.functional.elu) -> None:
        super().__init__()
        self.op = op
        self.W = torch.nn.Linear(op.output_size(), encoding_size)
        self.encoding_size = encoding_size
        self.activation = activation
    
    def output_size(self) -> int:
        return self.encoding_size
    
    def forward(self, x : List[Any]) -> torch.FloatTensor:
        return self.activation(self.W(self.op(x)))

