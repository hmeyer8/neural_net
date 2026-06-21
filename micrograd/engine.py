import numpy as np
import math
import matplotlib.pyplot as plt

class Value():
    def __init__(self, data, _children = (), _op = '', label = '')
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label 
        self._backward = lambda: None
        self.grad = 0 
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(self, other):
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
            out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward(self,other)
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
            out._backward = _backward

        return out
    
    def __pow__(self, other)
        assert isinstance(other, (int,float))
        out = Value(self.data ** other.data, (self, other), f'**{other}')

        def _backward(self, other):
            self.grad += other*self.data**(other-1) * out.grad
            self._backward = _backward

        return out

    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'RELU')
        def _backward(self):
            self.grad += (out.data>0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(child)

        self._grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"