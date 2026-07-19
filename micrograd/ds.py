import math
import numpy as np

class Value():
    def __init__(self, data, _children = (), _op = '', label = ''): 
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = self._backward

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward(self, other):
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0
            self._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward(self, other):
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
            self._backward = _backward
        return out
    
    