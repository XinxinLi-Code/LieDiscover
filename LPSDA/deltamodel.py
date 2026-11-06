# modified from https://github.com/kogyeonghoon/learning-symmetry-from-scratch/blob/main/pde/deltamodel.py

import torch

class VectorFieldOperation():
    def __init__(self,):
        # compute linear operations of vector field defined on xtu space
        # by default, vector fields have size (n_xtu,3) (singular) or (n_xtu,2,n_delta) (multiple)
        # (n_delta = number of vector fields)
        
        pass

    def inner_product(self, delta1, delta2):
        # computes inner products of two (stacks of) vector fields
        if (len(delta1.shape) == 2) & (len(delta2.shape) == 2):
            return (torch.einsum('ij,ij', delta1, delta2)) / delta1.shape[0]
        
        elif (len(delta1.shape) == 3) & (len(delta2.shape) == 2):
            return (torch.einsum('ija,ij->a', delta1, delta2)) / delta1.shape[0]
        
        elif (len(delta1.shape) == 2) & (len(delta2.shape) == 3):
            return (torch.einsum('ij,ijb->b', delta1, delta2)) / delta1.shape[0]
        
        elif (len(delta1.shape) == 3) & (len(delta2.shape) == 3):
            return (torch.einsum('ija,ijb->ab', delta1, delta2)) / delta1.shape[0]
        
        else:
            assert False
        
                    
    def normalize_delta(self, delta, eps=1e-8):
        # normalizes vector fields
        
        if len(delta.shape) == 2:
            delta_norm = self.inner_product(delta,delta) ** 0.5
            delta_norm = torch.clamp(delta_norm, min=eps)
            
            return delta / delta_norm
        
        elif len(delta.shape) == 3:
            delta_norm = self.inner_product(delta, delta).diag() ** 0.5
            delta_norm = torch.clamp(delta_norm, min=eps)
            
            return delta / delta_norm.view(1, 1, -1)
        else:
            assert False
    

