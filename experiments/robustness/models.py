import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from causal_meta.modules.categorical import Marginal, Conditional
from causal_meta.models.binary import BinaryStructuralModel, ModelA2B, ModelB2A

class MixtureMarginalAnalytic(nn.Module):
    def __init__(self, N, dtype=None):
        super(MixtureMarginal, self).__init__()
        self.N = N
        self.dtype = dtype
        self.theta = nn.Parameter(torch.zeros(N, dtype=dtype))
        self.phi = nn.Parameter(torch.zeros((N, N), dtype=dtype))

    def forward(self, inputs):
        theta = F.softmax(self.theta, dim=0)
        phi = F.softmax(self.phi, dim=0)
        psi = torch.matmul(phi, theta)
        ret = torch.log(psi[inputs.squeeze(1)])
        return ret

    def set_ground_truth(self, pi_A_th):
        self.theta.data = torch.log(pi_A_th)
        self.phi.data = torch.log(torch.eye(self.N, dtype=self.dtype, requires_grad=True))

class MixtureMarginal(nn.Module):
    def __init__(self, N, dtype=None):
        super(MixtureMarginal, self).__init__()
        self.N = N
        self.K = 200
        self.dtype = dtype
        self.theta = nn.Parameter(torch.zeros(self.K, dtype=dtype))
        self.phi = nn.Parameter(torch.zeros((N, self.K), dtype=dtype))

    def forward(self, inputs):
        theta = F.softmax(self.theta, dim=0)
        phi = F.softmax(self.phi, dim=0)
        psi = torch.matmul(phi, theta)
        ret = torch.log(psi[inputs.squeeze(1)])
        return ret

    def set_ground_truth(self, pi_A_th):
        self.theta.data = torch.rand(self.theta.size(), dtype=self.dtype)
        self.phi.data = torch.rand(self.phi.size(), dtype=self.dtype)

class Model(object):
    def __init__(self, N):
        self.N = N

    def set_maximum_likelihood(self, inputs):
        inputs_A, inputs_B = np.split(inputs.numpy(), 2, axis=1)
        num_samples = inputs_A.shape[0]
        pi_A = np.zeros((self.N,), dtype=np.float64)
        pi_B_A = np.zeros((self.N, self.N), dtype=np.float64)
        
        # Empirical counts for p(A)
        for i in range(num_samples):
            pi_A[inputs_A[i, 0]] += 1
        pi_A /= float(num_samples)
        assert np.isclose(np.sum(pi_A, axis=0), 1.)
        
        # Empirical counts for p(B | A)
        for i in range(num_samples):
            pi_B_A[inputs_A[i, 0], inputs_B[i, 0]] += 1
        pi_B_A /= np.maximum(np.sum(pi_B_A, axis=1, keepdims=True), 1.)
        sum_pi_B_A = np.sum(pi_B_A, axis=1)
        assert np.allclose(sum_pi_B_A[sum_pi_B_A > 0], 1.)

        return self.set_ground_truth(pi_A, pi_B_A)

class Model1(Model, ModelA2B):
    def __init__(self, N, dtype=None):
        Model.__init__(self, N)
        ModelA2B.__init__(self, MixtureMarginal(N, dtype=dtype), Conditional(N, dtype=dtype))

    def set_ground_truth(self, pi_A, pi_B_A):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)

        self.p_A.set_ground_truth(pi_A_th)
        self.p_B_A.w.data = torch.log(pi_B_A_th)

class Model2(Model, ModelB2A):
    def __init__(self, N, dtype=None):
        Model.__init__(self, N)
        ModelB2A.__init__(self, Marginal(N, dtype=dtype), Conditional(N, dtype=dtype))

    def set_ground_truth(self, pi_A, pi_B_A):
        pi_A_th = torch.from_numpy(pi_A)
        pi_B_A_th = torch.from_numpy(pi_B_A)
        
        log_joint = torch.log(pi_A_th.unsqueeze(1)) + torch.log(pi_B_A_th)
        log_p_B = torch.logsumexp(log_joint, dim=0)
        
        self.p_B.w.data = log_p_B
        self.p_A_B.w.data = log_joint.t() - log_p_B.unsqueeze(1)

class StructuralModel(BinaryStructuralModel):
    def __init__(self, N, dtype=None):
        model_A_B = Model1(N, dtype=dtype)
        model_B_A = Model2(N, dtype=dtype)
        super(StructuralModel, self).__init__(model_A_B, model_B_A)
        self.w = nn.Parameter(torch.tensor(0., dtype=dtype))
    
    def set_ground_truth(self, pi_A, pi_B_A):
        self.model_A_B.set_ground_truth(pi_A, pi_B_A)
        self.model_B_A.set_ground_truth(pi_A, pi_B_A)
    
    def set_maximum_likelihood(self, inputs):
        self.model_A_B.set_maximum_likelihood(inputs)
        self.model_B_A.set_maximum_likelihood(inputs)
