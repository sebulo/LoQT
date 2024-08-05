from bitsandbytes.nn import LinearNF4, Params4bit
import torch
import bitsandbytes as bnb
from typing import Callable, Optional, Tuple
from warnings import warn
import operator
from functools import reduce
import bitsandbytes.functional as F
import os


class LinearNF4WithGradient(LinearNF4):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_grad = torch.tensor(0)
        self.require_grad_W = False
        self.grad_offloading = False
        
    def forward(self, x: torch.Tensor):
                # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if getattr(self.weight, "quant_state", None) is None:
            if getattr(self, "quant_state", None) is not None:
                # the quant state got lost when the parameter got converted. This happens for example for fsdp
                # since we registered the module, we can recover the state here
                assert self.weight.shape[1] == 1
                if not isinstance(self.weight, Params4bit):
                    self.weight = Params4bit(self.weight, quant_storage=self.quant_storage)
                self.weight.quant_state = self.quant_state
            else:
                print(
                    "FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.",
                )
        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)
        if self.grad_offloading:
            self.weight_grad = self.weight_grad.to('cpu')
        #W is not transposed here as it is done in the functional implementation to avoid messing up code backprop that works
        out = matmul_4bit_withgradient(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state, weight_grad=self.weight_grad, require_grad_W=self.require_grad_W, grad_offloading = self.grad_offloading)
        out = out.to(inp_dtype)

        return out
    
def matmul_4bit_withgradient(A, B, quant_state, out=None, bias=None, weight_grad=None, require_grad_W=False, grad_offloading = False):
    assert quant_state is not None
    if A.numel() == A.shape[-1] and A.requires_grad == False:
        if A.shape[-1] % quant_state.blocksize != 0:
            warn(
                f"Some matrices hidden dimension is not a multiple of {quant_state.blocksize} and efficient inference kernels are not supported for these (slow). Matrix input size found: {A.shape}",
            )
            return MatMul4BitGradientWithGrad.apply(A, B, out, bias, quant_state, weight_grad, require_grad_W, grad_offloading)
        else:
            out = F.gemv_4bit(A, B.t(), out, state=quant_state)
            if bias is not None:
                out += bias
            return out
    else:
        return MatMul4BitGradientWithGrad.apply(A, B, out, bias, quant_state, weight_grad, require_grad_W, grad_offloading)


# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class MatMul4BitGradientWithGrad(torch.autograd.Function):
# forward is the same, but we added the fallback for pre-turing GPUs
# backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

    @staticmethod
    def forward(ctx, A, B, out=None, bias=None, quant_state: Optional[F.QuantState] = None, weight_grad=None, require_grad_W=False, Grad_offloading = False):
        # default of pytorch behavior if inputs are empty
        is_empty = False
        if prod(A.shape) == 0:
            is_empty = True
            B_shape = quant_state.shape
            ctx.save_for_backward(torch.tensor(is_empty), A, B, bias)

            if A.shape[-1] == B_shape[0]:
                return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
            else:
                return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

        # 1. Dequantize
        # 2. MatmulnN
        # local_rank = int(os.environ["LOCAL_RANK"])

        #print("Local_rank:", local_rank, "A device:", A.device, "B device:", B.device)
        
        quant_state.to(A.device)
    
        #print("Local_rank:", local_rank, "A device:", A.device, "B device:", B.device, "Dequabt device:", dequant.device)

        #A = A.to(local_rank)
        #B = B.to(local_rank)
        #bias = bias.to(local_rank) if bias is not None else None
        output = torch.nn.functional.linear(A, F.dequantize_4bit(B, quant_state).to(A.dtype).t(), bias)

        # 3. Save state
        ctx.quant_state = quant_state

        use_A, use_B = False, False
        if not require_grad_W:
            A = torch.tensor(False)
        if not ctx.needs_input_grad[0]:
            B = torch.tensor(False)

        if not torch.is_tensor(require_grad_W):
            require_grad_W = torch.tensor(require_grad_W)
        if not torch.is_tensor(Grad_offloading):
            Grad_offloading = torch.tensor(Grad_offloading)
        if not torch.is_tensor(is_empty):
            is_empty = torch.tensor(is_empty)

        ctx.save_for_backward(is_empty, A, B, bias, weight_grad, require_grad_W, Grad_offloading)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        is_empty = ctx.saved_tensors[0]
        if is_empty:
            is_empty, A, B, bias = ctx.saved_tensors
            bias_grad = None if bias is None else torch.zeros_like(bias)
            return torch.zeros_like(A), torch.zeros_like(B), None, bias_grad, None

        quant_state = ctx.quant_state
        is_empty, A, B, bias, weight_grad, require_grad_W, Grad_offloading = ctx.saved_tensors
        dtype_A, dtype_B = A.dtype, B.dtype
        if bias is not None:
            dtype_bias = bias.dtype

        req_gradA, _, _, req_gradBias, _, req_grad_dummyvar, req_grad_requires_grad_bool, _ = ctx.needs_input_grad
        grad_A, grad_B, grad_bias = None, None, None

        if req_gradBias:
            # compute grad_bias first before changing grad_output dtype
            grad_bias = grad_output.sum(0, dtype=dtype_bias)

        #TODO accumulate gradients
        #We cannot set require_grad on thr weight_grad, since it is int tensor
        #Instead we supply a bool in the forward pass and a dummy variable that we can write the gradient to
        if require_grad_W:
            print(f"grad_output shape: {grad_output.shape}, dim: {grad_output.dim()}")
            #check if weight_grad has correct shape, otherwise create it
            if grad_output.dim() == 3:
                weight_grad.data += torch.einsum('bji,bjk->ki', grad_output, A).t().to(weight_grad.device)
            else:
                weight_grad.data += torch.matmul(grad_output.t(), A).t().to(weight_grad.device)
            if torch.all(weight_grad == 0):
                print("weight_grad is zero in backward")
        if req_gradA:
            grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, quant_state).to(grad_output.dtype).t())
        
        # cleanup
        ctx.quant_state = None

        return grad_A, None, None, grad_bias, None, None, None, None
