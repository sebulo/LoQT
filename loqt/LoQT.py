import torch
import torch.nn as nn
import os
import json
import bitsandbytes as bnb
import bitsandbytes.functional as bnb_F
from dataclasses import dataclass
from typing import List
import torch.distributed as dist
from loqt.utils import create_zero_initialized_linear_layer, eigenH_decomposition
from loqt.bnb_with_gradient import LinearNF4WithGradient
import copy

@dataclass
class LoQT_Config:
    target_modules: List[str]
    r: int
    lora_alpha: int
    lora_dropout: float
    trainable_scaling: bool = False
    quantize_w: str = None
    use_double_quant: bool = False
    proj_type: str = 'std'
    quantize_projection_matrix: str = None
    compensate_quant_error_iterations: int = 0
    is_single_gpu: bool = False
    only_train_lora: bool = False
    use_offloading: bool = False
    use_eigenh_for_projection: bool = False
    init_lora_AB_as_random_and_zeros: bool = False
    train_projection_matrix: bool = False

class LoQTModel(nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        trainable_scaling=False,
        quantize_w=None,
        use_double_quant=False,
        proj_type='std',
        device,
        compute_dtype=torch.bfloat16,
        quantize_projection_matrix=None,
        compensate_quant_error_iterations=0,
        use_offloading=False,
        is_single_gpu=False,
        only_train_lora=False,
        model_config=None,
        use_eigenh_for_projection=False,
        init_lora_AB_as_random_and_zeros=False,
        train_projection_matrix=False
    ):
        if r <= 0:
            raise ValueError("r must be positive. If you want r == 0, use the original model.")


        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.trainable_scaling = trainable_scaling
        self.device = device
        self.forward = self.wrapped_model.forward
        self.proj_type = proj_type
        self.quantize_w = quantize_w
        self.quantize_projection_matrix = quantize_projection_matrix
        self.use_offloading = use_offloading
        self.is_single_gpu = is_single_gpu
        self.only_train_lora = only_train_lora
        self.model_config = model_config
        self.use_eigenh_for_projection = use_eigenh_for_projection
        self.init_lora_AB_as_random_and_zeros = init_lora_AB_as_random_and_zeros
        self.train_projection_matrix = train_projection_matrix

        # Initialize the configuration with the given parameters
        self._config = LoQT_Config(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            trainable_scaling=trainable_scaling,
            quantize_w=quantize_w,
            use_double_quant=use_double_quant,
            proj_type=proj_type,
            quantize_projection_matrix = self.quantize_projection_matrix,
            compensate_quant_error_iterations = compensate_quant_error_iterations,
            use_offloading = use_offloading,
            is_single_gpu = is_single_gpu,
            only_train_lora = only_train_lora,
            use_eigenh_for_projection=use_eigenh_for_projection,
            init_lora_AB_as_random_and_zeros=init_lora_AB_as_random_and_zeros,
            train_projection_matrix=train_projection_matrix
        )

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            new_module = LoraLinear(
                module,
                r=self.r,
                lora_alpha = lora_alpha,
                proj_type = self.proj_type,
                device = self.device,
                quantize_w=quantize_w,
                use_double_quant=use_double_quant,
                bnb_4bit_quant="nf4",
                compute_dtype=compute_dtype,
                quantize_projection_matrix = quantize_projection_matrix,
                compensate_quant_error_iterations = compensate_quant_error_iterations,
                use_offloading = self.use_offloading,
                is_single_gpu=is_single_gpu,
                use_eigenh_for_projection=use_eigenh_for_projection,
                init_lora_AB_as_random_and_zeros=init_lora_AB_as_random_and_zeros,
                train_projection_matrix=train_projection_matrix,
            )

            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent
    
    def set_W_requires_grad(self, requires_grad: bool):
        if self.only_train_lora:
            for module in self.modules():
                for param in module.parameters():
                    param.requires_grad = False
        
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.set_W_requires_grad(requires_grad)
    
    def merge(self):
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.merge()
        torch.cuda.synchronize()
    
    def disable_lora(self, flag):
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.lora_params_disabled = flag
    
    def set_LoRA_requires_grad(self, requires_grad=True):
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.set_LoRA_requires_grad(requires_grad)
    
    def lora_zero_init(self):
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.lora_zero_init()

    def reinitialize_LoRA_AB_after_merge(self):
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.reinitialize_LoRA_AB_after_merge()
                
    def quantize_all_lora_linear_layers(self):
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.quantize_LoRA_AB()
    
    def dequantize_all_lora_linear_layers(self):
        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.maybe_dequantize_LoRA_factors()

                
    def __repr__(self):
        repr_str = super().__repr__()

        # Iterate over the model's named parameters and add them to the representation string
        for name, param in self.named_parameters():
            if isinstance(param, nn.Parameter):
                repr_str += f"\n  ({name}): {param.size()}"
        return repr_str
    
    def save_pretrained(self, path, save_original_model=False):
        # Ensure all parameters are contiguous
        make_tensors_contiguous(self.wrapped_model)
        os.makedirs(path, exist_ok=True)
        if save_original_model:
            model_to_save = self.return_original_model()
            torch.save(model_to_save, os.path.join(path, "original_model.pth"))
        torch.save(self, os.path.join(path, "pytorch_model_full.pth"))
        # Save additional configuration
        with open(os.path.join(path, "loqt_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)
            
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.model_config, f, indent=4)

    @classmethod
    def from_pretrained(cls, path, device, saved_as_full_model=False):
        if saved_as_full_model:
            model2 = torch.load(os.path.join(path, "original_model.pth"), map_location=device)
        else:
            model2 = torch.load(os.path.join(path, "pytorch_model_full.pth"), map_location=device)
        return model2
    
    def return_original_model(self):
        # Create a deep copy of the wrapped model on CPU to avoid modifying the original model
        new_model = copy.deepcopy(self.wrapped_model)

        # Loop over new and old modules, dequantize old LoRA factors and merge them into the new weight matrix
        for module_new, module_old in zip(new_model.modules(), self.wrapped_model.modules()):
            if isinstance(module_old, LoraLinear):
                module_old.maybe_dequantize_LoRA_factors()
                # Merge the LoRA factors into the main weight matrix
                AB = module_old.scaling * (module_old.lora_A.weight.T @ module_old.lora_B.weight.T).T.detach()
                if self.quantize_w:
                    W_deq = bnb_F.dequantize_4bit(module_old.W.weight, module_old.W.weight.quant_state, quant_type=module_old.bnb_4bit_quant_type)
                    W_deq = W_deq.to(dtype=module_old.compute_dtype)
                else:
                    W_deq = module_old.W.weight
                # Update the weight data in the new model
                module_new.W.weight.data = W_deq + AB
                module_old.quantize_LoRA_AB()

        # Replace LoraLinear modules with standard Linear modules
        def replace_lora_linear(module):
            for name, child in module.named_children():
                if isinstance(child, LoraLinear):
                    # Create a standard Linear module with the same dimensions and copy the weights
                    new_linear = nn.Linear(child.in_features, child.out_features, bias=child.W.bias is not None).to(self.device)
                    new_linear.weight.data = child.W.weight.data.clone()
                    if child.W.bias is not None:
                        new_linear.bias.data = child.W.bias.data.clone()
                    # Replace the child module with the new Linear module
                    setattr(module, name, new_linear)
                else:
                    # Recursively replace LoraLinear modules in the child modules
                    replace_lora_linear(child)

        replace_lora_linear(new_model)

        return new_model
    

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get("device", None)
        self.device = device

        for module in self.modules():
            if isinstance(module, LoraLinear):
                module.to(device)
        
        return self

    @classmethod
    def model_memory_usage_in_MB(cls, model, include_quant_state=True):
        """
        Calculate the total memory usage of the model's parameters in megabytes (MB),
        including optional quantization state if specified.

        Args:
            model (LoQTModel): The model instance for which memory usage is calculated.
            include_quant_state (bool): If True, include the memory used by quantization states in the total.

        Returns:
            float: Total memory usage of the model in MB.
        """
        num_bfloat16_params = 0
        num_float32_params = 0
        num_int8_params = 0
        total_bytes = 0
        for name, param in model.named_parameters():
            # Basic parameter memory calculation
            if param.dtype == torch.float32:
                bytes_per_element = 4
                num_float32_params += param.numel()
            elif param.dtype == torch.bfloat16:
                bytes_per_element = 2
                num_bfloat16_params += param.numel()
            elif param.dtype in [torch.int8, torch.uint8]:
                bytes_per_element = 1
                num_int8_params += param.numel()
            else:
                print(f"Unknown parameter type: {param.dtype}, name: {name}")
                bytes_per_element = 4  # Default to 4 bytes

            # Add parameter memory to total
            param_memory = param.numel() * bytes_per_element
            total_bytes += param_memory

            # Check and add quantization state memory
            if include_quant_state and hasattr(param, 'quant_state'):
                quant_state = vars(param.quant_state)
                for attr, value in quant_state.items():
                    if isinstance(value, torch.Tensor):
                        state_bytes = value.numel() * (4 if value.dtype == torch.float32 else 1)
                        total_bytes += state_bytes

        # Convert total bytes to megabytes
        total_MB = total_bytes / (1024 ** 2)
        print(f"Float32 params: {num_float32_params}, BFloat16 params: {num_bfloat16_params}, Int8 params: {num_int8_params}")
        return total_MB


    
class LoraLinear(nn.Module):
    def __init__(
            self,
            W,
            r,
            lora_alpha,
            proj_type,
            device,
            quantize_w=None,
            use_double_quant=False,
            bnb_4bit_quant="nf4",
            compute_dtype=torch.bfloat16,
            quantize_projection_matrix=None,
            compensate_quant_error_iterations=0,
            use_offloading=False,
            is_single_gpu=False,
            use_eigenh_for_projection=False,
            init_lora_AB_as_random_and_zeros=False,
            train_projection_matrix=False
        ):
        super().__init__()
        assert isinstance(W, nn.Linear)
        
        self.compute_dtype = compute_dtype
        self.device = device
        self.scaling = lora_alpha
        self.quantize_w = quantize_w
        self.use_double_quant = use_double_quant
        self.bnb_4bit_quant_type = bnb_4bit_quant
        self.quantize_projection_matrix = quantize_projection_matrix
        self.in_features = W.in_features
        self.out_features = W.out_features
        self.use_offloading = use_offloading
        self.init_lora_AB_as_random_and_zeros = init_lora_AB_as_random_and_zeros
        self.train_projection_matrix = train_projection_matrix
        self.offload_device = 'cpu' if use_offloading else self.device
        self.W = self.maybe_quantize(W, quantize_w, use_double_quant, bnb_4bit_quant) 
        self.set_W_requires_grad(False)
        self.r = min(r, self.W.in_features)
        self.lora_params_disabled = False
        self.validate_inputs(W, proj_type)
        self.proj_type = self.determine_projection_type(W, proj_type)
        self.lora_A, self.lora_B = self.zero_initialize_LoRA_AB()
        self.set_LoRA_requires_grad(True)
        self.full_precision_W = None
        self.compensate_quant_error_iterations = compensate_quant_error_iterations
        self.is_single_gpu = is_single_gpu
        self.use_eigenh_for_projection = use_eigenh_for_projection
        
        
        # Determine the method and parameters for projection matrix computation
        self.projection_method = 'eigh' if self.use_eigenh_for_projection else 'svd'
        self.projection_out = 'u' if self.proj_type == 'left' else 'v'
        
    def validate_inputs(self, W, proj_type):
        if not isinstance(W, nn.Linear):
            raise ValueError("W must be an instance of nn.Linear")
        if proj_type not in ['std', 'left', 'right', ]:
            raise ValueError("proj_type must be 'std', 'left',or 'right'")
        # Quantize_projection_matrix and train_projection_matrix should not both be True
        if self.quantize_projection_matrix and self.train_projection_matrix:
            raise ValueError("quantize_projection_matrix and train_projection_matrix cannot both be True")
        if self.init_lora_AB_as_random_and_zeros and not self.train_projection_matrix:
            raise ValueError("init_lora_AB_as_random_and_zeros can only be True when train_projection_matrix is True")

    
    def determine_projection_type(self, W, proj_type):
        # Check if the number of input features is less than or equal to the number of output features
        if proj_type == 'std':
            if W.in_features <= W.out_features:
                # If true, use 'left' projection type
                return 'left'
            else:
                return 'right'
        else:
            return proj_type # If the projection type is not 'std', return it as is

    def maybe_quantize(self, W, quantize, use_double_quant, bnb_4bit_quant):
        use_bias = W.bias is not None

        if quantize is None:
            return W

        elif quantize=="4bit":
            linear_q = LinearNF4WithGradient(W.in_features, W.out_features, bias=use_bias, compress_statistics=use_double_quant)            
            W = W.to('cpu') # Without this, the weight is not quantized if it was already quantise before             
            new_weight = bnb.nn.Params4bit(data=W.weight, quant_type=bnb_4bit_quant, requires_grad=False)
            linear_q.weight = new_weight
            linear_q.grad_offloading = self.use_offloading
            linear_q.weight_grad = torch.tensor(0, device=self.offload_device, requires_grad=False)
            if use_bias:
                linear_q.bias = nn.Parameter(W.bias.data, requires_grad=True)
            return linear_q.to(self.device)
            
        else:
            raise ValueError("quantize must be None, or '4bit'")
                
    def set_LoRA_requires_grad(self, flag):
        if self.train_projection_matrix and flag:
            # set both A and B to requires_grad
            self.lora_A.weight.requires_grad = flag
            self.lora_B.weight.requires_grad = flag
            self.lora_params_disabled = False
        elif flag:
            # Set requires_grad for LoRA layers based on the projection type
            if self.proj_type == 'left': 
                self.lora_A.weight.requires_grad = False
                self.lora_B.weight.requires_grad = True  
                self.lora_params_disabled = False
            elif self.proj_type == 'right':
                self.lora_A.weight.requires_grad = True
                self.lora_B.weight.requires_grad = False
                self.lora_params_disabled = False
        else:
            self.lora_B.requires_grad = False
            self.lora_A.requires_grad = False
            self.lora_A.grad = None
            self.lora_B.grad = None        
            self.lora_params_disabled = True
            
    def lora_zero_init(self):
        self.lora_A.weight.data = torch.zeros((self.r, self.in_features), device=self.device, dtype=self.compute_dtype, requires_grad=True)
        self.lora_B.weight.data = torch.zeros((self.out_features, self.r), device=self.device, dtype=self.compute_dtype, requires_grad=True)
        self.quantize_LoRA_AB()
        
    def zero_initialize_LoRA_AB(self):
        lora_A, lora_B = self.initialize_LoRA_AB()
        self.lora_A_shape = lora_A.weight.shape # in, r
        self.lora_B_shape = lora_B.weight.shape # r, out
        if self.quantize_projection_matrix == '4bit':
            if self.proj_type == 'left':
                lora_A = self.maybe_quantize(lora_A, self.quantize_projection_matrix, self.use_double_quant, self.bnb_4bit_quant_type)
            else:
                lora_B = self.maybe_quantize(lora_B, self.quantize_projection_matrix, self.use_double_quant, self.bnb_4bit_quant_type)
        return lora_A, lora_B
    
    def initialize_LoRA_AB(self):
        lora_A = create_zero_initialized_linear_layer(self.in_features, self.r, False, self.device, dtype=self.compute_dtype)
        lora_B = create_zero_initialized_linear_layer(self.r, self.out_features, False, self.device, dtype=self.compute_dtype)
        return lora_A, lora_B
    
    def forward(self, X):
        if self.lora_params_disabled:
            return self.W(X)
        
        W_output = self.W(X) 
        
        lora_A_output = self.lora_A(X)
        lora_output = self.lora_B(lora_A_output)
        
        # return W_output.add_(self.scaling * lora_output)  # In-place addition
        return W_output + (self.scaling*lora_output)
    
    def optimize_quant_and_lora_jointly(self, P):
        proj_type = self.proj_type
        # Check if the original W layer has a bias
        use_bias = self.W.bias is not None
        
        # Create a new Linear layer with the same bias setting
        curr_W_linear = torch.nn.Linear(self.in_features, self.out_features, bias=use_bias, device=self.device, dtype=self.compute_dtype)
        
        curr_W = self.full_precision_W.data.clone() #out x in
        curr_W.requires_grad = False
        curr_W_linear.weight.requires_grad = False
        curr_W_linear.weight.data = curr_W #.clone() #in x out
        
        # If the original W layer has a bias, copy it to the new layer
        if use_bias:
            curr_W_linear.bias.data.copy_(self.W.bias.data)
            curr_W_linear.bias.requires_grad = False
        
        best_combination_norm = float('inf')
        best_B = None
        best_Q_W = None
        
        if self.compensate_quant_error_iterations == 0 or self.quantize_w is None:
            if proj_type == 'left':
                Q_W = self.maybe_quantize(curr_W_linear, self.quantize_w, self.use_double_quant, self.bnb_4bit_quant_type)
                B = torch.zeros((self.out_features, self.r), device=self.device, dtype=self.compute_dtype)
            elif proj_type == 'right':
                Q_W = self.maybe_quantize(curr_W_linear, self.quantize_w, self.use_double_quant, self.bnb_4bit_quant_type)
                B = torch.zeros((self.r, self.in_features), device=self.device, dtype=self.compute_dtype)
            else:
                raise ValueError("proj_type must be 'left' or 'right'")
            del curr_W
            return Q_W, B
        
        # calculate P_inv only once
        P = P.detach().to(self.compute_dtype)
        if self.projection_method == 'svd':
            P_inv = torch.pinverse(P.float()).to(self.compute_dtype) # requires float 
        else:
            P_inv = P.T
        
        for _ in range(self.compensate_quant_error_iterations):
            Q_W = self.maybe_quantize(curr_W_linear, self.quantize_w, self.use_double_quant, self.bnb_4bit_quant_type)
            deq_W = bnb_F.dequantize_4bit(Q_W.weight, Q_W.weight.quant_state, quant_type=self.bnb_4bit_quant_type) #out x in
            residual = (self.full_precision_W-deq_W).to(self.compute_dtype) 
            if proj_type == 'left':
                B = ((P_inv.T @ residual.t()).T)/self.scaling # out  x in
                curr_W = (self.full_precision_W - self.scaling*(B@P))
                curr_norm = torch.norm(self.full_precision_W-(deq_W+(self.scaling*(B@P))), 'fro')
            elif proj_type == 'right':
                B = ((residual.T @ P_inv.T).T)/self.scaling
                curr_W = (self.full_precision_W - self.scaling*(P@B))
                curr_norm = torch.norm(self.full_precision_W-(deq_W+(self.scaling*(P@B))), 'fro')
            else:
                raise ValueError("proj_type must be 'left' or 'right'")
            
            if curr_norm < best_combination_norm:
                
                best_combination_norm = curr_norm
                best_B = B
                best_Q_W = Q_W
            curr_W_linear.weight.data = curr_W

        del curr_W
        return best_Q_W, best_B
    
    @torch.no_grad()
    def merge(self):
        self.maybe_dequantize_LoRA_factors()  # Dequantizes lora_A and lora_B if quantized
        
        if not self.is_single_gpu:
            # Perform distributed averaging to ensure a consistent state across all nodes
            dist.all_reduce(self.lora_A.weight.data, op=dist.ReduceOp.AVG) 
            dist.all_reduce(self.lora_B.weight.data, op=dist.ReduceOp.AVG)
            dist.barrier() # synchronization point where all processes must arrive before any can proceed. 
            
        AB = self.scaling*(self.lora_A.weight.T @ self.lora_B.weight.T).T.detach()  # Multiply and transpose to match dimensions
        
        # LoRA params not neeeded before init_B is called - make small placeholder tensors
        self.lora_A.weight.data = torch.ones((1,1), device=self.device, dtype=self.compute_dtype)
        self.lora_B.weight.data = torch.ones((1,1), device=self.device, dtype=self.compute_dtype)

        self.set_LoRA_requires_grad(False)
        torch.cuda.empty_cache()

        if self.quantize_w is None:
            # self.W.weight.data = torch.ones_like(self.W.weight.data)
            self.W.weight.data.add_(AB)
            self.full_precision_W = self.W.weight.data.detach().clone().to(self.offload_device)
            self.full_precision_W.requires_grad = False
        else:
            if self.quantize_w == '4bit':
                W_deq = bnb_F.dequantize_4bit(self.W.weight, self.W.weight.quant_state, quant_type=self.bnb_4bit_quant_type)
                W_deq = W_deq.to(dtype=self.compute_dtype)
                new_W = W_deq + AB
                self.full_precision_W = new_W.data.detach().clone().to(self.offload_device)
                    
                self.W.weight.data, self.W.weight.quant_state = bnb_F.quantize_4bit(new_W, quant_type=self.bnb_4bit_quant_type)
                del new_W
                del W_deq
        torch.cuda.synchronize()

        
    def set_W_requires_grad(self, requires_grad):
        # Reset gradient for W as it's not tracked by optimizer.
        self.W.weight.grad = None

        # Handle quantization-specific settings
        if self.quantize_w == '4bit':
            if isinstance(self.W, LinearNF4WithGradient):
                self.W.require_grad_W = torch.tensor(requires_grad, device=self.device, requires_grad=False)
                if requires_grad:
                    self.W.weight_grad = torch.zeros(
                        (self.out_features, self.in_features),
                        device=self.offload_device,
                        requires_grad=True,
                        dtype=self.compute_dtype
                    )
                    self.W.require_grad_W = torch.tensor(True, device=self.device, requires_grad=False)
                else:
                    del self.W.weight_grad
                    self.W.weight_grad = torch.tensor(0, device=self.offload_device, requires_grad=False, dtype=self.compute_dtype)
                    self.W.require_grad_W = torch.tensor(False, device=self.device, requires_grad=False)
            else:
                self.W.weight.requires_grad = requires_grad
        else:
            # No quantization or unrecognized quantization type
            self.W.weight.requires_grad = requires_grad

        # Handle bias: always enable requires_grad if bias exists
        if self.W.bias is not None:
            self.W.bias.requires_grad = True
    
    def reinitialize_LoRA_AB_after_merge(self):
        if self.init_lora_AB_as_random_and_zeros:
            self.initialize_LoRA_AB_random_and_zero()
        else:
            self.init_LoRA_with_gradient_projections()
            
    def initialize_LoRA_AB_random_and_zero(self):
        if self.proj_type == 'left':
            self.lora_A.weight.data = torch.randn(self.lora_A_shape, device=self.device, dtype=self.compute_dtype)
            self.lora_B.weight.data = torch.zeros(self.lora_B_shape, device=self.device, dtype=self.compute_dtype)
        else:
            self.lora_A.weight.data = torch.zeros(self.lora_A_shape, device=self.device, dtype=self.compute_dtype)
            self.lora_B.weight.data = torch.randn(self.lora_B_shape, device=self.device, dtype=self.compute_dtype)
    
    def init_LoRA_with_gradient_projections(self):
        #Either it is a linear layer and has requires_grad or it is a quantized (bnb) layer and has require_grad_W
        self.lora_A.weight.data = torch.zeros(self.lora_A_shape, device=self.device, dtype=self.compute_dtype)
        self.lora_B.weight.data = torch.zeros(self.lora_B_shape, device=self.device, dtype=self.compute_dtype)
        
        assert isinstance(self.W, nn.Linear) or self.W.require_grad_W 
        #TODO ensure this is correct wrt W being transposed or not
        if isinstance(self.W, LinearNF4WithGradient):
            self.W.weight_grad = self.W.weight_grad.to(self.device)
            if not self.is_single_gpu:
                #Gradient is stored in a placeholder variable as we cannot set require_grad=True for NF4 linear layer              
                dist.all_reduce(self.W.weight_grad, op=dist.ReduceOp.AVG)
                dist.barrier()
            W_grad = self.W.weight_grad.T
        else:
            self.W.weight.grad = self.W.weight.grad.to(self.device)
            if not self.is_single_gpu:
                dist.all_reduce(self.W.weight.grad, op=dist.ReduceOp.AVG)
                dist.barrier()
            W_grad = self.W.weight.grad.T #Weight is stored as transpose in nn.linear
        
        
        if torch.all(W_grad == 0):
            print("Gradient of W is zero")
        #assert not torch.all(W_grad == 0), "Gradient of W is zero"

        # Compute the projection matrices using the unified function
        U_r, Q_r = compute_projection_matrix(W_grad, self.r, method=self.projection_method, out=self.projection_out)
        
        if isinstance(self.W, LinearNF4WithGradient):
            del self.W.weight_grad
        del W_grad
        
        self.full_precision_W = self.full_precision_W.to(self.device)
    
        if self.proj_type == 'left':
            self.lora_A.weight.data.copy_(U_r.T)
            if self.quantize_projection_matrix == '4bit':
                self.quantize_LoRA_AB()
                deq_U_r = bnb_F.dequantize_4bit(self.lora_A.weight, self.lora_A.weight.quant_state, quant_type=self.bnb_4bit_quant_type)
                self.W, self.lora_B.weight.data = self.optimize_quant_and_lora_jointly(deq_U_r)
            elif self.quantize_w:
                self.W, self.lora_B.weight.data = self.optimize_quant_and_lora_jointly(U_r.T)
        else:  # proj_type == 'right'
            self.lora_B.weight.data.copy_(Q_r)
            if self.quantize_projection_matrix == '4bit':
                self.quantize_LoRA_AB()
                deq_Q_r = bnb_F.dequantize_4bit(self.lora_B.weight, self.lora_B.weight.quant_state, quant_type=self.bnb_4bit_quant_type)
                self.W, self.lora_A.weight.data = self.optimize_quant_and_lora_jointly(deq_Q_r)
            elif self.quantize_w == '4bit':
                self.W, self.lora_A.weight.data = self.optimize_quant_and_lora_jointly(Q_r)
        self.full_precision_W = None
        self.set_LoRA_requires_grad(True)
        torch.cuda.empty_cache()

        
    def quantize_LoRA_AB(self):
        if self.proj_type == 'right':
            self.lora_B = self.maybe_quantize(self.lora_B, self.quantize_projection_matrix, self.use_double_quant, self.bnb_4bit_quant_type)
        else:
            self.lora_A = self.maybe_quantize(self.lora_A, self.quantize_projection_matrix, self.use_double_quant, self.bnb_4bit_quant_type)

    
    def maybe_dequantize_LoRA_factors(self):
        # Directly modify lora_A or lora_B based on the projection type and quantization status
        if self.quantize_projection_matrix == '4bit':
            if self.proj_type == 'right':
                self.lora_B = self.dequantize_linear(self.lora_B, self.quantize_projection_matrix, self.bnb_4bit_quant_type, shape=self.lora_B_shape)
            else:
                self.lora_A = self.dequantize_linear(self.lora_A, self.quantize_projection_matrix, self.bnb_4bit_quant_type, shape=self.lora_A_shape)
        # If no quantization is applied, no changes are needed
    

    def dequantize_linear(self, linear_layer, quantize,  bnb_4bit_quant, shape):
        if quantize is None:
            return linear_layer
        elif quantize == "4bit":
            # Prepare an empty tensor with the same properties to hold dequantized data
            deq_weights = torch.empty(shape, dtype=self.compute_dtype, device=linear_layer.weight.device)
            # Dequantize directly into the prepared tensor
            bnb_F.dequantize_4bit(linear_layer.weight, linear_layer.weight.quant_state, out=deq_weights, quant_type=bnb_4bit_quant)
            # Replace the weight tensor in the linear layer with the dequantized weights
            linear_layer.weight.data = deq_weights
            return linear_layer
        else:
            raise ValueError("quantize must be None, or '4bit'")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        device = args[0] if args else kwargs.get("device", None)
        self.device = device

        def check_and_convert(module):
            if isinstance(module, LinearNF4WithGradient):
                module.weight.quant_state.to(device)

        for module in [self.lora_A, self.lora_B, self.W]:
            check_and_convert(module)

        return self
    
    
def compute_projection_matrix(grad_W, r, method='svd', out='u'):
    """
    Compute the projection matrices either using Singular Value Decomposition (SVD) or Eigendecomposition.

    Args:
        grad_W (torch.Tensor): The gradient matrix for which the decomposition will be computed.
        r (int): The number of components (singular vectors or eigenvectors) to be considered.
        method (str): The decomposition method to use ('svd' or 'eigh').
        out (str): Determines which set of eigenvectors ('u' or 'v') to return. Only used if method is 'eigh'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The components (U_r and Q_r) truncated to 'r' dimensions.
    """
    grad_input_type = grad_W.dtype
    grad_W = grad_W.to(torch.float32)  # Convert to float32 for numerical stability

    if method == 'svd':
        U, s, Vh = torch.linalg.svd(grad_W, full_matrices=False)
        U_r = U[:, :r].to(grad_input_type)
        Q_r = Vh[:r, :].T.to(grad_input_type)
        return U_r, Q_r
    elif method == 'eigh':
        if out not in ['u', 'v']:
            raise ValueError("Invalid output type for eigh. Choose 'u' or 'v'.")
        eigenvectors = eigenH_decomposition(grad_W, out)
        if out == 'u':
            U_r = eigenvectors[:, :r].to(grad_input_type)
            Q_r = eigenvectors[:, :r].T.to(grad_input_type)
        elif out == 'v':
            U_r = eigenvectors[:r, :].to(grad_input_type)
            Q_r = eigenvectors[:r, :].T.to(grad_input_type)
        return U_r, Q_r
    else:
        raise ValueError("Invalid method. Choose 'svd' or 'eigh'.")


def make_tensors_contiguous(model):
    for param in model.parameters():
        param.data = param.data.contiguous()