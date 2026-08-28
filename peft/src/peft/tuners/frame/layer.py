# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
import pdb
import math
import warnings
from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose

# ==========================================================
# Tight	Fusion Frame Existence Test
def	tffet(k,l,n):
	if 2*l > n:
		l =	n-l
	exists = 'unknown'
	while exists ==	'unknown':
		if n%l == 0:
			if k >=	n/l:
				exists = True
			else:
				exists = False
		else:
			if k > math.ceil(n/l) +	1:
				exists = True
			elif k < math.ceil(n/l)	+ 1:
				exists = False
			else:
				n =	k*l	- n
				l =	n-l

	return exists

def	insert_tx(x, tff, r,c):
	tx = (1/math.sqrt(2)) *	torch.as_tensor([[math.sqrt(x),	math.sqrt(x)], [math.sqrt(2-x),	-math.sqrt(2-x)]])
	tff[r:r+2, c:c+2] =	tx

# construct	tight fusion frames
def	construct_tight_frames(k,l,n):
	existence =	tffet(k,l,n)
	if not existence:
		print('the given k,l,n values are invalid')
		exit

	# constructing a n length frame	for	C^l
	tff_lxn	= torch.zeros((l,n))
	tff_lxn[0,0] = 0
	tff_lxn[0,1] = 0
	target_norm	= n/l

	# fill the matrix
	col	= 0
	block_indices =	torch.zeros((l,2), dtype=int) -	1
	for	row	in range(l):
		if block_indices[row,0]	== -1:
			block_indices[row,0] = col
		curr_norm =	torch.norm(tff_lxn[row,	:col])**2
		req_norm = target_norm - curr_norm
		while (req_norm	>=1.) or math.isclose(req_norm,	1,abs_tol =	1e-5):
			tff_lxn[row, col] =	1
			req_norm -=	1
			col	+= 1
		if not math.isclose(req_norm, 0, abs_tol=1e-5):
			if row != l-1:
				block_indices[row+1,0] = col
			insert_tx(req_norm,	tff_lxn, row, col)
			col	+= 2

		block_indices[row,1] = col

	# part 2 - Modulated TFFs
	tffs = []
	for	_k in range(k):
		wt_mat = torch.as_tensor([2*math.pi*(_k)*_n/k for _n in	range(n)])
		tffs.append(torch.polar(tff_lxn, wt_mat))
	# tffs = torch.stack(tffs,0) * math.sqrt(l/n)
	tffs = torch.stack(tffs,0) * math.sqrt(1/k)
	return tffs, block_indices

def	construct_real_tff(k,l,n):
	tffs, block_indices	= construct_tight_frames(k,l,n)	# /	math.sqrt(2) # normalize the complex basis to get unit norm

	# size of real tffs	is (k,2l,2n)
	tffs_intrmd	= torch.view_as_real(tffs)
	tffs_even_l	= tffs_intrmd.view(k,l,2*n)	* ((-1)**torch.arange(2*n))
	tffs_odd_l = torch.roll(tffs_intrmd, shifts=(1), dims=(-1)).view(k,l,2*n)
	tffs_real =	torch.stack((tffs_even_l, tffs_odd_l), dim=2).view(k,2*l,2*n)

	block_indices *= 2

	return tffs_real, block_indices

# ==========================================================
# master copies of the frames, kept on cpu and keyed by (n, l)
tffs = {}
# per-device copies, keyed by ((n, l), device)
_tffs_on_device = {}


def get_tff(key, device):
    """Return the frame for `key` on `device`, caching one copy per device.

    The frames are read-only, so a single copy per device is shared by every layer that
    lives on it. This is what makes a model sharded over several GPUs work.
    """
    device = torch.device(device)
    cache_key = (key, device)
    tff = _tffs_on_device.get(cache_key)
    if tff is None:
        tff = tffs[key].to(device)
        _tffs_on_device[cache_key] = tff
    return tff


def FFT_SHIFT(matrix):
        m_clone = matrix.clone()
        m,n = m_clone.shape
        m = int(m / 2)
        n = int(n / 2)

        for i in range(m):
            for j in range(n):
                m_clone[i][j] = matrix[m+i][n+j]
                m_clone[m+i][n+j] = matrix[i][j]
                m_clone[m+i][j] = matrix[i][j+n]
                m_clone[i][j+n] = matrix[m+i][j]
        return m_clone

class FrameLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ["ff_coeffs"]
    # All names of other parameters that may contain adapter-related parameters
    # other_param_names = ("rank", "dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.n_ff_coeffs = {}
        # self.lora_alpha = {}
        self.scale = {}
        # self.dropout = nn.ModuleDict({})
        self.ff_coeffs = nn.ParameterDict({})
        self.indices = {}
        # per-adapter (n, l) keys into the module-level `tffs` cache for the in/out frames
        self.tff_keys = {}
        # For Embedding layer
        # self.lora_embedding_A = nn.ParameterDict({})
        # self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        # elif isinstance(base_layer, nn.Conv2d):
        #     in_features, out_features = base_layer.in_channels, base_layer.out_channels
        # elif isinstance(base_layer, nn.Embedding):
        #     in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        # elif isinstance(base_layer, Conv1D):
        #     in_features, out_features = (
        #         base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
        #     )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features


    def update_layer(self, adapter_name, n_ff_coeffs, scale, layer_num, init_std, tff_l, tff_block_size, share_entry, entry_seed, init_frame_weights=None, tff_l_out=None, tff_block_size_out=None, tff_num_blocks=None):
        print(f'{layer_num} {adapter_name} {n_ff_coeffs} {scale} {share_entry = }')
        if n_ff_coeffs <= 0:
            raise ValueError(f"`n_ff_coeffs` should be a positive integer value but the value passed is {n_ff_coeffs}")
        self.n_ff_coeffs[adapter_name] = n_ff_coeffs
        self.scale[adapter_name] = scale

        # out-dim frame parameters default to the in-dim ones (square behavior)
        if tff_l_out is None:
            tff_l_out = tff_l
        if tff_block_size_out is None:
            tff_block_size_out = tff_block_size

        n_in, l_in = self.in_features, tff_l
        n_out, l_out = self.out_features, tff_l_out
        if n_in % l_in != 0:
            raise ValueError(f"`tff_l` ({l_in}) must divide in_features ({n_in})")
        if n_out % l_out != 0:
            raise ValueError(f"`tff_l_out` ({l_out}) must divide out_features ({n_out})")

        # construct the in/out tffs if not already cached
        for n, l in ((n_in, l_in), (n_out, l_out)):
            if (n, l) not in tffs:
                k = n // l
                tff, _ = construct_real_tff(k, l//2, n//2)
                tffs[(n, l)] = tff.view(-1, n)
        self.tff_keys[adapter_name] = ((n_in, l_in), (n_out, l_out))

        if tff_num_blocks is not None:
            if n_in % tff_num_blocks != 0 or n_out % tff_num_blocks != 0:
                raise ValueError(
                    f"`tff_num_blocks` ({tff_num_blocks}) must divide both in_features ({n_in}) "
                    f"and out_features ({n_out})"
                )
            cb_in = n_in // tff_num_blocks
            cb_out = n_out // tff_num_blocks
        else:
            cb_in = tff_block_size
            cb_out = tff_block_size_out

        num_coeffs_per_subspace = cb_in * cb_out
        num_blks = n_in // cb_in
        if num_blks != n_out // cb_out:
            raise ValueError(
                f"in/out block counts must match: in_features ({n_in}) / tff_block_size ({cb_in}) = {num_blks} but "
                f"out_features ({n_out}) / tff_block_size_out ({cb_out}) = {n_out // cb_out}"
            )
        total_num_frame_coefficients = num_blks * num_coeffs_per_subspace

        if share_entry:
            print('\033[32m Using shared entry... \033[0m')
            indices = torch.randperm(total_num_frame_coefficients,generator=torch.Generator().manual_seed(entry_seed))[:n_ff_coeffs]
        else:
            indices = torch.randperm(total_num_frame_coefficients,generator=torch.Generator().manual_seed(entry_seed + 1000*layer_num))[:n_ff_coeffs]

        subspaces = indices // num_coeffs_per_subspace
        indices_within_subspace = indices % num_coeffs_per_subspace

        row_indices, col_indices = indices_within_subspace // cb_out, indices_within_subspace % cb_out
        final_row_indices, final_col_indices = row_indices + subspaces * cb_in, col_indices + subspaces * cb_out
        self.indices[adapter_name] = final_row_indices * n_out + final_col_indices

        self.indices[adapter_name] = torch.stack([self.indices[adapter_name] // n_out, self.indices[adapter_name] % n_out], dim=0)
        self.ff_coeffs[adapter_name] = nn.Parameter(torch.randn(n_ff_coeffs) * init_std, requires_grad=True)
  
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def reset_frame_parameters(self, adapter_name, init_frame_weights):
        if init_frame_weights is False:
            return

        if adapter_name in self.ff_coeffs.keys():
            if init_frame_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.ff_coeffs[adapter_name].weight, a=math.sqrt(5))
            elif init_frame_weights.lower() == "gaussian":
                nn.init.normal_(self.ff_coeffs[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_frame_weights=}")
        if adapter_name in self.ff_coeffs.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.ff_coeffs[adapter_name])


class Linear(nn.Module, FrameLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_ff_coeffs: int = 0,
        scale: float = 0.1,
        layer_num: int = 0,
        init_std: float = 1.0,
        tff_l: int = 2,
        tff_block_size: int = 768,
        tff_l_out: Optional[int] = None,
        tff_block_size_out: Optional[int] = None,
        tff_num_blocks: Optional[int] = None,
        share_entry: bool = False,
        entry_seed: int = 2024,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_frame_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FrameLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, n_ff_coeffs, scale, layer_num, init_std, tff_l, tff_block_size, share_entry, entry_seed, init_frame_weights, tff_l_out=tff_l_out, tff_block_size_out=tff_block_size_out, tff_num_blocks=tff_num_blocks)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.ff_coeffs.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.ff_coeffs.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        ff_coeffs = self.ff_coeffs[adapter]
        device = ff_coeffs.device
        dtype = ff_coeffs.dtype
        indices = self.indices[adapter].to(device)
        scale = self.scale[adapter]
        key_in, key_out = self.tff_keys[adapter]

        dense_s = torch.zeros((self.in_features, self.out_features), dtype=torch.float32, device=device)
        dense_s[indices[0, :], indices[1, :]] = ff_coeffs.float()

        tff_in = get_tff(key_in, device).to(dtype=torch.float32)
        tff_out = get_tff(key_out, device).to(dtype=torch.float32)

        delta_w = tff_in.T @ dense_s @ tff_out * scale / math.sqrt(self.in_features * self.out_features)
        # forward applies x @ delta_w, so the base-weight delta is its transpose
        output_tensor = transpose(delta_w.T, self.fan_in_fan_out)

        return output_tensor.to(dtype=dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ff_coeffs.keys():
                    continue
                
                ff_coeffs = self.ff_coeffs[active_adapter]
                indices = self.indices[active_adapter].to(ff_coeffs.device)
                scale = self.scale[active_adapter]
                key_in, key_out = self.tff_keys[active_adapter]

                compute_dtype = x.dtype
                if compute_dtype == torch.bfloat16:
                    compute_dtype = torch.float16

                device = ff_coeffs.device
                dense_s = torch.zeros((self.in_features, self.out_features), dtype=compute_dtype, device=device)
                dense_s[indices[0, :], indices[1, :]] = ff_coeffs.to(compute_dtype)

                tff_in = get_tff(key_in, device).to(compute_dtype)
                tff_out = get_tff(key_out, device).to(compute_dtype)
                delta_w = tff_in.T @ dense_s @ tff_out * scale / math.sqrt(self.in_features * self.out_features)
                x, delta_w = x.to(compute_dtype), delta_w.to(compute_dtype)
                result += torch.einsum('ijk,kl->ijl', x, delta_w)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "frame." + rep