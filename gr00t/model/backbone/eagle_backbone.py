# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature
import numpy as np

import gr00t

DEFAULT_EAGLE_PATH = os.path.join(
    os.path.dirname(gr00t.__file__), "model", "backbone", "eagle2_hg_model"
)


class EagleBackbone(nn.Module):

    def __init__(
        self,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = False,
        use_flash_attention: bool = False,
        load_bf16: bool = False,
        eagle_path: str | None = None,
        project_to_dim: int = 1536,
        ##################################################
        save_hidden_state: bool = False,
        save_hidden_state_dir: str = "debug_hidden_states",
        save_hidden_state_max_steps: int = 1,
        ##################################################
    ):
        """
        Args:
            tune_llm: whether to tune the LLM model (default: True)
            tune_visual: whether to tune the visual model (default: False)
        """
        super().__init__()
        assert not reproject_vision, "Reproject vision is not implemented here, set to False"

        ##################################################
        self.save_hidden_state = save_hidden_state
        self.save_hidden_state_dir = save_hidden_state_dir
        self.save_hidden_state_max_steps = save_hidden_state_max_steps
        self._save_hidden_state_step_count = 0
        ##################################################

        config = AutoConfig.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
        self.eagle_model = AutoModel.from_config(config, trust_remote_code=True)

        if project_to_dim is not None:
            self.eagle_linear = torch.nn.Linear(2048, project_to_dim)
        else:
            self.eagle_linear = torch.nn.Identity()

        # needed since we don't use these layers. Also saves compute
        while len(self.eagle_model.language_model.model.layers) > select_layer:
            self.eagle_model.language_model.model.layers.pop(-1)

        self.select_layer = select_layer
        
        # self.kept_layer_idx_list = [1, 2, 10, 11] # CKA + cosine
        
        # self.kept_layer_idx_list = [0, 1, 9, 10, 11] # cosine
        
        self.kept_layer_idx_list = [0, 1, 2, 10, 11] # cka
        
        # self.kept_layer_idx_list = [0, 1, 2, 3, 4] # same cka
        # self.kept_layer_idx_list = [0, 1, 2, 3] # smaller than cka
        # self.kept_layer_idx_list = [1, 3, 7, 8, 10] # random
        
        # 3/4
        # self.kept_layer_idx_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        
        # 2/4
        # self.kept_layer_idx_list = [0, 1, 2, 3, 4, 5]
        
        # # 1/4
        # self.kept_layer_idx_list = [0, 1, 2]
        
        
        
        
        # if len(self.eagle_model.language_model.model.layers) > len(self.kept_layer_idx_list):
        #     print("Start pruning  VLM...")
        #     print(f"Pruning {len(self.eagle_model.language_model.model.layers)} layers to {len(self.kept_layer_idx_list)} layers. Kept layers are: {self.kept_layer_idx_list}")    
        #     self.prun_layers()
        
        self.set_trainable_parameters(tune_llm, tune_visual)
        
        # import pdb 
        # pdb.set_trace()

    def prun_layers(self):
        # prun paligemma 
        print("Start pruning  VLM...")
        print(f"Pruning {len(self.eagle_model.language_model.model.layers)} layers to {len(self.kept_layer_idx_list)} layers. Kept layers are: {self.kept_layer_idx_list}")    
        old_language_transformer_layers = self.eagle_model.language_model.model.layers
        self.eagle_model.language_model.model.layers = torch.nn.ModuleList(
            [old_language_transformer_layers[i] for i in self.kept_layer_idx_list]
        ) 
        self.select_layer = len(self.kept_layer_idx_list)
        # self.paligemma.config.text_config.num_hidden_layers = len(self.kept_layer_idx_list)
        
        
        # pass
        
    def _save_eagle_hidden_states(self, hidden_states):
        """Dump each layer's hidden state tensor to {save_hidden_state_dir}/layer_{i}.npy"""
        os.makedirs(self.save_hidden_state_dir, exist_ok=True)
        print(
            f"[EagleBackbone] Saving {len(hidden_states)} hidden state layers to "
            f"'{self.save_hidden_state_dir}' (step {self._save_hidden_state_step_count})"
        )
        for layer_idx, hs in enumerate(hidden_states):
            hs_np = hs.detach().cpu().float().numpy()
            np.save(
                os.path.join(
                    self.save_hidden_state_dir,
                    f"step{self._save_hidden_state_step_count}_layer_{layer_idx}.npy",
                ),
                hs_np,
            )

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
            
        if not tune_llm:
            self.eagle_model.language_model.requires_grad_(False)
            
            
        if not tune_visual:
            self.eagle_model.vision_model.requires_grad_(False)
            self.eagle_model.mlp1.requires_grad_(False)
        print(f"Tune backbone llm: {self.tune_llm}")
        print(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_llm and not tune_visual:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No backbone trainable parameters found.")
        
        self.eagle_model.language_model.requires_grad_(False)
        # set first 6 layers trainable here
        
        # for i, layer in enumerate(self.eagle_model.language_model.model.layers):
        #     if i < 6:
        #         for p in layer.parameters():
        #             p.requires_grad = True

        # fix bug multi gpus here:
        self.eagle_model.language_model.lm_head.requires_grad_(False)
        
        

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.eagle_model.language_model and not self.tune_llm:
                self.eagle_model.language_model.eval()
            if self.eagle_model.vision_model and not self.tune_visual:
                self.eagle_model.vision_model.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward_eagle(self, vl_input: BatchFeature) -> BatchFeature:
        eagle_prefix = "eagle_"
        eagle_input = {
            k.removeprefix(eagle_prefix): v
            for k, v in vl_input.items()
            if k.startswith(eagle_prefix)
        }
        del eagle_input["image_sizes"]

        # import pdb 
        # pdb.set_trace()
        
        eagle_output = self.eagle_model(**eagle_input, output_hidden_states=True, return_dict=True)
        eagle_features = eagle_output.hidden_states[self.select_layer]
        
        # save hidden state here
        # for layer_idx, _att_weight in enumerate(list(attn_maps)):
        #     _att_weight_np = _att_weight.detach().cpu().float().numpy()
        #     np.save(f"attn/vlm/npfiles/gr15_atan_layer_{layer_idx}.npy", _att_weight_np)
        # for layer_idx, hs in enumerate(list(eagle_output.hidden_states)):
        #     hs_np = hs.detach().cpu().float().numpy()
        #     np.save(f"attn/gr15_pr_keeprandom_12/hs/vlm/layer_{layer_idx}.npy", hs_np)
        
        # import pdb 
        # pdb.set_trace()
        
        if self.save_hidden_state and self._save_hidden_state_step_count < self.save_hidden_state_max_steps:
            self._save_eagle_hidden_states(eagle_output.hidden_states)
            self._save_hidden_state_step_count += 1
        
         
        eagle_features = self.eagle_linear(eagle_features)
        return eagle_features, eagle_input["attention_mask"]

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()

        eagle_embeds, eagle_mask = self.forward_eagle(vl_input)

        # YL (TODO HACK): to resolve DDP issue when tune_visual=True
        # Ensure all trainable parameters in vision_model are used in the forward pass for DDP compatibility
        if self.training and self.tune_visual:
            dummy_term = torch.tensor(
                0.0, device=eagle_embeds.device, dtype=eagle_embeds.dtype, requires_grad=True
            )
            for param in self.eagle_model.vision_model.parameters():
                if param.requires_grad:
                    dummy_term = dummy_term + 0.0 * param.sum()
            eagle_embeds = eagle_embeds + dummy_term

        return BatchFeature(
            data={"backbone_features": eagle_embeds, "backbone_attention_mask": eagle_mask}
        )  # [B, T2, hidden_size]
