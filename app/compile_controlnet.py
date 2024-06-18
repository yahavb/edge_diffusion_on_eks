import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import copy
import time
import torch
import torch.nn as nn
import torch_neuronx


from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DEISMultistepScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

# Compatibility for diffusers<0.18.0
from packaging import version
import diffusers
diffusers_version = version.parse(diffusers.__version__)
use_new_diffusers = diffusers_version >= version.parse('0.18.0')
if use_new_diffusers:
    from diffusers.models.attention_processor import Attention
else:
    from diffusers.models.cross_attention import CrossAttention


def get_attention_scores(self, query, key, attn_mask):    
    dtype = query.dtype

    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if(query.size() == key.size()):
        attention_scores = cust_badbmm(
            key,
            query.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=1).permute(0,2,1)
        attention_probs = attention_probs.to(dtype)

    else:
        attention_scores = cust_badbmm(
            query,
            key.transpose(-1, -2),
            self.scale
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.to(dtype)
        
    return attention_probs

def cust_badbmm(a, b, scale):
    bmm = torch.bmm(a, b)
    scaled = bmm * scale
    return scaled


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.timestep_cond = None
        self.cross_attention_kwargs = None

    def forward(self, 
                sample, timestep, encoder_hidden_states, 
                added_cond_kwargs,
                down_block_additional_residual_0,
                down_block_additional_residual_1,
                down_block_additional_residual_2,
                down_block_additional_residual_3,
                down_block_additional_residual_4,
                down_block_additional_residual_5,
                down_block_additional_residual_6,
                down_block_additional_residual_7,
                down_block_additional_residual_8,
                down_block_additional_residual_9,
                down_block_additional_residual_10,
                down_block_additional_residual_11,
                mid_block_additional_residual,
                ):
        down_block_additional_residuals = [
            down_block_additional_residual_0,
            down_block_additional_residual_1,
            down_block_additional_residual_2,
            down_block_additional_residual_3,
            down_block_additional_residual_4,
            down_block_additional_residual_5,
            down_block_additional_residual_6,
            down_block_additional_residual_7,
            down_block_additional_residual_8,
            down_block_additional_residual_9,
            down_block_additional_residual_10,
            down_block_additional_residual_11
            ]
        out_tuple = self.unet(sample, timestep, encoder_hidden_states, self.timestep_cond, self.cross_attention_kwargs,
                            down_block_additional_residuals=down_block_additional_residuals,
                            mid_block_additional_residual=mid_block_additional_residual,
                            added_cond_kwargs=added_cond_kwargs, return_dict=False)
        return out_tuple
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.device = unetwrap.unet.device
        self._load_ip_adapter_weights = unetwrap.unet._load_ip_adapter_weights
        self._load_ip_adapter_loras = unetwrap.unet._load_ip_adapter_loras
        self.attn_processors = unetwrap.unet.attn_processors
        self.down_blocks = unetwrap.unet.down_blocks
        self.up_blocks = unetwrap.unet.up_blocks
        self.encoder_hid_proj= unetwrap.unet.encoder_hid_proj
    
    def state_dict(self):
        return self.unetwrap.state_dict()

    def forward(self, sample, timestep, encoder_hidden_states, 
                down_block_additional_residuals,
                mid_block_additional_residual,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None, return_dict=False):
        
        self.unetwrap.timestep_cond=timestep_cond
        self.unetwrap.cross_attention_kwargs=cross_attention_kwargs

        sample = self.unetwrap(
            sample, 
            timestep.float().expand((sample.shape[0],)), 
            encoder_hidden_states,
            added_cond_kwargs,
            *down_block_additional_residuals,
            mid_block_additional_residual,
            )[0]
        return UNet2DConditionOutput(sample=sample)

class ControlNetWrap(nn.Module):
    def __init__(self, controlnet):
        super().__init__()
        self.controlnet = controlnet
        self.conditioning_scale = 0.5
        self.guess_mode = False

    def forward(self, sample, timestep, encoder_hidden_states,
                controlnet_cond):
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample=sample, 
            timestep=timestep, 
            encoder_hidden_states=encoder_hidden_states, 
            controlnet_cond=controlnet_cond, 
            conditioning_scale=self.conditioning_scale, 
            guess_mode=self.guess_mode, 
            return_dict=False)
        return *down_block_res_samples, mid_block_res_sample

class NeuronControlNet(nn.Module):
    def __init__(self, controlnet_wrap):
        super().__init__()
        self.controlnet_wrap = controlnet_wrap
    
    def forward(self, sample, timestep, encoder_hidden_states, 
                controlnet_cond, conditioning_scale=0.5, guess_mode=False, return_dict=False):
        
        self.controlnet_wrap.conditioning_scale = conditioning_scale
        self.controlnet_wrap.guess_mode = guess_mode
        sample = self.controlnet_wrap(
            sample, 
            timestep.float().expand((sample.shape[0],)), 
            encoder_hidden_states,
            controlnet_cond,
        )
        down_block_res_samples = sample[:-1]
        mid_block_res_sample = sample[-1]
        return down_block_res_samples, mid_block_res_sample

class NeuronControlNetCore(nn.Module):
    def __init__(self, cacher, controlnet_core):
        super().__init__()
        self.cacher = cacher
        self.controlnet_core = controlnet_core

    def forward(self, sample, timestep, encoder_hidden_states, 
                down_block_additional_residuals,
                mid_block_additional_residual,
                timestep_cond=None,
                cross_attention_kwargs=None,
                added_cond_kwargs=None, return_dict=False): 
        
        self.controlnet_core.cross_attention_kwargs = cross_attention_kwargs
        self.controlnet_core.conditioning_scale = self.cacher.conditioning_scale
        self.controlnet_core.guess_mode = self.cacher.guess_mode
        
        res = self.controlnet_core(
            self.cacher.control_model_input,
            timestep.float().expand((self.cacher.control_model_input.shape[0],)), 
            self.cacher.controlnet_prompt_embeds,
            self.cacher.controlnet_cond,
            sample, 
            encoder_hidden_states,
            timestep_cond,
            added_cond_kwargs,
            )
        return UNet2DConditionOutput(sample=res)
        

class NeuronControlNetCoreWrapper(nn.Module):
    def __init__(self, controlnet, unet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.cross_attention_kwargs = None
        self.conditioning_scale = 0.5
        self.guess_mode = False

    def forward(self, control_model_input, timestep, controlnet_prompt_embeds, controlnet_cond,
                latent_model_input, prompt_embeds, added_cond_kwargs,
                ):
        
        self.controlnet.conditioning_scale = self.conditioning_scale
        self.controlnet.guess_mode = self.guess_mode
        controlnet_res = self.controlnet(
            sample=control_model_input, 
            timestep = timestep.float().expand((control_model_input.shape[0],)), 
            encoder_hidden_states=controlnet_prompt_embeds, 
            controlnet_cond=controlnet_cond)
        
        self.unet.cross_attention_kwargs = self.cross_attention_kwargs
        noise_pred = self.unet(latent_model_input, timestep.float().expand((latent_model_input.shape[0],)), 
                               prompt_embeds, 
                               added_cond_kwargs,
                               *controlnet_res,
                               )[0]
        
        return noise_pred
        

class NeuronTextEncoder(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = torch.float32
        self.device = text_encoder.device

    def forward(self, emb, attention_mask = None):
        return [self.neuron_text_encoder(emb)['last_hidden_state']]


class NeuronSafetyModelWrap(nn.Module):
    def __init__(self, safety_model):
        super().__init__()
        self.safety_model = safety_model

    def forward(self, clip_inputs):
        return list(self.safety_model(clip_inputs).values())



# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'sd_1_5_controlnet'

# Model ID for SD version pipeline
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/sd-controlnet-depth"



def trace_text_encoder():
    # --- Compile CLIP text encoder and save ---

    # Only keep the model being compiled in RAM to minimze memory pressure
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)
    text_encoder = copy.deepcopy(pipe.text_encoder)
    del pipe

    # Apply the wrapper to deal with custom return type
    text_encoder = NeuronTextEncoder(text_encoder)

    # Compile text encoder
    # This is used for indexing a lookup table in torch.nn.Embedding,
    # so using random numbers may give errors (out of range).
    emb = torch.tensor([[49406, 18376,   525,  7496, 49407,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]])

    with torch.no_grad():
        start_time = time.time()
        text_encoder_neuron = torch_neuronx.trace(
                text_encoder.neuron_text_encoder, 
                emb, 
                compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder'),
                compiler_args=["--enable-fast-loading-neuron-binaries"]
                )
        text_encoder_neuron_compile_time = time.time() - start_time
        print('text_encoder_neuron_compile_time:', text_encoder_neuron_compile_time)

    # Save the compiled text encoder
    text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
    torch_neuronx.async_load(text_encoder_neuron)
    torch.jit.save(text_encoder_neuron, text_encoder_filename)

    # delete unused objects
    del text_encoder
    del text_encoder_neuron
    del emb

# --- Compile VAE decoder and save ---
    
def trace_vae_decoder():

    # Only keep the model being compiled in RAM to minimze memory pressure
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)
    decoder = copy.deepcopy(pipe.vae.decoder)
    del pipe

    # Compile vae decoder
    decoder_in = torch.randn([1, 4, 57, 57])
    with torch.no_grad():
        start_time = time.time()
        decoder_neuron = torch_neuronx.trace(
            decoder, 
            decoder_in, 
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder'),
            compiler_args=["--model-type=vae-inference", "--enable-fast-loading-neuron-binaries"]
        )
        vae_decoder_compile_time = time.time() - start_time
        print('vae_decoder_compile_time:', vae_decoder_compile_time)

    # Save the compiled vae decoder
    decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
    torch_neuronx.async_load(decoder_neuron)
    torch.jit.save(decoder_neuron, decoder_filename)

    # delete unused objects
    del decoder
    del decoder_in
    del decoder_neuron


def trace_controlnet():

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)

    # Replace original cross-attention module with custom cross-attention module for better performance
    if use_new_diffusers:
        Attention.get_attention_scores = get_attention_scores
    else:
        CrossAttention.get_attention_scores = get_attention_scores

    # Apply double wrapper to deal with custom return type
    pipe.controlnet = NeuronControlNet(ControlNetWrap(pipe.controlnet))

    # Only keep the model being compiled in RAM to minimze memory pressure
    controlnet = copy.deepcopy(pipe.controlnet.controlnet_wrap)
    del pipe

    # Compile controlnet - FP32
    sample_1b = torch.randn([1, 4, 57, 57])
    timestep_1b = torch.tensor(999).float().expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 768])
    controlnet_cond = torch.randn([1, 3, 456, 456])
    controlnet.conditioning_scale = 0.5
    controlnet.guess_mode = False

    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b, controlnet_cond

    with torch.no_grad():
        start_time = time.time()
        controlnet_neuron = torch_neuronx.trace(
            controlnet,
            example_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'controlnet'),
            compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
        )
        controlnet_compile_time = time.time() - start_time
        print('controlnet_compile_time:', controlnet_compile_time)

    # Enable asynchronous and lazy loading to speed up model load
    torch_neuronx.async_load(controlnet_neuron)
    torch_neuronx.lazy_load(controlnet_neuron)

    # save compiled unet
    controlnet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'controlnet/model.pt')
    torch.jit.save(controlnet_neuron, controlnet_filename)

    # delete unused objects
    del controlnet
    del controlnet_neuron
    del sample_1b
    del timestep_1b
    del encoder_hidden_states_1b


# --- Compile UNet and save ---
    
def trace_unet(ip_adapter):

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)

    # Replace original cross-attention module with custom cross-attention module for better performance
    if use_new_diffusers:
        Attention.get_attention_scores = get_attention_scores
    else:
        CrossAttention.get_attention_scores = get_attention_scores

    # Apply double wrapper to deal with custom return type
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

    # Only keep the model being compiled in RAM to minimze memory pressure
    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe

    # Compile unet - FP32
    sample_1b = torch.randn([1, 4, 57, 57])
    timestep_1b = torch.tensor(999).float().expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 768])

    # 12 samples
    down_block_res_samples_sizes = [
        torch.Size([1, 320, 57, 57]),
        torch.Size([1, 320, 57, 57]),
        torch.Size([1, 320, 57, 57]),
        torch.Size([1, 320, 29, 29]),
        torch.Size([1, 640, 29, 29]),
        torch.Size([1, 640, 29, 29]),
        torch.Size([1, 640, 15, 15]),
        torch.Size([1, 1280, 15, 15]),
        torch.Size([1, 1280, 15, 15]),
        torch.Size([1, 1280, 8, 8]),
        torch.Size([1, 1280, 8, 8]),
        torch.Size([1, 1280, 8, 8]),
        ]
    down_block_res_samples = [torch.randn(size) for size in down_block_res_samples_sizes]
    mid_block_res_sample = torch.randn(torch.Size([1, 1280, 8, 8]))
    unet.cross_attention_kwargs = None
    unet.timestep_cond = None

    if ip_adapter:
        image_embeds = [torch.randn([1, 1, 257, 1280])]
        added_cond_kwargs = {"image_embeds": image_embeds}

    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b, added_cond_kwargs, *down_block_res_samples, mid_block_res_sample

    with torch.no_grad():
        start_time = time.time()
        unet_neuron = torch_neuronx.trace(
            unet,
            example_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'unet'),
            compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
        )
        unet_compile_time = time.time() - start_time
        print('unet_compile_time:', unet_compile_time)

    # Enable asynchronous and lazy loading to speed up model load
    torch_neuronx.async_load(unet_neuron)
    torch_neuronx.lazy_load(unet_neuron)

    # save compiled unet
    unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
    torch.jit.save(unet_neuron, unet_filename)

    # delete unused objects
    del unet
    del unet_neuron
    del sample_1b
    del timestep_1b
    del encoder_hidden_states_1b


def trace_controlnet_core(ip_adapter):

    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)

    # Replace original cross-attention module with custom cross-attention module for better performance
    if use_new_diffusers:
        Attention.get_attention_scores = get_attention_scores
    else:
        CrossAttention.get_attention_scores = get_attention_scores

    controlnet = ControlNetWrap(pipe.controlnet)
    controlnet.conditioning_scale = 0.5
    controlnet.guess_mode = False

    unet = UNetWrap(pipe.unet)
    unet.cross_attention_kwargs = None
    unet.timestep_cond = None
    
    controlnet_core = NeuronControlNetCoreWrapper(controlnet, unet)

    del pipe

    # Compile controlnet - FP32
    sample_1b = torch.randn([1, 4, 57, 57])
    timestep_1b = torch.tensor(999).float().expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 768])
    controlnet_cond = torch.randn([1, 3, 456, 456])
    latent_model_input = torch.randn([1, 4, 57, 57])
    prompt_embeds = torch.randn([1, 77, 768])

    if ip_adapter:
        image_embeds = [torch.randn([1, 1, 257, 1280])]
        added_cond_kwargs = {"image_embeds": image_embeds}

    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b, controlnet_cond,\
        latent_model_input, prompt_embeds, added_cond_kwargs

    with torch.no_grad():
        start_time = time.time()
        controlnet_neuron = torch_neuronx.trace(
            controlnet_core,
            example_inputs,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'controlnet_core'),
            compiler_args=["--model-type=unet-inference", "--enable-fast-loading-neuron-binaries"]
        )
        controlnetcore_compile_time = time.time() - start_time
        print('controlnetcore_compile_time:', controlnetcore_compile_time)

    # Enable asynchronous and lazy loading to speed up model load
    torch_neuronx.async_load(controlnet_neuron)
    torch_neuronx.lazy_load(controlnet_neuron)

    # save compiled unet
    controlnet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'controlnet_core/model.pt')
    torch.jit.save(controlnet_neuron, controlnet_filename)

    # delete unused objects
    del controlnet
    del controlnet_neuron
    del sample_1b
    del timestep_1b
    del encoder_hidden_states_1b

# --- Compile VAE post_quant_conv and save ---
    
def trace_vae_post_quant():

    # Only keep the model being compiled in RAM to minimze memory pressure
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)

    post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
    del pipe

    # Compile vae post_quant_conv
    post_quant_conv_in = torch.randn([1, 4, 57, 57])
    with torch.no_grad():
        start_time = time.time()
        post_quant_conv_neuron = torch_neuronx.trace(
            post_quant_conv, 
            post_quant_conv_in,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )
        vae_post_quant_conv_compile_time = time.time() - start_time
        print('vae_post_quant_conv_compile_time:', vae_post_quant_conv_compile_time)

    # Save the compiled vae post_quant_conv
    post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
    torch_neuronx.async_load(post_quant_conv_neuron)
    torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

    # delete unused objects
    del post_quant_conv



# --- Compile safety checker and save ---

def trace_safety_checker():

    # Only keep the model being compiled in RAM to minimze memory pressure
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)

    safety_model = copy.deepcopy(pipe.safety_checker.vision_model)
    del pipe

    clip_input = torch.randn([1, 3, 224, 224])
    with torch.no_grad():
        start_time = time.time()
        safety_model = torch_neuronx.trace(
            safety_model, 
            clip_input,
            compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model'),
            compiler_args=["--enable-fast-loading-neuron-binaries"]
        )
        safety_model_compile_time = time.time() - start_time
        print('safety_model_compile_time:', safety_model_compile_time)

    # Save the compiled safety checker
    safety_model_neuron_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model/model.pt')
    torch_neuronx.async_load(safety_model)
    torch.jit.save(safety_model, safety_model_neuron_filename)

    # delete unused objects
    del safety_model

# print('Total compile time:', text_encoder_neuron_compile_time + vae_decoder_compile_time + unet_compile_time + vae_post_quant_conv_compile_time + safety_model_compile_time)


def cpu_fp32():
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DEISMultistepScheduler
    from diffusers.utils import load_image
    from transformers import pipeline
    import numpy as np
    import torch

    from PIL import Image


    depth_estimator = pipeline('depth-estimation')
    # download an image
    image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")

    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    control_image.save('cpu-fp32-contol-image.png')

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float32
    )

    # speed up diffusion process with faster scheduler and memory optimization
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")

    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(0.5)

    # generate image
    id_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")
    generator = torch.Generator(device="cpu").manual_seed(26)
    print(id_image.size)
    print(control_image.size)
    image = pipe(
        prompt="A photo of Einstein presenting his diploma.",
        negative_prompt="blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy",
        image=control_image,
        guidance_scale=7.5,
        ip_adapter_image=id_image,
        num_inference_steps=20,
        generator=generator,
        control_guidance_end=0.5,
        controlnet_conditioning_scale=0.5,
    ).images[0]

    image.save('cpu-fp32.png')

if __name__ == "__main__":

    ip_adapter = True
    # DEBUG only: trace unet and controlnet separately causes perf degrade. 
    # use the trace_controlnet_core() instead
    trace_unet(ip_adapter)
    trace_controlnet()
    trace_text_encoder()
    trace_vae_post_quant()
    trace_vae_decoder()
    # trace_controlnet_core(ip_adapter)
    trace_safety_checker()