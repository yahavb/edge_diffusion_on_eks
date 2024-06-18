import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DEISMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

from PIL import Image

import time
import torch
import torch.nn as nn
import torch_neuronx

from diffusers import StableDiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

import time
import math

# Specialized benchmarking class for stable diffusion.
# We cannot use any of the pre-existing benchmarking utilities to benchmark E2E stable diffusion performance,
# because the top-level StableDiffusionPipeline cannot be serialized into a single Torchscript object.
# All of the pre-existing benchmarking utilities (in neuronperf or torch_neuronx) require the model to be a
# traced Torchscript.
def benchmark(n_runs, test_name, model, model_inputs):
     # model inputs can be tuple or dictionary
    if not isinstance(model_inputs, tuple) and not isinstance(model_inputs, dict):
        model_inputs = (model_inputs,)
        
    def run_model():
        if isinstance(model_inputs, dict):
            return model(**model_inputs)
        else : #tuple
            return model(*model_inputs)
    
    warmup_run = run_model()

    latency_collector = LatencyCollector()
    # can't use register_forward_pre_hook or register_forward_hook because StableDiffusionPipeline is not a torch.nn.Module
    
    for _ in range(n_runs):
        latency_collector.pre_hook()
        res = run_model()
        latency_collector.hook()
    
    p0_latency_ms = latency_collector.percentile(0) * 1000
    p50_latency_ms = latency_collector.percentile(50) * 1000
    p90_latency_ms = latency_collector.percentile(90) * 1000
    p95_latency_ms = latency_collector.percentile(95) * 1000
    p99_latency_ms = latency_collector.percentile(99) * 1000
    p100_latency_ms = latency_collector.percentile(100) * 1000

    report_dict = dict()
    report_dict["Latency P0"] = f'{p0_latency_ms:.1f}'
    report_dict["Latency P50"]=f'{p50_latency_ms:.1f}'
    report_dict["Latency P90"]=f'{p90_latency_ms:.1f}'
    report_dict["Latency P95"]=f'{p95_latency_ms:.1f}'
    report_dict["Latency P99"]=f'{p99_latency_ms:.1f}'
    report_dict["Latency P100"]=f'{p100_latency_ms:.1f}'

    report = f'RESULT FOR {test_name}:'
    for key, value in report_dict.items():
        report += f' {key}={value}'
    print(report)

class LatencyCollector:
    def __init__(self):
        self.start = None
        self.latency_list = []

    def pre_hook(self, *args):
        self.start = time.time()

    def hook(self, *args):
        self.latency_list.append(time.time() - self.start)

    def percentile(self, percent):
        latency_list = self.latency_list
        pos_float = len(latency_list) * percent / 100
        max_pos = len(latency_list) - 1
        pos_floor = min(math.floor(pos_float), max_pos)
        pos_ceil = min(math.ceil(pos_float), max_pos)
        latency_list = sorted(latency_list)
        return latency_list[pos_ceil] if pos_float - pos_floor > 0.5 else latency_list[pos_floor]


# For saving compiler artifacts
COMPILER_WORKDIR_ROOT = 'sd_1_5_controlnet'

# Model ID for SD version pipeline
model_id = "runwayml/stable-diffusion-v1-5"
controlnet_id = "lllyasviel/sd-controlnet-depth"

controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config, timestep_spacing="linspace")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
pipe.set_ip_adapter_scale(0.5)

unet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'unet/model.pt')
controlcore_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'controlnet_core/model.pt')
text_encoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'text_encoder/model.pt')
controlnet_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'controlnet/model.pt')
decoder_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_decoder/model.pt')
post_quant_conv_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'vae_post_quant_conv/model.pt')
safety_model_neuron_filename = os.path.join(COMPILER_WORKDIR_ROOT, 'safety_model/model.pt')

from compile_controlnet import NeuronUNet, UNetWrap, NeuronTextEncoder
from compile_controlnet import ControlNetWrap, NeuronControlNet 
from compile_controlnet import NeuronControlNetCore
from compile_controlnet import NeuronSafetyModelWrap

class NeuronControlNetCacher:
    pass

cacher = NeuronControlNetCacher()

def caching_forward(control_model_input, t, 
                    encoder_hidden_states, controlnet_cond,
                    conditioning_scale, guess_mode, return_dict=False
                    ):
    cacher.control_model_input = control_model_input
    cacher.controlnet_prompt_embeds = encoder_hidden_states
    cacher.controlnet_cond = controlnet_cond
    cacher.conditioning_scale = conditioning_scale
    cacher.guess_mode = guess_mode
    return None, None

# pipe.controlnet.forward = caching_forward
# device_ids = [0,1]
# loaded_core = torch_neuronx.DataParallel(torch.jit.load(controlcore_filename), device_ids, set_dynamic_batching=False)

# core = NeuronControlNetCore(cacher, loaded_core)
# pipe.unet.forward = core

# Load the compiled UNet onto two neuron cores.
pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
device_ids = [0,1]
pipe.unet.forward = torch_neuronx.DataParallel(torch.jit.load(unet_filename), device_ids, set_dynamic_batching=False)
pipe.controlnet.forward = torch_neuronx.DataParallel(torch.jit.load(controlnet_filename), device_ids, set_dynamic_batching=False)

# Load other compiled models onto a single neuron core.
pipe.text_encoder = NeuronTextEncoder(pipe.text_encoder)
pipe.text_encoder.neuron_text_encoder = torch.jit.load(text_encoder_filename)
pipe.vae.decoder = torch.jit.load(decoder_filename)
pipe.vae.post_quant_conv = torch.jit.load(post_quant_conv_filename)
pipe.safety_checker.vision_model = NeuronSafetyModelWrap(torch.jit.load(safety_model_neuron_filename))

# generate control image
from transformers import pipeline
depth_estimator = pipeline('depth-estimation')
# download an image
image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")

image = depth_estimator(image)['depth']
image = np.array(image)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)

control_image.save('cpu-fp32-contol-image.png')

# IP adapter image
id_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_einstein_base.png")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus-face_sd15.bin")
pipe.set_ip_adapter_scale(0.5)

# generate image
generator = torch.manual_seed(26)
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

image.save('neuron-fp32.png')

def benchmark_wrapper(img):
    return pipe(
    prompt="A photo of Einstein presenting his diploma.",
    negative_prompt="blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy",
    image=control_image,
    guidance_scale=7.5,
    ip_adapter_image=img,
    num_inference_steps=20,
    generator=generator,
    control_guidance_end=0.5,
    controlnet_conditioning_scale=0.5,
)

benchmark(10, "controlnet-ip-adapter", benchmark_wrapper, (id_image,))