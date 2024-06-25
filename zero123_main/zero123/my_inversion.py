from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm, trange
import torch
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
import shutil
from torch.optim.adam import Adam
from PIL import Image
import time
import cv2
from ldm.util import instantiate_from_config
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from ldm.models.diffusion.ddim import DDIMSampler
import os


# os.environ['CUDA_VISIBLE_DEVICES']='7'



def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))




class LocalBlend:
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


        
class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_word)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), prefix='cross_attention')
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), prefix='self_attention')

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:,:,:3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((256, 256)))
    return image






def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()




class NullInversion:

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)


    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray],index:int):
        # prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        # alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        # beta_prod_t = 1 - alpha_prod_t
        # pred_original_sample = (sample.to(device) - beta_prod_t.to(device) ** 0.5 * model_output) / alpha_prod_t.to(device) ** 0.5
        # pred_sample_direction = (1 - alpha_prod_t_prev.to(device)) ** 0.5 * model_output
        # prev_sample = alpha_prod_t_prev.to(device) ** 0.5 * pred_original_sample.to(device) + pred_sample_direction.to(device)
        # return prev_sample
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((1, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((1, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((1, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((1, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (sample - sqrt_one_minus_at * model_output) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * model_output
        noise = sigma_t * noise_like(sample.shape, device, False) * 1
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray],index:int):
        # timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        # alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        # beta_prod_t = 1 - alpha_prod_t
        # next_original_sample = (sample.to(device) - beta_prod_t.to(device) ** 0.5 * model_output.to(device)) / alpha_prod_t.to(device) ** 0.5
        # next_sample_direction = (1 - alpha_prod_t_next.to(device)) ** 0.5 * model_output
        # next_sample = alpha_prod_t_next.to(device) ** 0.5 * next_original_sample.to(device) + next_sample_direction.to(device)
        # return next_sample
        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((1, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((1, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((1, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((1, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (sample - sqrt_one_minus_at * model_output) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * model_output
        noise = sigma_t * noise_like(sample.shape, device, False) * 1
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    
    def get_noise_pred_single(self, latents, t, context,is_uncond=False,uncond_embedding=None,uncond_embedding_c=None):
        # noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        if not is_uncond:
            cond={}
            cond["c_concat"]=[self.latents_image]
            cond["c_crossattn"]=[context]
            noise_pred=self.model.apply_model(latents,t,cond)
        elif is_uncond:
            cond={}
            cond["c_concat"]=[uncond_embedding_c]
            cond["c_crossattn"]=[uncond_embedding]
            noise_pred=self.model.apply_model(latents,t,cond)
        return noise_pred


#需要修改
    def get_noise_pred(self, latents, t, is_forward=True, context=None,index=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        # noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        cond={}
        cond["c_concat"]=[torch.cat([torch.zeros(self.latents_image.shape).cuda(),self.latents_image])]
        cond["c_crossattn"]=[context]
        noise_pred=self.model.apply_model(latents_input,t,cond)
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents,index)
        else:
            latents = self.prev_step(noise_pred, t, latents,index)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        # latents = 1 / 0.18215 * latents.detach()
        # image = self.model.vae.decode(latents)['sample']
        # if return_type == 'np':
        #     image = (image / 2 + 0.5).clamp(0, 1)
        #     image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        #     image = (image * 255).astype(np.uint8)
        image=self.model.decode_first_stage(latents)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            # if type(image) is Image:
            #     image = np.array(image)
            # if type(image) is torch.Tensor and image.dim() == 4:
            #     latents = image
            # else:
            #     image = torch.from_numpy(image).float() / 127.5 - 1
            #     image = image.permute(2, 0, 1).unsqueeze(0).to(device)
            #     latents = self.model.vae.encode(image)['latent_dist'].mean
            #     latents = latents * 0.18215
            n_samples=1
            latents=self.model.encode_first_stage((image.to(device))).mode().detach()\
                               .repeat(n_samples, 1, 1, 1)
            self.latents_image=latents
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt


    ## add for zero123
    @torch.no_grad()
    def init_prompt_mine(self, input_im):
        n_samples=1
        c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
        u_c=model.get_learned_conditioning(torch.zeros(input_im.shape).to(device)).tile(n_samples, 1, 1)
        T = torch.tensor([0, 0, 0, 0])
        T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
        c = torch.cat([c, T], dim=-1)
        c = model.cc_projection(c)
        self.prompt=c
        self.context=torch.cat([u_c,c])









    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        # sampler = DDIMSampler(self.model)
        # timesteps=[981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721,
        # 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441,
        # 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161,
        # 141, 121, 101,  81,  61,  41,  21,   1]
        # print(self.model.num_timesteps)
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            t=torch.full((1,), t, device=device, dtype=torch.long)
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent,len(self.model.scheduler.timesteps) - i - 1)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_c=torch.zeros(latents[0].shape).cuda()
        uncond_embeddings_list = []
        uncond_embeddings_c_list=[]
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings_c=uncond_embeddings_c.clone().detach()
            uncond_embeddings.requires_grad = True
            uncond_embeddings_c.requires_grad = True
            optimizer = Adam([uncond_embeddings,uncond_embeddings_c], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            t=torch.full((1,), t, device=device, dtype=torch.long)
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, cond_embeddings, True, uncond_embeddings,uncond_embeddings_c)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur,i)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            uncond_embeddings_c_list.append(uncond_embeddings_c[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                # print(context)
                latent_cur = self.get_noise_pred(latent_cur, t, False, context,i)
        bar.close()
        return uncond_embeddings_list,uncond_embeddings_c_list
    
    def invert(self, image_path: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        #add for zero123
        # print(image_gt)
        # print(image_gt.shape)
        image_gt=transforms.ToTensor()(image_gt).unsqueeze(0).to(device)
        image_gt = image_gt * 2 - 1
        image_gt = transforms.functional.resize(image_gt, [256, 256])
        self.init_prompt_mine(image_gt)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings,uncond_embeddings_c = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings,uncond_embeddings_c
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        self.model = model
        # self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.ddpm_num_timesteps=model.num_timesteps
        self.prompt = None
        # self.context = None
        self.latents_image=None
        self.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose=True)



@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt,
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 3.0,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    uncond_embeddings_c=None,
    start_time=50,
    return_type='image'
):
    batch_size = 1
    ptp_utils.register_attention_control(model, controller)
    height = width = 256
    
    # text_input = model.tokenizer(
    #     prompt,
    #     padding="max_length",
    #     max_length=model.tokenizer.model_max_length,
    #     truncation=True,
    #     return_tensors="pt",
    # )
    # text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    # max_length = text_input.input_ids.shape[-1]
    #add for zero123
    # print(image_gt)
    # print(image_gt.shape)

#需要修改
    n_samples=1
    c = model.get_learned_conditioning(prompt).tile(n_samples, 1, 1)
    T = torch.tensor([0, 0, 0, 0])
    T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
    c = torch.cat([c, T], dim=-1)
    c = model.cc_projection(c)


    # if uncond_embeddings is None:
    #     uncond_input = model.tokenizer(
    #         [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    #     )
    #     uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    # else:
    #     uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    n_samples=1
    latents_c=model.encode_first_stage((prompt.to(device))).mode().detach()\
                        .repeat(n_samples, 1, 1, 1)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        # if uncond_embeddings_ is None:
        #     context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        # else:
        #     context = torch.cat([uncond_embeddings_, text_embeddings])
        cond={}
        cond["c_concat"]=[torch.cat([uncond_embeddings_c[i],latents_c])]
        cond["c_crossattn"]=[torch.cat([uncond_embeddings[i],c])]

        latents = ptp_utils.diffusion_step(model, controller, latents, cond, t, guidance_scale, low_resource=False)
        
    if return_type == 'image':
        image = model.decode_first_stage(latents)
        image=torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        image=image.cpu().permute(0, 2, 3, 1).numpy()
        image=(image * 255).astype(np.uint8)
    else:
        image = latents
    return image, latent



def run_and_display(prompts=None, controller=None, latent=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True,uncond_embeddings_c=None,prefix='inversion'):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(model, prompts, controller, latent=latent, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings,uncond_embeddings_c=uncond_embeddings_c)
    if verbose:
        ptp_utils.view_images(images, prefix=prefix)
    return images, x_t







def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    print(config.model)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model




if __name__ == '__main__':
    # Load Stable Diffusion
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = 'your token'
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = 50
    GUIDANCE_SCALE = 3.0
    MAX_NUM_WORDS = 77
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN, scheduler=scheduler).to(device）
    # try:
    #     ldm_stable.disable_xformers_memory_efficient_attention()
    # except AttributeError:
    #     print("Attribute disable_xformers_memory_efficient_attention() is missing")
    # tokenizer = ldm_stable.tokenizer

    config='/home/chenyangsen/kinneyyang/Adversarial_Content_Attack/zero123_main/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml'
    ckpt='/home/chenyangsen/kinneyyang/threestudio-main/load/zero123/zero123-xl.ckpt'
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)
    # print(model)
    model.scheduler=scheduler
    null_inversion = NullInversion(model)


    # Batch Images Load
    image_nums = 1
    # all_prompts = open('temp_2/1/prompts.txt').readlines()
    all_latents = torch.zeros(image_nums, 4, 32, 32)
    all_uncons = torch.zeros(image_nums, NUM_DDIM_STEPS, 77, 768)

    img_filepath = '/home/chenyangsen/kinneyyang/Adversarial_Content_Attack/third_party/Natural-Color-Fool/dataset/ast'
    filepath_list = os.listdir(img_filepath)
    avg_ssim, avg_mse, avg_psnr = 0, 0, 0
    for i in trange(image_nums):
        img_path = os.path.join(img_filepath, filepath_list[i])
        idx = int(filepath_list[i].split('.')[0]) - 1
        print(img_path, filepath_list[i], idx)
        raw_image = Image.open(img_path).convert("RGB")
        # prompts = [all_prompts[idx].strip()]
        # print(prompts)

        start = time.time()
        # Image Inversion
        (image_gt, image_enc), x_t, uncond_embeddings,uncond_embeddings_c= null_inversion.invert(img_path, offsets=(0,0,0,0), verbose=True)
        print('Inversion Time:', time.time() - start)
        print(x_t.shape)
        print(len(uncond_embeddings), uncond_embeddings[0].shape)

        all_latents[idx] = x_t
        for k in range(NUM_DDIM_STEPS):
            all_uncons[idx][k] = uncond_embeddings[k]

        controller = AttentionStore()
        # uncond_embeddings=torch.zeros(uncond_embeddings.shape).cuda()
        image_inv, x_t = run_and_display(image_gt, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings, verbose=False,uncond_embeddings_c=uncond_embeddings_c)
        image_gt=torch.clamp((image_gt + 1.0) / 2.0, min=0.0, max=1.0)
        image_gt=image_gt.cpu().permute(0, 2, 3, 1).numpy()
        image_gt=(image_gt * 255).astype(np.uint8)
        print(image_gt.shape)
        print("showing from left to right: the ground truth image, the vq-autoencoder reconstruction, the null-text inverted image")
        ptp_utils.view_images([image_inv[0]], prefix='1/pair/%d' % (idx))
        ptp_utils.view_images([image_gt[0]], prefix='1/original/%d' % (idx))
        ptp_utils.view_images([image_inv[0]], prefix='1/inversion/%d' % (idx))


    # torch.save(all_latents, 'temp_2/1/all_latents.pth')
    # torch.save(all_uncons, 'temp_2/1/all_uncons.pth')
