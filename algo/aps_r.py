import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np


class APS_R(Algo):
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 num_candidates=8,              # Number of candidates to select from
                 threshold=0.25,                # Apply guidance after int(threshold * num_steps)
                 batch_size=8):                 # Batch size for loss computation
        super(APS_R, self).__init__(net, forward_op)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.num_candidates = num_candidates
        self.batch_size = batch_size
        self.threshold = threshold
        assert self.num_candidates % self.batch_size == 0, 'Number of candidates should be divisible by batch size.'

    @torch.no_grad()
    def inference(self, observation, num_samples=1, verbose=False):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max   
        num_batches = self.num_candidates // self.batch_size

        num_steps = self.scheduler.num_steps

    
        pbar = tqdm(range(num_steps))
        x_results = torch.empty(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device)
        
        for j in range(num_samples):
            x_next = x_initial[j:j+1]
            for i in pbar:
                if  i/num_steps<0.20 and i/num_steps>0.001:
                    rt=5
                else:
                    rt=1
                for r in range(rt):
                    x_cur = x_next
                    sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
                    denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))
                    score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
                    if i < int(num_steps * self.threshold):
                        x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * torch.randn_like(x_cur)
                    elif i > int(num_steps * self.threshold) and i < num_steps - 1:
                        # sample possible next steps
                        sigma_next = self.scheduler.sigma_steps[i+1]
                        epsilon = torch.randn(self.num_candidates, *x_cur.shape[1:], device=device)
                        x_candidates = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon # (num_candidates, ...)
                        N, *spatial = x_candidates.shape

                        x0s = torch.zeros_like(x_candidates)
                        for k in range(num_batches):
                            x_batch = x_candidates[k*self.batch_size:(k+1)*self.batch_size]
                            denoised_batch = self.net(x_batch / self.scheduler.scaling_steps[i+1], torch.as_tensor(sigma_next).to(x_cur.device))

                            x0s[k*self.batch_size:(k+1)*self.batch_size]  = denoised_batch
                        ys = self.forward_op.forward(x0s)
                        ys_diff = ys - ys.mean(dim=0, keepdim=True)
                
                        ys_err = torch.mean(0.5 * self.forward_op.gradient_m(ys, observation),dim=0,keepdim=True)
                
                        coef = (
                            torch.matmul(
                                ys_err.reshape(ys_err.shape[0], -1),
                                ys_diff.reshape(ys_diff.shape[0], -1).T,
                            )
                            / x_candidates.shape[0]
                        )
                
                        direc = (x_candidates - x_cur * scaling_factor - factor * score)/np.sqrt(factor)
            
                        g = coef @ direc.reshape(N, -1)

                        d = x_cur.numel()/x_cur.shape[0]

                        if r==rt-1:
                            x_next = x_cur * scaling_factor + factor * score -  torch.nn.functional.normalize(g.reshape(1, *spatial),dim=(1, 2, 3))*np.sqrt(factor)*(d**0.5)
                            denoised_batch = self.net(x_next / self.scheduler.scaling_steps[i+1], torch.as_tensor(sigma_next).to(x_cur.device))
                            loss_scale = self.forward_op.loss(denoised_batch, observation)[0]
                            pbar.set_description(f'Iteration {(i+1) + 1}/{num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
                        else:
                            x_next = x_cur * scaling_factor + factor * score -  torch.nn.functional.normalize(g.reshape(1, *spatial),dim=(1, 2, 3))*np.sqrt(factor)*(d**0.5)
                            sigma_next = self.scheduler.sigma_steps[i+1]
                            
                            x_next = self.scheduler.scaling_steps[i]*((x_next/self.scheduler.scaling_steps[i+1]) + np.sqrt(sigma ** 2 - sigma_next ** 2) * torch.randn_like(x_next))
                    else:
                        x_next = denoised
            x_results[j] = x_next
        return x_results



# ----------- deterministic sampler ------------#
# Generate x_0 from x_t for any t.


@torch.no_grad()
def ode_sampler(
    net,
    x_initial,
    num_steps=18,
    sigma_start=80.0,
    sigma_eps=0.002,
    rho=7,
):
    if num_steps == 1:
        denoised = net(x_initial, sigma_start)
        return denoised
    last_sigma = sigma_eps
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=x_initial.device)

    t_steps = (
        sigma_start ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (last_sigma ** (1 / rho) - sigma_start ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # Main sampling loop.
    x_next = x_initial
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        t_hat = t_cur
        x_hat = x_cur

        # Euler step.
        denoised = net(x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

    return x_next