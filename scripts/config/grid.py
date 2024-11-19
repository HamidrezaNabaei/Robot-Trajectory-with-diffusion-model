import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ## value kwargs
    ('discount', 'd'),
]

logbase = 'logs'

base = {
    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 11,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.GridWorldRenderer',
        'attention': True,

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 10000,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 5000,
        'loss_type': 'l2',
        'n_train_steps':1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 2000,
        'sample_freq': 2000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 100,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'values': {
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.GridWorldRenderer',
        'attention': True,

        ## value-specific kwargs
        'discount': 0.99,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 10000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 5000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 30,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
    },

    'plan': {
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 100,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,
        'attention': True,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.99,

        ## loading
        'diffusion_loadpath': 'f:diffusion/defaults_H{horizon}_T{n_diffusion_steps}',
        'value_loadpath': 'f:values/defaults_H{horizon}_T{n_diffusion_steps}_d{discount}',

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': True,
        'suffix': '0',
    },
}

#------------------------ overrides ------------------------#

gridworld_medium_expert_v2_old = {
    'diffusion': {
        **base['diffusion'],
        'horizon': 8,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}

random_obstacle_v2 = {
    'diffusion': {
        **base['diffusion'],
        'horizon': 8,
        'dim_mults': (1, 4, 8),
    },
    'values': {
        **base['values'],
        'horizon': 8,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        **base['plan'],
        'horizon': 8,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}

