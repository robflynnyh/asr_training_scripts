import torch, torch_optimizer
from torch.optim.lr_scheduler import CyclicLR
from speachy.utils.general.model_utils import load_schedular_data
from ranger.ranger2020 import Ranger # https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer/blob/master/ranger/ranger2020.py

def optimizer(model, args):
    implemented_optimizers = ['adamw', 'adam', 'madgrad', 'novograd', 'ranger']
    weight_decay = 1e-6 if 'weight_decay' not in args else args.weight_decay
    if args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), weight_decay=weight_decay, lr=args.min_lr)
    elif args.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), weight_decay=weight_decay, lr=args.min_lr)
    elif args.optimizer_type == 'madgrad':
        optimizer = torch_optimizer.MADGRAD(model.parameters(), lr=args.min_lr, momentum=0.9, weight_decay=weight_decay, eps=1e-6)
    elif args.optimizer_type == 'novograd':
        optimizer = torch_optimizer.NovoGrad(model.parameters(), lr= args.min_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_averaging=False, amsgrad=False,)
    elif args.optimizer_type == 'ranger':
        optimizer = Ranger(model.parameters(), lr = args.min_lr, weight_decay = weight_decay, use_gc = True)
    else:
        raise ValueError(f'Unknown optimizer type: {args.optimizer_type}, implemented optimizers: {implemented_optimizers}')

    schedular = CyclicLR(optimizer, base_lr=args.min_lr, max_lr=args.max_lr, step_size_up=args.step_size, step_size_down=args.step_size*4, mode='triangular', cycle_momentum=False)

    return optimizer, schedular


def update_schedular(args, optim, schedular):
    if schedular is None:
        return None
    max_lr, min_lr, step_size = load_schedular_data(args)
    if max_lr != args.max_lr or min_lr != args.min_lr or step_size != args.step_size:
        print('Updating schedular')
        args.max_lr = max_lr
        args.min_lr = min_lr
        args.step_size = step_size
        schedular = CyclicLR(optim, base_lr=args.min_lr, max_lr=args.max_lr, step_size_up=args.step_size, step_size_down=args.step_size*2, mode='triangular', cycle_momentum=False)
    return schedular