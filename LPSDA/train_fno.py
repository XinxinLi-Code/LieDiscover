# modified from https://github.com/kogyeonghoon/learning-symmetry-from-scratch/blob/main/pde/train_fno.py

import argparse
import os
import random
import torch
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader

if __package__:
    from .utils import *
    from .fno_train_helper import *
    from .ode_transform import no_transform, transform
    from .deltamodel import VectorFieldOperation
    from .fno import FNO1d
    from .coordinates import bind_coordinates
    from .generators import build_generator, configure_sources, add_source_arguments, record_inputs, record_augmentation
else:
    from utils import *
    from fno_train_helper import *
    from ode_transform import no_transform, transform
    from deltamodel import VectorFieldOperation
    from fno import FNO1d
    from coordinates import bind_coordinates
    from generators import build_generator, configure_sources, add_source_arguments, record_inputs, record_augmentation

def arg_parse():
    parser = argparse.ArgumentParser(description='Train a PDE solver')
    # PDE arguments
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--pde', type=str, default='burger', choices=('burger', 'heat', 'kdv'))
    parser.add_argument('--train_samples', type=int, default=10, help='Number of training trajectories')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')

    # Model parameters
    parser.add_argument('--n_delta', type=int, default=None, help='Number of delta transformations')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--transform_batch_size', type=int, default=32, help='Batch size for transforms')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='Batch size for test')
    parser.add_argument('--delta_exp', type=str, default='lps_liediscover', choices=('lps_liediscover',), help='Use discovered generators')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.4, help='LR decay factor')
    parser.add_argument('--sigma', nargs='+', type=float, default=[0.1], help='Sigma values')
    parser.add_argument('--n_transform', type=int, default=16, help='Number of transformations')
    parser.add_argument('--test_mode', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Test mode')
    parser.add_argument('--p_original', type=float, default=0.5, help='Probability of using original data')
    parser.add_argument('--early_stopping', type=int, default=0, help='Early stopping epochs')
    parser.add_argument('--n_iters', type=int, default=None, help='Number of iterations per epoch')
    parser.add_argument('--scheduler', type=str, default='step', help='LR scheduler: "step" or "plateau"')
    parser.add_argument('--patience', type=int, default=30, help='Patience for plateau scheduler')
    parser.add_argument('--split', type=int, default=4, help='LR split')
    parser.add_argument('--resample', type=str, default='diric', help='Resampling method')

    # Misc
    parser.add_argument('--time_history', type=int, default=20, help='Time steps for input')
    parser.add_argument('--time_future', type=int, default=20, help='Time steps for output')
    parser.add_argument('--print_interval', type=int, default=70, help='Print interval')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sample', type=int, default=0)

    add_source_arguments(parser)
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    configure_sources(args)
    return args

def init_experiment(args):
    experiment_name = os.path.join(args.output_root, f'exp_fno_{args.pde}')
    exp_path, outfile = init_path(experiment_name, args.exp_name, args)
    print_out = Writer(outfile)
    print_out(f"Experiment start at: {datetime.now()}")
    return exp_path, print_out

def load_data(args):
    train_dataset = PDEDataset(path=args.data_dir, pde_name=args.pde, mode='train')
    if not 1 <= args.train_samples <= len(train_dataset):
        raise ValueError(f'train_samples must be between 1 and {len(train_dataset)}; the default is 10.')
    val_dataset = PDEDataset(path=args.data_dir, pde_name=args.pde, mode='val')
    test_dataset = PDEDataset(path=args.data_dir, pde_name=args.pde, mode='test')
    bind_coordinates(args, train_dataset, val_dataset, test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.transform_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def init_delta_model(args):
    # Every explicit method uses the same official normalization and rotation.
    return build_generator(args, VectorFieldOperation())

def init_model_optimizer(args):
    data_creator = DataCreator(time_history=args.time_history,
                               time_future=args.time_future).to(args.device)

    model = FNO1d(time_history=args.time_history,
                  time_future=args.time_future, modes=6).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.scheduler == 'step':
        milestones = np.arange(args.split) * int(args.num_epochs / args.split)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.4, patience=args.patience, min_lr=1e-6)
    else:
        scheduler = None

    return model, data_creator, optimizer, scheduler

def transform_data(train_loader, args, delta_model, print_out):
    data_transformed_list = []
    t_eff_list = []
    reject_list = []

    for i in range(args.n_transform):
        if i == 0:
            print_out(f"Applying no_transform ({i+1}/{args.n_transform})...")
            data_transformed, t_eff, reject = no_transform(train_loader)
            print_out("no_transform completed.")
        else:
            print_out(f"Applying transform {i+1}/{args.n_transform}...")
            data_transformed, t_eff, reject = transform(
                train_loader,
                delta_model,
                sigma=torch.tensor(args.sigma).to(args.device),
                device=args.device,
                n_delta=args.n_delta,
                resample=args.resample
            )
            print_out(f"Transform {i+1} completed.")

        data_transformed_list.append(data_transformed)
        t_eff_list.append(t_eff)
        reject_list.append(reject)

    return data_transformed_list, t_eff_list, reject_list

def create_augmented_loader(data_transformed_list, t_eff_list, reject_list, args):
    data_transformed = torch.stack(data_transformed_list, dim=0)
    t_eff = torch.stack(t_eff_list, dim=0)
    reject = torch.stack(reject_list, dim=0)

    augmented_dataset = AugmentedDataset(data_transformed,
                                         t_eff,
                                         reject,
                                         args.train_samples,
                                         args.n_transform,
                                         p_original=args.p_original)

    augmented_loader = DataLoader(augmented_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
    return augmented_loader

def validate_model(args, model, val_loader, data_creator, criterion, print_out):
    model.eval()
    val_loss, _, _, _ = test(args, model, val_loader, data_creator, criterion, device=args.device, print_out=print_out)
    return val_loss

def train_loop(model, optimizer, scheduler, augmented_loader, val_loader, test_loader, args, data_creator, criterion, print_out):
    # Initialize metric tracker
    metric_names = ['train_loss','val_loss','test_loss','test_loss_std','ntest_loss','ntest_loss_std']
    metric_tracker = MetricTracker(metric_names)

    min_val_loss = float('inf')
    min_epoch = 0
    terminate = False
    current_lr = args.lr

    if args.test_mode:
        n_iters = 2
    elif args.n_iters is not None:
        n_iters = args.n_iters
    else:
        n_iters = next(iter(augmented_loader))[0].shape[1] * 2

    for epoch in range(args.num_epochs):
        print_out(f"Epoch {epoch}")
        model.train()

        # Training loop
        train_losses = []
        for iteration in range(n_iters):
            iteration_losses = []
            for u, t_eff in augmented_loader:
                optimizer.zero_grad()
                # Determine start_time for each sample in the batch
                start_time_list = []
                for j in range(u.shape[0]):
                    t_min, t_max = t_eff[j, 0].item(), t_eff[j, 1].item()
                    start_range = int(t_min)
                    end_range = int(t_max - data_creator.time_history * 2)
                    start_time_list.append(random.choice(range(start_range, end_range)))
                
                x, y = data_creator.create_data(u, start_time_list)
                x, y = x.to(args.device), y.to(args.device)
                x = x.permute(0, 2, 1)  # [batch, space, time_in]
                
                pred = model(x)
                loss = criterion(pred.permute(0, 2, 1), y).sum()
                loss.backward()
                iteration_losses.append(loss.detach() / x.shape[0])
                optimizer.step()

            iteration_losses = torch.stack(iteration_losses)
            train_losses.append(torch.mean(iteration_losses))
            if (iteration % args.print_interval) == 0:
                print_out(f'Training Loss (progress: {iteration / n_iters:.2f}): {torch.mean(iteration_losses)}')

        # End of epoch training loss
        train_loss_epoch = torch.mean(torch.stack(train_losses))

        # Validation
        print_out("Evaluation on validation dataset:")
        val_loss = validate_model(args, model, val_loader, data_creator, criterion, print_out)
        
        # Check if this is the best validation loss so far
        if val_loss < min_val_loss:
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.exp_path, 'model.pt'))
            print_out("Saved model")
            min_val_loss = val_loss
            min_epoch = epoch

            # Compute test metrics only when validation is at its minimum
            print_out("Evaluation on test dataset (new best validation):")
            test_loss, test_loss_std, ntest_loss, ntest_loss_std = test(args, model, test_loader, data_creator, criterion, device=args.device, print_out=print_out)
            test_loss = test_loss.item()
            test_loss_std = test_loss_std.item()
            ntest_loss = ntest_loss.item()
            ntest_loss_std = ntest_loss_std.item()

        else:
            # If not improved, set test metrics to large values or no improvement
            test_loss, test_loss_std, ntest_loss, ntest_loss_std = 1e10, 1e10, 1e10, 1e10

            # Early stopping
            if args.early_stopping != 0:
                if args.scheduler == 'plateau':
                    new_lr = optimizer.param_groups[0]['lr']
                    if (epoch - min_epoch > args.early_stopping) and (new_lr <= 1e-6):
                        terminate = True
                        print_out("Early stopping")
                else:
                    if (epoch - min_epoch > args.early_stopping):
                        terminate = True
                        print_out("Early stopping")

        # Record metrics
        metric_tracker.update({
            'train_loss': train_loss_epoch.item(),
            'val_loss': val_loss.item(),
            'test_loss': test_loss,
            'test_loss_std': test_loss_std,
            'ntest_loss': ntest_loss,
            'ntest_loss_std': ntest_loss_std,
        })
        metric_tracker.aggregate()
        metric_tracker.to_pandas().to_csv(os.path.join(args.exp_path, 'metrics.csv'))

        # Scheduler step
        if args.scheduler == 'plateau':
            scheduler.step(val_loss.item())
        elif args.scheduler == 'step':
            scheduler.step()

        print_out(f"current time: {datetime.now()}")

        if terminate or epoch == args.num_epochs - 1:
            # Load the best model and do final test evaluation
            model.load_state_dict(torch.load(os.path.join(args.exp_path, 'model.pt'), map_location=args.device, weights_only=True))
            print_out("Final evaluation on test dataset:")
            test_loss, test_loss_std, ntest_loss, ntest_loss_std = test(args, model, test_loader, data_creator, criterion, device=args.device, print_out=print_out)
            test_loss = test_loss.item()
            test_loss_std = test_loss_std.item()
            ntest_loss = ntest_loss.item()
            ntest_loss_std = ntest_loss_std.item()

            # Update metrics at the end as well
            # (Here we record the final loaded best model metrics)
            metric_tracker.update({
                'train_loss': train_loss_epoch.item(),
                'val_loss': val_loss.item(),
                'test_loss': test_loss,
                'test_loss_std': test_loss_std,
                'ntest_loss': ntest_loss,
                'ntest_loss_std': ntest_loss_std,
            })
            metric_tracker.aggregate()
            metric_tracker.to_pandas().to_csv(os.path.join(args.exp_path, 'metrics.csv'))

            break

    print_out(f"Experiment end at: {datetime.now()}")


if __name__ == "__main__":
    args = arg_parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.exp_name = default_experiment_name(args) if args.exp_name is None else args.exp_name

    train_loader, val_loader, test_loader = load_data(args)
    exp_path, print_out = init_experiment(args)
    args.exp_path = exp_path
    record_inputs(args, train_loader, val_loader, test_loader)

    delta_model = init_delta_model(args)
    model, data_creator, optimizer, scheduler = init_model_optimizer(args)

    data_transformed_list, t_eff_list, reject_list = transform_data(train_loader, args, delta_model, print_out)

    augmented_loader = create_augmented_loader(data_transformed_list, t_eff_list, reject_list, args)
    record_augmentation(args, reject_list, print_out)

    # Store exp_path in args for saving model
    args.exp_path = exp_path

    criterion = nn.MSELoss(reduction="none")

    train_loop(model, optimizer, scheduler, augmented_loader, val_loader, test_loader, args, data_creator, criterion, print_out)
