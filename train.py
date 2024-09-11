import os

import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

from utils import train_utils

import resource
import psutil
import copy

def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    # you can convert that object to a dictionary 
    return f'{point}: mem (CPU python)={usage[2]/1024.0}MB; mem (CPU total)={dict(psutil.virtual_memory()._asdict())["used"] / 1024**2}MB'


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using {device}.')


def train(net, optimizer, input, target, criterion, gradient_clipping):
    optimizer.zero_grad(set_to_none=True)
        
    out = net(input.float())
    
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()    
    gradient_norm = 0
    for p in net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        gradient_norm += param_norm.item() ** 2
    gradient_norm = gradient_norm ** 0.5
    
    torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clipping)
    
    gradient_norm_test = 0
    for p in net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        gradient_norm_test += param_norm.item() ** 2
    gradient_norm_test = gradient_norm_test ** 0.5
    
    
    assert gradient_norm_test < 1.5 * gradient_clipping

    optimizer.step()

    return loss.item(), gradient_norm

def trainer(gpu_id, train_loader, val_loader, directory, training_parameters, logging, filename_ending,
            domain_range, d_time, results_dict, world_size=None):
    
    if training_parameters['distributed_training']:
        train_utils.ddp_setup(rank=gpu_id, world_size=world_size)
        print(f'GPU ID: {gpu_id}')
        # if gpu_id==0:
        if gpu_id>-1:
            logging.basicConfig(filename=os.path.join(directory, f'experiment_{gpu_id}.log'), level=logging.INFO)
            logging.info('Starting the logger in the training process.')
            print('Starting the logger in the training process.')
    
        flag_tensor = torch.zeros(1).to(f'cuda:{gpu_id}')
    
    if device == 'cpu':
        assert not training_parameters['data_loader_pin_memory']
    
    d = len(next(iter(train_loader))[0].shape) - 2
    criterion = train_utils.get_criterion(training_parameters, domain_range, d)
    
    in_channels = next(iter(train_loader))[0].shape[1]
    out_channels = next(iter(train_loader))[1].shape[1]
    
    model = train_utils.setup_model(training_parameters, device, in_channels, out_channels)
        
    if training_parameters['distributed_training']:
        model = DDP(model, device_ids=[gpu_id])
    
    if training_parameters['init'] != 'default':
        train_utils.initialize_weights(model, training_parameters['init'])

    n_parameters = 0
    for parameter in model.parameters():
        n_parameters += parameter.nelement()

    train_utils.log_and_save_evaluation(n_parameters, 'NumberParameters', results_dict, logging)
    
    logging.info(f'GPU memory allocated: {torch.cuda.memory_reserved(device=device)}')
    logging.info(using('After setting up the model'))
    
    # create your optimizer
    if training_parameters['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=training_parameters['learning_rate'], betas=(0.9, 0.999), weight_decay=training_parameters['weight_decay'])
    elif training_parameters['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=training_parameters['learning_rate'])
    
    report_every = 1
    early_stopper = train_utils.EarlyStopper(patience=int(training_parameters['early_stopping'] / report_every), min_delta=0.0001)
    running_loss = 0
    grad_norm = 0
    
    training_loss_list = []
    validation_loss_list = []
    grad_norm_list = []
    epochs = []
            
    best_loss = torch.inf
        
    lr_schedule = training_parameters['lr_schedule']
    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    logging.info(f'Training starts now.')

    for epoch in range(training_parameters['n_epochs']):

        logging.info(using('At the start of the epoch'))
        
        if training_parameters['distributed_training']:
            dist.all_reduce(flag_tensor,op=dist.ReduceOp.SUM)
            if flag_tensor == 1:
                logging.info("Training stopped")
                break
            train_loader.sampler.set_epoch(epoch)
            
        model.train()

        for input, target in train_loader:
            input = input.to(device)
            target = target.to(device)
            batch_loss, batch_grad_norm = train(model, optimizer, input, target, criterion, training_parameters['gradient_clipping'])
            running_loss += batch_loss
            grad_norm += batch_grad_norm
                    
        if lr_schedule == 'step' and early_stopper.counter > 10:
            # stepwise scheduler only happens once per epoch and only if the validation has not been going down for at least 10 epochs
            scheduler.step(scheduler, optimizer, epoch)
        
        # The none-main processes do not have to report anything
        if training_parameters['distributed_training'] and gpu_id != 0:
            continue
        
        if epoch % report_every == report_every - 1:
            epochs.append(epoch)
            if not training_parameters['uncertainty_quantification'].endswith('dropout'):
                model.eval()
            
            validation_loss = 0
            with torch.no_grad():
                for input, target in val_loader:
                    input = input.to(device)
                    target = target.to(device)
                    output_target = model(input)
                    validation_loss += criterion(output_target, target).item()

            validation_loss_list.append(validation_loss / report_every / len(val_loader))
            training_loss_list.append(running_loss / report_every / (len(train_loader)))
            grad_norm_list.append(grad_norm / report_every / (len(train_loader)))
            running_loss = 0.0
            grad_norm = 0
            
            if validation_loss < best_loss:
                best_loss = validation_loss
                filename = os.path.join(directory, f'Datetime_{d_time}_Loss_{filename_ending}.pt')
                
                if training_parameters['distributed_training']:
                    train_utils.checkpoint(model.module, filename)
                else:
                    train_utils.checkpoint(model, filename)

            # Early stopping
            if training_parameters['early_stopping'] and epoch > 50:
                if early_stopper.early_stop(validation_loss):
                    logging.info(f'EP {epoch}: Early stopping')
                    
                    if training_parameters['distributed_training']:
                        flag_tensor += 1
                    else:
                        break
        if epoch > report_every - 2:
            logging.info(f'[{epoch + 1:5d}] Training loss: {training_loss_list[-1]:.8f}, Validation loss: '
                         f'{validation_loss_list[-1]:.8f}, Gradient norm: {grad_norm_list[-1]:.8f}')

    logging.info(using('After finishing all epochs'))

    # only one GPU has to report everything
    if training_parameters['distributed_training'] and gpu_id != 0:
        return model
    
    optimizer.zero_grad(set_to_none=True)
    if training_parameters['distributed_training']:
        train_utils.resume(model.module, filename)
    else:
        train_utils.resume(model, filename)
    
    plt.plot(epochs, training_loss_list, label='training loss')
    plt.plot(epochs, validation_loss_list, label='validation loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'Datetime_{d_time}_Loss_{filename_ending}.png'))
    plt.plot(epochs, grad_norm_list, label='gradient norm')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'Datetime_{d_time}_analytics_{filename_ending}.png'))
    plt.close()

    if training_parameters['distributed_training']:
        model = model.module
    
    train_utils.checkpoint(model, filename)
    
    if training_parameters['distributed_training']:
        dist.destroy_process_group()
        
    return model, filename