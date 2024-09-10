import os

import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib.pyplot as plt

from utils import train_utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using {device}.')


def train(net, optimizer, input, target, criterion, gradient_clipping, **kwargs):
    train_horizon = kwargs.get("train_horizon", None)
    uncertainty_quantification = kwargs.get("uncertainty_quantification", None)
    optimizer.zero_grad(set_to_none=True)        
    if train_horizon is None:
        out = net(input.float())    
        loss = criterion(out, target)
    else:
        # Multi step loss
        out = net(input.float())
        multiloss = criterion(out, target[:,:,0])
        for step in range(train_horizon-1):
            if uncertainty_quantification.startswith('scoring-rule'):
                out = out.mean(axis = -1)
            out = net(out)
            multiloss += criterion(out, target[:,:,step])
        loss = multiloss/train_horizon

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

def trainer(gpu_id, train_loader, val_loader, directory, training_parameters, data_parameters, logging, filename_ending,
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
    criterion = train_utils.get_criterion(training_parameters, domain_range, d, device)
    
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
    
    logging.info(f'Memory allocated: {torch.cuda.memory_reserved(device=device)}')
    
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

    # Additional parameters
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    if training_parameters["model"] == "SFNO":
        train_horizon = data_parameters["train_horizon"]
    else:
        train_horizon = 1

    t = train_horizon
    # Start training loop
    logging.info(f'Training starts now.')
    for epoch in range(training_parameters['n_epochs']):        
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
            if training_parameters["model"] == "SFNO":
                batch_loss, batch_grad_norm = train(model, optimizer, input, target, criterion,training_parameters['gradient_clipping'],
                                                     train_horizon = data_parameters["train_horizon"], 
                                                     uncertainty_quantification = uncertainty_quantification)
            else:
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
            if not uncertainty_quantification.endswith('dropout'):
                model.eval()
            
            validation_loss = 0
            with torch.no_grad():
                for input, target in val_loader:
                    input = input.to(device)
                    target = target.to(device)
                    if train_horizon == 1:
                        out = model(input)
                        validation_loss += criterion(out, target).item()
                    else:
                        out = model(input)
                        validation_loss += criterion(out, target[:,:,0]).item() / t
                        for step in range(t-1):
                            print(f"Training step {t}")
                            if uncertainty_quantification.startswith('scoring-rule'):
                                out = out.mean(axis = -1)
                            out = model(out)
                            validation_loss += criterion(out, target[:,:,step]) / t

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