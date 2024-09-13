import os

import torch
from torch import optim
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
        for step in range(1,train_horizon):
            if uncertainty_quantification.startswith('scoring-rule'):
                out = out.mean(axis = -1)
            out = net(out)
            multiloss += criterion(out, target[:,:,step])
        loss = multiloss/train_horizon

    loss.backward()
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

def trainer(train_loader, val_loader, directory, training_parameters, data_parameters, logging, filename_ending,
            domain_range, d_time, results_dict, world_size=None):   
    
    if device == 'cpu':
        assert not training_parameters['data_loader_pin_memory']
    
    d = len(next(iter(train_loader))[0].shape) - 2
    criterion = train_utils.get_criterion(training_parameters, domain_range, d, device)
    
    in_channels = next(iter(train_loader))[0].shape[1]
    out_channels = next(iter(train_loader))[1].shape[1]
    
    model = train_utils.setup_model(training_parameters, device, in_channels, out_channels)
       
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    # Additional parameters
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    if training_parameters["model"] == "SFNO":
        train_horizon = data_parameters["train_horizon"]
        autoregressive = True
    else:
        train_horizon = 1
        autoregressive = False

    # Iterate over autoregressive steps, if necessary
    for t in range(1,train_horizon+1):
        if t == 1:
            logging.info(f'Training starts now.')
            prev_epochs = 0
        else:
            logging.info(f'Training for autoregressive step {t} starts now.')
            # Save epoch state
            prev_epochs = epoch+1
            # Reset parameters
            best_loss = torch.inf
            early_stopper.counter = 0
            early_stopper.min_validation_loss = float("inf")
            scheduler.step()
            logging.info(f'Learning rate reduced to: {scheduler.get_last_lr()}')

        for epoch in range(prev_epochs, training_parameters['n_epochs']+ prev_epochs):              
            model.train()
            for input, target in train_loader:
                input = input.to(device)
                target = target.to(device)
                if training_parameters["model"] == "SFNO":
                    batch_loss, batch_grad_norm = train(model, optimizer, input, target, criterion,training_parameters['gradient_clipping'],
                                                        train_horizon = t, uncertainty_quantification = uncertainty_quantification)
                else:
                    batch_loss, batch_grad_norm = train(model, optimizer, input, target, criterion, training_parameters['gradient_clipping'])
                running_loss += batch_loss
                grad_norm += batch_grad_norm
                        
            if lr_schedule == 'step' and early_stopper.counter > 5:
                # stepwise scheduler only happens once per epoch and only if the validation has not been going down for at least 10 epochs
                scheduler.step()
                logging.info(f'Learning rate reduced to: {scheduler.get_last_lr()}')
                       
            if epoch % report_every == report_every - 1:
                epochs.append(epoch)
                if not uncertainty_quantification.endswith('dropout'):
                    model.eval()
                
                validation_loss = 0
                with torch.no_grad():
                    for input, target in val_loader:
                        input = input.to(device)
                        target = target.to(device)
                        if not autoregressive:
                            out = model(input)
                            validation_loss += criterion(out, target).item()
                        else:
                            out = model(input)
                            validation_loss += criterion(out, target[:,:,0]).item() / t
                            for step in range(1,t):
                                if uncertainty_quantification.startswith('scoring-rule'):
                                    out = out.mean(axis = -1)
                                out = model(out)
                                validation_loss += criterion(out, target[:,:,step]).item() / t

                validation_loss_list.append(validation_loss / report_every / len(val_loader))
                training_loss_list.append(running_loss / report_every / (len(train_loader)))
                grad_norm_list.append(grad_norm / report_every / (len(train_loader)))
                running_loss = 0.0
                grad_norm = 0
                
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    filename = os.path.join(directory, f'Datetime_{d_time}_Loss_{filename_ending}.pt')
                    train_utils.checkpoint(model, filename)

                # Early stopping
                if training_parameters['early_stopping'] and epoch > 50:
                    if early_stopper.early_stop(validation_loss):
                        logging.info(f'EP {epoch}: Early stopping')    
                        break
                        
            if epoch > report_every - 2:
                logging.info(f'[{epoch + 1:5d}] Training loss: {training_loss_list[-1]:.8f}, Validation loss: '
                            f'{validation_loss_list[-1]:.8f}, Gradient norm: {grad_norm_list[-1]:.8f}')        

    
    optimizer.zero_grad(set_to_none=True)
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
    
    train_utils.checkpoint(model, filename)
        
    return model, filename