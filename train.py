import os

import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

import matplotlib.pyplot as plt

from utils import training_utils

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using {device}.')


def train(net, optimizer, input, criterion, target, i, balanced_loss, weight_active, gradient_clipping, data_parameters, one_hot, objective):
    optimizer.zero_grad(set_to_none=True)
    
    sigmoid = torch.nn.Sigmoid()
    
    logits = net(input, return_logits=True)
    
    logits_norm = logits.detach().cpu().data.norm(2)
    
    
    if objective == 'mask' or not one_hot:
        prediction = sigmoid(logits)
        if isinstance(criterion, torch.nn.BCELoss):
            if ((0 > prediction) | (1 < prediction)).any():
                print(f'Training: Input value outside [0,1] observed.')
            if prediction.isnan().any():
                print(f'Training: Input value nan observed.')
            if balanced_loss == True:
                criterion.weight = get_bce_weight(weight_active, target)  
        loss = criterion(prediction, target)
    else:
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = compute_ce(logits, target, data_parameters, criterion)
        else:
            prediction = utils.classification_layer(logits, data_parameters, sigmoid, one_hot)
            loss = criterion(prediction, target)

    loss.backward(retain_graph=False)
    
    
    gradient_norm = 0
    for p in net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        gradient_norm += param_norm.item() ** 2
    gradient_norm = gradient_norm ** 0.5
    
    # if gradient_norm > 2:
    #     logging.warning(f'Gradient norm: {gradient_norm}')
    #     if isinstance(criterion, torch.nn.CrossEntropyLoss):
    #         logging.warning(f'Logits       : {logits[0]}')
    #     else:
    #         logging.warning(f'Prediction   : {prediction[0]}')
    #     logging.warning(f'Target       : {target[0]}')
    
    # if gradient_norm < 0.00001:
    #     logging.warning(f'Gradient norm: {gradient_norm}')
    #     if isinstance(criterion, torch.nn.CrossEntropyLoss):
    #         logging.warning(f'Logits       : {logits[0]}')
    #     else:
    #         logging.warning(f'Prediction   : {prediction[0]}')
    #     logging.warning(f'Target       : {target[0]}')
            
    torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clipping)
    
    gradient_norm_test = 0
    for p in net.parameters():
        param_norm = p.grad.detach().data.norm(2)
        gradient_norm_test += param_norm.item() ** 2
    gradient_norm_test = gradient_norm_test ** 0.5
    
    
    assert gradient_norm_test < 1.5 * gradient_clipping
    
    # # For debugging purposes
    # old_params = []
    # for p in net.parameters():    
    #     old_params.append(copy.deepcopy(p).flatten())
    # old_params = torch.cat(old_params, dim=0)
    
    optimizer.step()
    
    # # For debugging purposes
    # new_params = []
    # for p in net.parameters():
    #     new_params.append(copy.deepcopy(p).flatten())
    # new_params = torch.cat(new_params, dim=0)
    
    # distance_in_param_space = torch.norm(new_params - old_params, p=2)
    
    return loss.item(), gradient_norm, logits_norm.item()


def trainer(gpu_id, train_loader, val_loader, directory, training_parameters, logging, filename,
            d_time, world_size=None):
    
    model_name = training_parameters['model']
        
    if training_parameters['distributed_training']:
        training_utils.ddp_setup(rank=gpu_id, world_size=world_size)
        print(f'GPU ID: {gpu_id}')
        # if gpu_id==0:
        if gpu_id>-1:
            logging.basicConfig(filename=os.path.join(directory, f'experiment_{gpu_id}.log'), level=logging.INFO)
            logging.info('Starting the logger in the training process.')
            print('Starting the logger in the training process.')
        # input_validation.to(f'cuda:{gpu_id}')
        # target_validation.to(f'cuda:{gpu_id}')
    
        # flag tensor for (early) stopping     
        flag_tensor = torch.zeros(1).to(f'cuda:{gpu_id}')
    
    if device == 'cpu':
        assert not training_parameters['data_loader_pin_memory']
    
    criterion = training_utils.get_criterion(training_parameters) # TODO
    
    net = training_utils.setup_nn(training_parameters, device)
    
    if training_parameters['distributed_training']:
        net = DDP(net, device_ids=[gpu_id])
    
    if training_parameters['init'] != 'default':
        training_utils.initialize_weights(net, training_parameters['init'])

    n_parameters = 0
    for parameter in net.parameters():
        n_parameters += parameter.nelement()
    logging.info(f'Number parameters: {n_parameters}')

    logging.info(f'Memory allocated: {torch.cuda.memory_reserved(device=device)}')
    
    # create your optimizer
    if training_parameters['optimizer'] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=training_parameters['learning_rate'], betas=(0.9, 0.999))
    elif training_parameters['optimizer'] == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=training_parameters['learning_rate'])
    
    report_every = 1
    early_stopper = training_utils.EarlyStopper(patience=int(training_parameters['early_stopping'] / report_every), min_delta=0.0001)
    running_loss = 0
    grad_norm = 0
    logits_norm = 0
    
    training_loss_list = []
    validation_loss_list = []
    grad_norm_list = []
    epochs = []
            
    best_loss = torch.inf
    
    sigmoid = torch.nn.Sigmoid() # Do we need that somewhere?
    
    lr_schedule = training_parameters['lr_schedule']
    if lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    logging.info(f'Training starts now.')

    for epoch in range(training_parameters['n_epochs']):
        
        if training_parameters['distributed_training']:
            dist.all_reduce(flag_tensor,op=dist.ReduceOp.SUM)
            if flag_tensor == 1:
                logging.info("Training stopped")
                break
            train_loader.sampler.set_epoch(epoch)
            
        net.train()

        for input, target in train_loader:
                        
            batch_loss, batch_grad_norm = train(net, optimizer, input, criterion, target, epoch, training_parameters['gradient_clipping'])
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
            net.eval()
            
            output_target = training_utils.predict_batchwise(net, input_validation, batch_size, num_samples_min, return_logits=True)

            if objective == 'mask' or not training_parameters['one_hot']:
                prediction = sigmoid(logits)
                if isinstance(criterion, torch.nn.BCELoss):
                    if ((0 > prediction) | (1 < prediction)).any():
                        print(f'Training: Input value outside [0,1] observed.')
                    if prediction.isnan().any():
                        print(f'Training: Input value nan observed.')
                    if training_parameters['balanced_loss'] == True:
                        criterion.weight = get_bce_weight(weight_active, target)  
                validation_loss = criterion(prediction, compute_ce(logits, target_validation, data_parameters, criterion))
            else:
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    validation_loss = compute_ce(logits, target_validation, data_parameters, criterion)
                    # do NOT execute before calling compute_ce, since it changes the logits:
                    prediction = utils.classification_layer(logits, data_parameters, sigmoid, training_parameters['one_hot']) 
                else:
                    prediction = utils.classification_layer(logits, data_parameters, sigmoid, training_parameters['one_hot'])
                    validation_loss = criterion(prediction, target_validation)

            validation_loss_list.append(validation_loss.cpu().detach().numpy())
            training_loss_list.append(running_loss / report_every / (len(train_loader)))
            grad_norm_list.append(grad_norm / report_every / (len(train_loader)))
            running_loss = 0.0
            grad_norm = 0
            
            if validation_loss < best_loss:
                best_loss = validation_loss
                filename = os.path.join(directory, f'Datetime_{d_time}_Loss_batch_size_' +
                                                   f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.pt')
                
                if training_parameters['distributed_training']:
                    training_utils.checkpoint(net.module, filename)
                else:
                    training_utils.checkpoint(net, filename)

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
        return net
    
    optimizer.zero_grad(set_to_none=True)
    if training_parameters['distributed_training']:
        training_utils.resume(net.module, filename)
    else:
        training_utils.resume(net, filename)
    
    plt.plot(epochs, training_loss_list, label='training loss')
    plt.plot(epochs, validation_loss_list, label='validation loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,
                             f'Datetime_{d_time}_Loss_batch_size_'
                             f'{training_parameters["batch_size"]}_hidden_dim_{training_parameters["hidden_dim"]}.png'))
    plt.plot(epochs, grad_norm_list, label='gradient norm')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(directory,
                             f'Datetime_{d_time}_analytics_{filename}.png'))
    plt.close()

    if training_parameters['distributed_training']:
        net = net.module
    
    torch.save(net, os.path.join(directory,
                                 f'Datetime_{d_time}_Loss.'
                                 f'pt'))
    
    if training_parameters['distributed_training']:
        dist.destroy_process_group()
        
    return net