import torch

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def get_criterion(training_parameters):
    if training_parameters['loss'] == 'MSE':
        criterion = nn.MSELoss()
    elif training_parameters['loss'] == 'MAE':
        criterion = nn.L1Loss()
    elif training_parameters['loss'] == 'BCE':
        criterion = nn.BCELoss()
    elif training_parameters['loss'] == 'CE':
        criterion = torch.nn.CrossEntropyLoss()
    elif training_parameters['loss'] == 'ASL':
        update = False
        if update:
            logging.info('Activated automatic update rule for ASL.')
        else:
            logging.info('Deactivated automatic update rule for ASL.')
        asl_obj = ASL(gamma_negative=training_parameters['gamma_neg'], p_target=training_parameters['p_target'],
                      update=True)
        criterion = asl_obj.asl
    else:
        raise NotImplementedError(
            f'Wrong loss specification: expected one out of "MSE", "BCE" and "ASL", '
            f'but received {training_parameters["loss"]}.')
    return criterion

def initialize_weights(model, init):
    for name, param in model.named_parameters():
        if 'weight' in name and param.data.dim() == 2:
            if init == 'xavier':
                nn.init.xavier_uniform(param)
            elif init == 'he':
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            else:
                raise NotImplementedError(f'Please choose init as "default", "xavier" or "he". You chose: {init}.')

def setup_model(training_parameters, data_parameters, input_validation, target_validation, device, objective):
    if training_parameters['resume_training']:
        net = torch.load(training_parameters['resume_training'], map_location=device)
        return net
    
    if training_parameters['model'] == 'mlp':
        assert data_parameters['num_samples_min'] == data_parameters['num_samples_max']
        net = utils.MLP(input_validation.shape[1] * input_validation.shape[2], output_size=target_validation.shape[1], h=training_parameters['hidden_dim'], 
                        objective=objective, data_parameters=data_parameters, one_hot=training_parameters['one_hot']).to(device)
    elif training_parameters['model'] == 'rnn':
        input_size = input_validation.shape[-1]
        net = utils.RNNModel(input_size=input_size, hidden_size=training_parameters['hidden_dim'],
                       n_layers=training_parameters['num_layers'], output_size=target_validation.shape[1],
                       device=device).to(device)
    elif training_parameters['model'] == 'transformer':
        input_size = input_validation.shape[-1]
        net = utils.TransformerModel(in_features=input_size, embedding_size=training_parameters['hidden_dim'],
                               out_features=target_validation.shape[1], nhead=training_parameters['n_head'],
                               dim_feedforward=4 * training_parameters['hidden_dim'],
                               num_layers=training_parameters['num_layers'], dropout=training_parameters['dropout'],
                               activation="relu",
                               classifier_dropout=training_parameters['dropout'],
                               num_classifier_layer=training_parameters['num_layers_classifier']).to(device)
    elif training_parameters['model'] == 'set-transformer':
        input_size = input_validation.shape[-1]            
        net = utils.SetTransformer(dim_input=input_validation.shape[-1], num_outputs=1, dim_output=target_validation.shape[1], 
                                   dim_embedding=training_parameters['dim_embedding'], embedding=training_parameters['embedding'],
                                   dim_hidden_embedding=training_parameters['dim_hidden_embedding'],
                                   num_inds=training_parameters['num_inds'], one_hot=training_parameters['one_hot'],
                                   dim_hidden=training_parameters['hidden_dim'], objective=objective,
                                    num_heads=training_parameters['n_head'], ln=training_parameters['layer_normalization'], 
                                    sab_in_output=training_parameters['sab_in_output'],
                                    num_layers_enc=training_parameters['num_layers_enc'], num_layers_dec=training_parameters['num_layers_classifier'], 
                                    activation_fct='ReLU',data_parameters=data_parameters, dropout=training_parameters['dropout']).to(device)
    return net

def scheduler_step(scheduler, optimizer, epoch):
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    if before_lr != after_lr:
        print("[time: %d] Epoch %d: SGD lr %.8f -> %.8f" % (time(), epoch, before_lr, after_lr))
        
def predict_data_loader(net, dataloader, num_samples_min):
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)
            # Random subset of the samples, to enable the model to work with different sizes of data sets
            input_subset = choose_subset(input, num_samples_min)
            prediction_list.append(net(input_subset))
            
            target_list.append(target)
    return torch.cat(prediction_list, dim=0), torch.cat(target_list, dim=0)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
