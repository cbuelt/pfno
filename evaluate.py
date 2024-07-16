import torch

from models import generate_mcd_samples, LA_Wrapper
from utils import losses
import numpy as np

def log_and_save_evaluation(value, key, results_dict, logging):
    value = np.round(value, decimals=5)
    logging.info(f'{key}: {value}')
    if not key in results_dict.keys():
        results_dict[key] = []
    results_dict[key].append(value)

def generate_samples(uncertainty_quantification, model, a, u, n_samples):
    if uncertainty_quantification == 'dropout':
        out = generate_mcd_samples(model, a, u.shape, n_samples=n_samples)
    elif uncertainty_quantification == 'laplace':
        out = model.predictive_samples(a)
    elif uncertainty_quantification.startswith('scoring-rule'):
        out = model(a, n_samples = n_samples)
    return out

def evaluate(model, training_parameters, loader, device, domain_range):
    uncertainty_quantification = training_parameters['uncertainty_quantification']
    if uncertainty_quantification.endswith('dropout'):
        model.train()
    elif uncertainty_quantification == 'scoring-rule-mu-std':
        model.eval()
    
    mse = 0
    es = 0
    coverage = 0
    int_width = 0
    alpha = training_parameters['alpha']
    
    d = len(next(iter(loader))[0].shape) - 2
    l2loss = losses.LpLoss(d=d, p=2, L=domain_range)
    energy_score = losses.EnergyScore(d = d, p = 2, type = "lp", L=domain_range)

    with torch.no_grad():    
        for sample in loader:
            a, u = sample
            a = a.to(device)
            u = u.to(device)
            batch_size = a.shape[0]
            out = generate_samples(uncertainty_quantification, model, a, u, training_parameters['n_samples_uq'])
            mse += l2loss(out.mean(axis = -1), u).item() / batch_size
            es += energy_score(out, u).item() / batch_size
            # Calculate coverage
            q_lower = torch.quantile(out, alpha/2, axis = -1)
            q_upper = torch.quantile(out, 1-alpha/2, axis = -1)
            coverage += ((u>q_lower) & (u<q_upper)).float().mean().item() / batch_size
            int_width += torch.linalg.norm(q_upper - q_lower).item() / batch_size
    
    return mse, es, coverage, int_width
    
def start_evaluation(model, training_parameters, train_loader, validation_loader, test_loader, results_dict, device, domain_range, logging):
    logging.info(f'Starting evaluation: model {training_parameters["model"]} & uncertainty quantification {training_parameters["uncertainty_quantification"]}')
    
    data_loaders = {'train': train_loader, 'validation': validation_loader, 'test': test_loader}

    if training_parameters['uncertainty_quantification'] == 'laplace':
        model = LA_Wrapper(model, n_samples=training_parameters['n_samples'], method = "last_layer", hessian_structure = "full", optimize = True)
        model.fit(train_loader)
    
    for name, loader in data_loaders.items():
        logging.info(f'Evaluating the model on {name} data.')
        
        mse, es, coverage, int_width = evaluate(model, training_parameters, loader, device, domain_range)
        
        log_and_save_evaluation(mse, 'MSE' + name, results_dict, logging)
        log_and_save_evaluation(es, 'EnergyScore' + name, results_dict, logging)
        log_and_save_evaluation(coverage, 'Coverage' + name, results_dict, logging)
        log_and_save_evaluation(int_width, 'IntervalWidth' + name, results_dict, logging)
        
    