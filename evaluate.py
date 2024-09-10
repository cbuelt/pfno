import torch

from models import generate_mcd_samples, LA_Wrapper
from utils import losses, train_utils
import numpy as np

def generate_samples(uncertainty_quantification, model, a, u, n_samples):
    if uncertainty_quantification.endswith('dropout'):
        model.train()
    elif uncertainty_quantification == 'scoring-rule-reparam':
        model.eval()    
    if uncertainty_quantification == 'dropout':
        out = generate_mcd_samples(model, a, u.shape, n_samples=n_samples)
    elif uncertainty_quantification == 'laplace':
        out = model.predictive_samples(a)
    elif uncertainty_quantification.startswith('scoring-rule'):
        out = model(a, n_samples = n_samples)
    return out

def autoregressive_step(uncertainty_quantification, model, a):
    if uncertainty_quantification.endswith('dropout'):
        model.train()
    elif uncertainty_quantification == 'scoring-rule-reparam':
        model.eval()    
    if uncertainty_quantification == 'laplace':
        out = model.model(a)
    elif uncertainty_quantification == "dropout":
        out = model(a)
    else:
        out = model(a).mean(axis = -1)
    return out

def evaluate_autoregressive(model, training_parameters, data_parameters, loader, name, device, domain_range):

    uncertainty_quantification = training_parameters['uncertainty_quantification']    
    mse = 0
    es = 0
    coverage = 0
    interval_width = 0
    crps = 0
    gaussian_nll = 0
    alpha = training_parameters['alpha']
    # Get pred horizon
    if name == "Test":
        pred_horizon = data_parameters['pred_horizon']
    else:
        pred_horizon = data_parameters['train_horizon']
    stepwise_evaluation = data_parameters['stepwise_evaluation']
    
    nlon, weights = domain_range
    l2loss = losses.SphericalL2Loss(nlon = nlon, weights = weights.to(device))
    energy_score = losses.EnergyScore(type = "spherical", nlon = nlon, weights = weights.to(device))
    crps_loss = losses.CRPS(nlon = nlon, weights = weights.to(device))
    gaussian_nll_loss = losses.GaussianNLL(weights = weights.to(device))
    coverage_loss = losses.Coverage(alpha, weights = weights.to(device))
    interval_width_loss = losses.IntervalWidth(alpha, weights = weights.to(device))

    if stepwise_evaluation:
        with torch.no_grad():    
            for sample in loader:
                a, u = sample
                a = a.to(device)
                u = u.to(device)
                batch_size = a.shape[0]
                # Autoregressive steps
                for step in range(pred_horizon):
                    if step == 0:
                        out = generate_samples(uncertainty_quantification, model, a, u[:,0], training_parameters['n_samples_uq'])
                    else:
                        out = out.mean(axis = -1)
                        out = generate_samples(uncertainty_quantification, model, out, u[:,0], training_parameters['n_samples_uq'])
                    # Losses
                    mse += l2loss(out.mean(axis = -1), u[:,step]).item() * batch_size / len(loader.dataset) / pred_horizon
                    es += energy_score(out, u[:,step]).item() * batch_size / len(loader.dataset) / pred_horizon
                    crps += crps_loss(out, u[:,step]).item() * batch_size / len(loader.dataset) / pred_horizon
                    gaussian_nll += gaussian_nll_loss(out, u[:,step]).item() * batch_size / len(loader.dataset) / pred_horizon
                    coverage += coverage_loss(out, u[:,step]).item() * batch_size / len(loader.dataset) / pred_horizon
                    interval_width += interval_width_loss(out, u[:,step]).item() * batch_size / len(loader.dataset) / pred_horizon
    else:
        with torch.no_grad():    
            for sample in loader:
                a, u = sample
                a = a.to(device)
                u = u[:,-1].to(device)
                batch_size = a.shape[0]

                # Autoregressive steps
                for _ in range(pred_horizon-1):
                    a = autoregressive_step(uncertainty_quantification, model, a)
                # Final step
                out = generate_samples(uncertainty_quantification, model, a, u, training_parameters['n_samples_uq'])
                # Losses
                mse += l2loss(out.mean(axis = -1), u).item() * batch_size / len(loader.dataset)
                es += energy_score(out, u).item() * batch_size / len(loader.dataset)
                crps += crps_loss(out, u).item() * batch_size / len(loader.dataset)
                gaussian_nll += gaussian_nll_loss(out, u).item() * batch_size / len(loader.dataset)
                coverage += coverage_loss(out, u).item() * batch_size / len(loader.dataset)
                interval_width += interval_width_loss(out, u).item() * batch_size / len(loader.dataset)
        
    return mse, es, crps, gaussian_nll, coverage, interval_width



def evaluate(model, training_parameters, loader, device, domain_range):
    uncertainty_quantification = training_parameters['uncertainty_quantification']    
    mse = 0
    es = 0
    coverage = 0
    interval_width = 0
    crps = 0
    gaussian_nll = 0
    alpha = training_parameters['alpha']
    
    d = len(next(iter(loader))[0].shape) - 2
    l2loss = losses.LpLoss(d=d, p=2, L=domain_range)
    energy_score = losses.EnergyScore(d = d, p = 2, type = "lp", L=domain_range)
    crps_loss = losses.CRPS()
    gaussian_nll_loss = losses.GaussianNLL()
    coverage_loss = losses.Coverage(alpha)
    interval_width_loss = losses.IntervalWidth(alpha)

    with torch.no_grad():    
        for sample in loader:
            a, u = sample
            a = a.to(device)
            u = u.to(device)
            batch_size = a.shape[0]
            out = generate_samples(uncertainty_quantification, model, a, u, training_parameters['n_samples_uq'])
            mse += l2loss(out.mean(axis = -1), u).item() * batch_size / len(loader.dataset)
            es += energy_score(out, u).item() * batch_size / len(loader.dataset)
            crps += crps_loss(out, u).item() * batch_size / len(loader.dataset)
            gaussian_nll += gaussian_nll_loss(out, u).item() * batch_size / len(loader.dataset)
            coverage += coverage_loss(out, u).item() * batch_size / len(loader.dataset)
            interval_width += interval_width_loss(out, u).item() * batch_size / len(loader.dataset)
    
    return mse, es, crps, gaussian_nll, coverage, interval_width
    
def start_evaluation(model, training_parameters, data_parameters,train_loader, validation_loader, test_loader, results_dict, device, domain_range, logging, filename):
    logging.info(f'Starting evaluation: model {training_parameters["model"]} & uncertainty quantification {training_parameters["uncertainty_quantification"]}')

    if data_parameters["dataset_name"] == "era5" or data_parameters["dataset_name"] == "SSWE":
        data_loaders = {'Validation': validation_loader, 'Test': test_loader}
    else:
        data_loaders = {'Train': train_loader, 'Validation': validation_loader, 'Test': test_loader}

    if training_parameters['uncertainty_quantification'] == 'laplace':
        model = LA_Wrapper(model, n_samples=training_parameters['n_samples'], method = "last_layer", hessian_structure = "full", optimize = True)
        model.fit(train_loader)
        train_utils.checkpoint(model, filename)
    
    for name, loader in data_loaders.items():
        logging.info(f'Evaluating the model on {name} data.')
        
        if training_parameters["model"] == "SFNO":
            mse, es, crps, gaussian_nll, coverage, int_width = evaluate_autoregressive(model, training_parameters, data_parameters, loader, name, device, domain_range)
        else:
            mse, es, crps, gaussian_nll, coverage, int_width = evaluate(model, training_parameters, loader, device, domain_range)
        
        train_utils.log_and_save_evaluation(mse, 'MSE' + name, results_dict, logging)
        train_utils.log_and_save_evaluation(es, 'EnergyScore' + name, results_dict, logging)
        train_utils.log_and_save_evaluation(crps, 'CRPS' + name, results_dict, logging)
        train_utils.log_and_save_evaluation(gaussian_nll, 'Gaussian NLL' + name, results_dict, logging)
        train_utils.log_and_save_evaluation(coverage, 'Coverage' + name, results_dict, logging)
        train_utils.log_and_save_evaluation(int_width, 'IntervalWidth' + name, results_dict, logging)
        
    