import torch

from models import generate_mcd_samples, LA_Wrapper
from utils import losses, train_utils
import numpy as np


def generate_samples(
    uncertainty_quantification: str,
    model,
    a: torch.Tensor,
    u: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Mehtod to generate samples from the underlying model with the specified uncertainty quantification method.

    Args:
        uncertainty_quantification (str): Method for uncertainty quantification
        model (_type_): Neural network model
        a (torch.Tensor): Input data
        u (torch.Tensor): Target data
        n_samples (int): Number of samples to generate

    Returns:
        torch.Tensor: _description_
    """
    if uncertainty_quantification.endswith("dropout"):
        model.train()
    elif uncertainty_quantification == "scoring-rule-reparam":
        model.eval()
    if uncertainty_quantification == "dropout":
        out = generate_mcd_samples(model, a, u.shape, n_samples=n_samples)
    elif uncertainty_quantification == "laplace":
        out = model.predictive_samples(a, n_samples=n_samples)
    elif uncertainty_quantification.startswith("scoring-rule"):
        out = model(a, n_samples=n_samples)
    return out


def evaluate_autoregressive(
    model,
    training_parameters: dict,
    data_parameters: dict,
    loader,
    name: str,
    device,
    domain_range,
):
    """Method to evaluate the given model in an autoregressive manner.

    Args:
        model (_type_): Underlying model
        training_parameters (dict): Dictionary containing training parameters
        data_parameters (dict): Dictionary containing data parameters
        loader (_type_): Data loader
        name (str): Name indicating the data set
        device (_type_): Device to run the model on
        domain_range (_type_): Either list of domain range for LP based loss or nlon and weights for spherical loss

    Returns:
        _type_: Tuple of evaluation metrics
    """
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    mse = 0
    es = 0
    coverage = 0
    interval_width = 0
    crps = 0
    gaussian_nll = 0
    alpha = training_parameters["alpha"]
    # Get pred horizon
    if name == "Test":
        pred_horizon = data_parameters["pred_horizon"]
        _, _, nlon, weights = domain_range
    else:
        pred_horizon = data_parameters["train_horizon"]
        nlon, weights, _, _ = domain_range
    stepwise_evaluation = data_parameters["stepwise_evaluation"]

    l2loss = losses.SphericalL2Loss(nlon=nlon, weights=weights.to(device))
    energy_score = losses.EnergyScore(
        type="spherical", nlon=nlon, weights=weights.to(device)
    )
    crps_loss = losses.CRPS(nlon=nlon, weights=weights.to(device))
    gaussian_nll_loss = losses.GaussianNLL(weights=weights.to(device))
    coverage_loss = losses.Coverage(alpha, weights=weights.to(device))
    interval_width_loss = losses.IntervalWidth(alpha, weights=weights.to(device))

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
                        out = generate_samples(
                            uncertainty_quantification,
                            model,
                            a,
                            u[:, :, 0],
                            training_parameters["n_samples_uq"],
                        )
                    else:
                        out = out.mean(axis=-1)
                        out = generate_samples(
                            uncertainty_quantification,
                            model,
                            out,
                            u[:, :, 0],
                            training_parameters["n_samples_uq"],
                        )
                    # Losses
                    mse += (
                        l2loss(out.mean(axis=-1), u[:, :, step]).item()
                        * batch_size
                        / len(loader.dataset)
                        / pred_horizon
                    )
                    es += (
                        energy_score(out, u[:, :, step]).item()
                        * batch_size
                        / len(loader.dataset)
                        / pred_horizon
                    )
                    crps += (
                        crps_loss(out, u[:, :, step]).item()
                        * batch_size
                        / len(loader.dataset)
                        / pred_horizon
                    )
                    gaussian_nll += (
                        gaussian_nll_loss(out, u[:, :, step]).item()
                        * batch_size
                        / len(loader.dataset)
                        / pred_horizon
                    )
                    coverage += (
                        coverage_loss(out, u[:, :, step]).item()
                        * batch_size
                        / len(loader.dataset)
                        / pred_horizon
                    )
                    interval_width += (
                        interval_width_loss(out, u[:, :, step]).item()
                        * batch_size
                        / len(loader.dataset)
                        / pred_horizon
                    )
    else:
        with torch.no_grad():
            for sample in loader:
                a, u = sample
                a = a.to(device)
                u = u[:, :, -1].to(device)
                batch_size = a.shape[0]
                # Autoregressive steps
                for _ in range(pred_horizon - 1):
                    a = train_utils.autoregressive_step(
                        uncertainty_quantification, model, a
                    )
                # Final step
                out = generate_samples(
                    uncertainty_quantification,
                    model,
                    a,
                    u,
                    training_parameters["n_samples_uq"],
                )
                # Losses
                mse += (
                    l2loss(out.mean(axis=-1), u).item()
                    * batch_size
                    / len(loader.dataset)
                )
                es += energy_score(out, u).item() * batch_size / len(loader.dataset)
                crps += crps_loss(out, u).item() * batch_size / len(loader.dataset)
                gaussian_nll += (
                    gaussian_nll_loss(out, u).item() * batch_size / len(loader.dataset)
                )
                coverage += (
                    coverage_loss(out, u).item() * batch_size / len(loader.dataset)
                )
                interval_width += (
                    interval_width_loss(out, u).item()
                    * batch_size
                    / len(loader.dataset)
                )

    return mse, es, crps, gaussian_nll, coverage, interval_width


def evaluate(model, training_parameters: dict, loader, device, domain_range):
    """ Method to evaluate the given model.

    Args:
        model (_type_): Underlying model
        training_parameters (dict): Dictionary containing training parameters
        loader (_type_): Data loader
        device (_type_): Device to run the model on
        domain_range (_type_): Either list of domain range for LP based loss or nlon and weights for spherical loss

    Returns:
        _type_: Tuple of evaluation metrics
    """
    uncertainty_quantification = training_parameters["uncertainty_quantification"]
    mse = 0
    es = 0
    coverage = 0
    interval_width = 0
    crps = 0
    gaussian_nll = 0
    alpha = training_parameters["alpha"]

    d = len(next(iter(loader))[0].shape) - 2
    l2loss = losses.LpLoss(d=d, p=2, L=domain_range)
    energy_score = losses.EnergyScore(d=d, p=2, type="lp", L=domain_range)
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
            out = generate_samples(
                uncertainty_quantification,
                model,
                a,
                u,
                training_parameters["n_samples_uq"],
            )
            mse += (
                l2loss(out.mean(axis=-1), u).item() * batch_size / len(loader.dataset)
            )
            es += energy_score(out, u).item() * batch_size / len(loader.dataset)
            crps += crps_loss(out, u).item() * batch_size / len(loader.dataset)
            gaussian_nll += (
                gaussian_nll_loss(out, u).item() * batch_size / len(loader.dataset)
            )
            coverage += coverage_loss(out, u).item() * batch_size / len(loader.dataset)
            interval_width += (
                interval_width_loss(out, u).item() * batch_size / len(loader.dataset)
            )

    return mse, es, crps, gaussian_nll, coverage, interval_width

def start_evaluation(
    model,
    training_parameters:dict,
    data_parameters:dict,
    train_loader,
    validation_loader,
    test_loader,
    results_dict:dict,
    device,
    domain_range,
    logging,
    filename:str,
    **kwargs,
):
    """ Performs evaluation of the model on the given data sets.

    Args:
        model (_type_): Underlying model
        training_parameters (dict): Dictionary of training parameters
        data_parameters (dict): Dictionary of data parameters
        train_loader (_type_): Train loader
        validation_loader (_type_): Validation loader
        test_loader (_type_): Test loader
        results_dict (dict): Dictionary to store results
        device (_type_): Device to run the model on
        domain_range (_type_): Either list of domain range for LP based loss or nlon and weights for spherical loss
        logging (_type_): Logging object
        filename (str): Filename to save the model
    """
    # Need to add additional train loader for autoregressive Laplace
    laplace_train_loader = kwargs.get("laplace_train_loader", None)
    logging.info(
        f'Starting evaluation: model {training_parameters["model"]} & uncertainty quantification {training_parameters["uncertainty_quantification"]}'
    )
    # Don't evaluate for train on era5 and SSWE for computational reasons
    if (
        data_parameters["dataset_name"] == "era5"
        or data_parameters["dataset_name"] == "SSWE"
    ):
        data_loaders = {"Validation": validation_loader, "Test": test_loader}
    else:
        data_loaders = {
            "Train": train_loader,
            "Validation": validation_loader,
            "Test": test_loader,
        }

    if training_parameters["uncertainty_quantification"] == "laplace" and (not isinstance(model, LA_Wrapper)):
        model = LA_Wrapper(
            model,
            n_samples=training_parameters["n_samples_uq"],
            method="last_layer",
            hessian_structure="full",
            optimize=True,
        )
        if laplace_train_loader is not None:
            model.fit(laplace_train_loader)
        else:
            model.fit(train_loader)
        train_utils.checkpoint(model, filename)

    for name, loader in data_loaders.items():
        logging.info(f"Evaluating the model on {name} data.")

        if training_parameters["model"] == "SFNO":
            mse, es, crps, gaussian_nll, coverage, int_width = evaluate_autoregressive(
                model,
                training_parameters,
                data_parameters,
                loader,
                name,
                device,
                domain_range,
            )
        else:
            mse, es, crps, gaussian_nll, coverage, int_width = evaluate(
                model, training_parameters, loader, device, domain_range
            )

        train_utils.log_and_save_evaluation(mse, "MSE" + name, results_dict, logging)
        train_utils.log_and_save_evaluation(
            es, "EnergyScore" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(crps, "CRPS" + name, results_dict, logging)
        train_utils.log_and_save_evaluation(
            gaussian_nll, "Gaussian NLL" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(
            coverage, "Coverage" + name, results_dict, logging
        )
        train_utils.log_and_save_evaluation(
            int_width, "IntervalWidth" + name, results_dict, logging
        )
