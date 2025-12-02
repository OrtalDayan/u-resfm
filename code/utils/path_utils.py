import os
from utils.Phases import Phases


def join_and_create(path, folder):
    full_path = os.path.join(path, folder)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def path_to_datasets(dataset="megadepth"):
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the code directory, then into datasets
    code_dir = os.path.dirname(current_dir)
    datasetsPath = os.path.join(code_dir, 'datasets', dataset)
    os.makedirs(datasetsPath, exist_ok=True)
    return datasetsPath


def path_to_images(conf):  # not in use
    return ""

# def path_to_condition(conf):
#     experiments_folder = os.path.join('..', 'results')
#     if not os.path.exists(experiments_folder):
#         os.mkdir(experiments_folder)
#     exp_name = conf.get_string('exp_name')
#     return join_and_create(experiments_folder, exp_name)


# def path_to_exp(conf):
#     exp_ver = conf.get_string('exp_version')
#     exp_ver_path = join_and_create(path_to_condition(conf), exp_ver)

#     return exp_ver_path


def path_to_condition(conf):
    """
    Creates the base path with exp_version as the first level.
    Returns: Z:/MVG/resfm-main-orig/code/results/<exp_version>/
    """
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the code directory
    code_dir = os.path.dirname(current_dir)
    # Create the full path to the results directory
    experiments_folder = os.path.join(code_dir, 'results')
    
    if not os.path.exists(experiments_folder):
        os.makedirs(experiments_folder, exist_ok=True)
    
    exp_ver = conf.get_string('exp_version')
    return join_and_create(experiments_folder, exp_ver)


def path_to_exp(conf):
    """
    Creates the full experiment path with exp_name as the second level.
    Returns: Z:/MVG/resfm-main-orig/code/results/<exp_version>/<exp_name>/
    """
    # exp_name = conf.get_string('exp_name')
    # exp_name_path = join_and_create(path_to_condition(conf), exp_name)

    exp_name = conf.get_string('results_path', default=None)
    # exp_name_path = join_and_create(path_to_condition(conf), exp_name)

    # return exp_name_path
    return exp_name


# def path_to_exp(conf):
#     """
#     Get path to experiment results directory for a specific scan.
    
#     Args:
#         conf: Configuration dictionary
    
#     Returns:
#         Path to experiment directory for the scan
#     """
#     # Scan must be provided
#     if not scan:
#         raise ValueError("scan parameter is required for path_to_exp")
    
#     exp_version = conf.get('exp_version', 'default')
    
#     # Determine suffix based on exp_version and configuration
#     if "with_outliers" in exp_version:
#         suffix = "esfm_w_outliers"  # Matches bash script suffix
#     elif "removed_outliers" in exp_version:
#         suffix = "esfm_removed_outliers"  # Matches bash script suffix
#     else:
#         suffix = "ba"  # Default fallback
    
#     # Append appropriate suffix to scan name
#     exp_dir = os.path.join(results_dir, exp_version, f"{scan}_{suffix}")
    
#     # Create directory if it doesn't exist
#     os.makedirs(exp_dir, exist_ok=True)
    
#     return exp_dir



def path_to_phase(conf, phase):
    exp_path = path_to_exp(conf)
    return join_and_create(exp_path, phase.name)


def path_to_scan(conf, phase, scan=None):
    exp_path = path_to_phase(conf, phase)
    scan = conf.get_string("dataset.scan") if scan is None else scan
    return join_and_create(exp_path, scan)


def path_to_model(conf, phase, epoch=None, scan=None, best=False):
    if conf.get_string('pretrainedPath', default=None) is not None and phase is not Phases.FINE_TUNE and best:
        return conf.get_string('pretrainedPath')

    # if phase in [Phases.TRAINING, Phases.VALIDATION, Phases.TEST]:
    #     parent_folder = path_to_exp(conf)
    # else:
    #     parent_folder = path_to_scan(conf, phase, scan=scan)

    parent_folder =conf.get_string('results_path', default=None)

    if best:
        models_path = join_and_create(parent_folder, 'models')
        print('models_path', models_path)
        if epoch is None:
            modelsList = os.listdir(models_path)
            modelsList = [int(model.split('_Ep')[1].split('.pt')[0]) for model in modelsList if model.startswith('Model')]
            epoch = sorted(modelsList)[-1]
    else:
        models_path = join_and_create(parent_folder, 'models_all')

    if epoch is None:
        model_file_name = "Final_Model.pt"
    else:
        model_file_name = "Model_Ep{}.pt".format(epoch)

    return os.path.join(models_path, model_file_name)

def path_to_model_resume_optimizing(conf, epoch=None, scan=None):

    parent_folder = path_to_exp(conf)
    print(parent_folder)
    models_path = os.path.join(parent_folder, 'OPTIMIZATION', scan, 'models_all')
    modelsList = os.listdir(models_path)
    modelsList = [int(model.split('_Ep')[1].split('.pt')[0]) for model in modelsList if model.startswith('Model')]
    epoch = sorted(modelsList)[-1]
    model_file_name = "Model_Ep{}.pt".format(epoch)

    return os.path.join(models_path, model_file_name), epoch


def path_to_model_resume_learning(conf):

    parent_folder = path_to_exp(conf)
    print(parent_folder)
    # else:
    #     parent_folder = path_to_scan(conf, phase, scan=scan)

    models_path = os.path.join(parent_folder, 'models_all')
    modelsList = os.listdir(models_path)
    modelsList = [int(model.split('_Ep')[1].split('.pt')[0]) for model in modelsList if model.startswith('Model')]
    epoch = sorted(modelsList)[-1]
    model_file_name = "Model_Ep{}.pt".format(epoch)

    return os.path.join(models_path, model_file_name), epoch

def path_to_learning_data(conf, phase):
    return join_and_create(path_to_condition(conf), phase)



def path_to_cameras(conf, phase, epoch=None, scan=None,round=""):

    scan_path = path_to_exp(conf)
    # scan_path = conf.get_string('results_path', default=None)
    cameras_path = join_and_create(scan_path, 'forFigures')
    cameras_path = join_and_create(cameras_path, round)

    if epoch is None:
        cameras_file_name = f"{scan}_Final_Cameras"
    else:
        cameras_file_name = "{0}_Cameras_Ep{1}".format(scan, epoch)

    return os.path.join(cameras_path, cameras_file_name)

def path_to_outliers(conf, phase, epoch=None, scan=None):
    scan_path = path_to_scan(conf, phase, scan=scan)
    cameras_path = join_and_create(scan_path, 'outliers_results')

    if epoch is None:
        cameras_file_name = "Final_outliers"
    else:
        cameras_file_name = "outliers_Ep{}".format(epoch)

    return os.path.join(cameras_path, cameras_file_name)


def path_to_metrics(conf, phase, epoch=None, scan=None):
    scan_path = path_to_scan(conf, phase, scan=scan)
    excels_path = join_and_create(scan_path, 'metrics')

    if epoch is None:
        metrics_file_name = "Final_metrics"
    else:
        metrics_file_name = "metrics_Ep{}".format(epoch)

    metrics_file_name += '.xlsx'

    return os.path.join(excels_path, metrics_file_name)

def path_to_plots(conf, phase, epoch=None, scan=None, triangulation=False, filtered=False, extraText=None):
    # scan_path = path_to_scan(conf, phase, scan=scan)
    scan_path = conf.get_string('results_path', default=None)
    plots_path = join_and_create(scan_path, 'plots')

    if epoch is None:
        plots_file_name = "Final_plots.html"
    else:
        plots_file_name = "Plot_Ep{}.html".format(epoch)

    if extraText is not None:
        plots_file_name = plots_file_name.split(".html")[0]+f"{extraText}.html"

    if triangulation:
        plots_file_name = plots_file_name.split(".html")[0]+"_triangulation.html"

    if filtered:
        plots_file_name = plots_file_name.split(".html")[0]+"_filtered4.html"

    return os.path.join(plots_path, plots_file_name)


def path_to_reconstructions(conf, phase, epoch=None, scan=None, name=None):
    # scan_path = path_to_scan(conf, phase, scan=scan) + '_ba'
    scan_path = conf.get_string('results_path', default=None)
    reconstructions_path = join_and_create(scan_path, 'colmap_reconstructions')

    if epoch is None:
        plots_file_name = "Final_recon"
    else:
        plots_file_name = "recon_Ep{}".format(epoch)

    reconstructions_path = join_and_create(reconstructions_path, plots_file_name)
    if name is not None:
        reconstructions_path = join_and_create(reconstructions_path, name)

    return reconstructions_path


def path_to_wandb_logs(conf, phase, scan=None):
    # if phase is Phases.OPTIMIZATION:
    #     # exp_path = path_to_scan(conf, phase, scan=None)
    #     # exp_path = conf.get_string('results_path', default=None)
    #     scan_path = conf.get_string('results_path', default=None)
    # else:
    #     exp_path = path_to_exp(conf)
    exp_path = path_to_exp(conf)
    logs_path = join_and_create(exp_path, "wandb")
    return logs_path

def path_to_logs(conf, phase):
    phase_path = path_to_phase(conf, phase)
    logs_path = join_and_create(phase_path, "logs")
    return logs_path


def path_to_code_logs(conf):
    exp_path = path_to_exp(conf)
    code_path = join_and_create(exp_path, "code")
    return code_path


def path_to_conf(conf_file):
    return os.path.join('confs', conf_file)

def get_lsf_file_path(conf, args):
    return os.path.join(path_to_exp(conf), f'{args.run_name}.lsf')

def get_lsf_output_path(conf):
    return os.path.join(path_to_exp(conf), f'output_%J_%I.txt'), os.path.join(path_to_exp(conf), f'err_%J_%I.txt')