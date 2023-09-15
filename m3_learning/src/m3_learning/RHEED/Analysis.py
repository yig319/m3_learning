import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.stats import zscore
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt

from sklearn.decomposition import PCA
from m3_learning.RHEED.Viz import Viz

def detect_peaks(curve_x, curve_y, camera_freq, laser_freq, curve_params):
    """
    Detects peaks in a curve based on the provided parameters.

    Args:
        curve_x (numpy.array): The x-values of the curve.
        curve_y (numpy.array): The y-values of the curve.
        camera_freq (float): The frequency of the camera.
        laser_freq (float): The frequency of the laser.
        convolve_step (int): The step size for convolution.
        prominence (float): The prominence threshold for peak detection.

    Returns:
        tuple: A tuple containing the peak positions, partial curve x-values, and partial curve y-values.
    """
    convolve_step = curve_params['convolve_step']
    prominence = curve_params['prominence']
    mode = curve_params['mode']


    dist = int(camera_freq/laser_freq*0.6)
    step = np.hstack((np.ones(convolve_step), -1*np.ones(convolve_step)))
    dary_step = np.convolve(curve_y, step, mode=mode)
    dary_step = np.abs(dary_step)

    filtered_curve_y = dary_step/convolve_step
    x_peaks, properties = signal.find_peaks(dary_step, prominence=prominence, distance=dist)
    x_peaks = x_peaks[x_peaks>dist]
    x_peaks = x_peaks[x_peaks<len(curve_y)-dist]
    
    # get all partial curve 
    xs, ys = [], []
    for i in range(1, len(x_peaks)):
        xs.append(list(curve_x[5+x_peaks[i-1]:x_peaks[i]]))
        ys.append(list(curve_y[5+x_peaks[i-1]:x_peaks[i]]))
    return x_peaks/camera_freq, xs, ys

def remove_outlier(x, y, ub):

    """
    Removes outliers from the given data based on the provided upper bound.

    Args:
        x (numpy.array): The x-values of the data.
        y (numpy.array): The y-values of the data.
        ub (float): The upper bound for z-score filtering.

    Returns:
        tuple: A tuple containing the filtered x-values and y-values.
    """
    
    z = zscore(y, axis=0, ddof=0)
    x = np.delete(x, np.where(z>ub))
    y = np.delete(y, np.where(z>ub))
    return x, y

def smooth(y, box_pts):
    """
    Applies a smoothing filter to the given data using a moving average window.

    Args:
        y (numpy.array): The input data.
        box_pts (int): The size of the moving average window.

    Returns:
        numpy.array: The smoothed data.
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def normalize_0_1(y, I_start, I_end, I_diff=None, unify=True):
    """
        Normalizes the given data to the range [0, 1] based on the provided intensity values.

    Args:
        y (numpy.array): The input data.
        I_start (float): The start intensity value.
        I_end (float): The end intensity value.
        I_diff (float, optional): The intensity difference used for normalization. Defaults to None.
        unify (bool, optional): Whether to unify the normalization range regardless of the intensity order. Defaults to True.

    Returns:
        numpy.array: The normalized data.
    """
    if not I_diff:
        I_diff = I_end-I_start
    
    # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
    if I_end - I_start == 0: # avoid devide by 0
        y_nor = (y-I_start)
    elif unify:
        if I_end < I_start:
            y_nor = (y-I_start)/I_diff
        else:
            y_nor = (I_start-y)/I_diff
    else:
        if I_end < I_start:
            y_nor = (y-I_end)/I_diff
        else:
            y_nor = (y-I_start)/I_diff
    return y_nor

def de_normalize_0_1(y_nor_fit, I_start, I_end, I_diff=None, unify=True):
    """
    De-normalizes the given normalized data back to the original range based on the provided intensity values.

    Args:
        y_nor_fit (numpy.array): The normalized data to be de-normalized.
        I_start (float): The start intensity value.
        I_end (float): The end intensity value.
        I_diff (float, optional): The intensity difference used for normalization. Defaults to None.
        unify (bool, optional): Whether to unify the normalization range regardless of the intensity order. Defaults to True.

    Returns:
        numpy.array: The de-normalized data.
    """
    if not I_diff:
        I_diff = I_end-I_start
    if not unify:
        I_diff = np.abs(I_diff)
    
    # use I/I0, I0 is saturation intensity (last value) and scale to 0-1 based 
    if I_end - I_start == 0: # avoid devide by 0
        y_nor = (y-I_start)
    elif unify:
        if I_end < I_start:
            y_fit = I_start-y_nor_fit*I_diff
        else:
            y_fit = y_nor_fit*I_diff+I_start

    else:
        if I_end < I_start:
            y_fit = y_nor_fit*I_diff+I_end
        else:
            y_fit = y_nor_fit*I_diff+I_start
    return y_fit
    

def denoise_fft(sample_x, sample_y, cutoff_freq, denoise_order, sample_frequency, viz=False):

    nyquist = 0.5 * sample_frequency
    low = cutoff_freq / nyquist
    b, a = butter(denoise_order, low, btype='low')

    # Apply the low-pass filter to denoise the signal
    denoised_sample_y = filtfilt(b, a, sample_y)

    # Compute the frequency spectrum of the original and denoised signals
    freq = np.fft.rfftfreq(len(sample_x), d=1/sample_frequency)
    fft_original = np.abs(np.fft.rfft(sample_y))
    fft_denoised = np.abs(np.fft.rfft(denoised_sample_y))

    if viz:
        fig, axes = plt.subplots(2, 1, figsize=(8, 4))
        axes[0].scatter(sample_x, sample_y, label='Original Signal')
        axes[0].plot(sample_x, denoised_sample_y, color='r', label='Denoised Signal')
        axes[0].legend()

        axes[1].plot(freq, fft_original, label='Original Spectrum')
        axes[1].plot(freq, fft_denoised, label='Denoised Spectrum')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_yscale('log')
        axes[1].legend()
        plt.tight_layout()
        plt.title('fft filter')
        plt.show()
    return sample_x, denoised_sample_y

from scipy.signal import medfilt
def denoise_median(sample_x, sample_y, kernel_size, viz=False):
    denoised_sample_y = medfilt(sample_y, kernel_size=kernel_size)
    if viz:
        plt.figure(figsize=(8,2))
        plt.scatter(sample_x, sample_y, label='Original Signal')
        plt.plot(sample_x, denoised_sample_y, color='r', label='Denoised Signal')
        plt.tight_layout()
        plt.title('median filter')
        plt.show()

    return sample_x, denoised_sample_y

def process_rheed_data(sample_x, sample_y, camera_freq, denoise_params, viz=False):    

    """Processes RHEED data by interpolating, denoising, and applying dimensionality reduction.

    Args:
        savgol_window_order (tuple, optional): The order of the Savitzky-Golay filter window. Defaults to (15, 3).
        pca_component (int, optional): The number of components for PCA. Defaults to 10.

    Returns:
        tuple: A."""

    savgol_window_order = denoise_params['savgol_window_order']
    pca_component = denoise_params['pca_component']
    fft_cutoff_order = denoise_params['fft_cutoff_order']
    median_kernel_size = denoise_params['median_kernel_size'] 

    denoised_sample_y = sample_y

    # denoise the data
    if not isinstance(savgol_window_order, type(None)):
        denoised_sample_y = savgol_filter(sample_y, savgol_window_order[0], savgol_window_order[1])
        if viz:
            fig, ax = plt.subplots(1, 1, figsize=(8, 2))
            ax.scatter(sample_x, sample_y, label='Original Signal')
            ax.plot(sample_x, denoised_sample_y, color='r', label='Denoised Signal')
            plt.legend()
            plt.tight_layout()
            plt.title('savgol_filter')
            plt.show()

    # # apply PCA
    # if pca_component:
    #     sample_y = denoised_sample_y 
    #     pca = PCA(n_components=pca_component)
    #     denoised_sample_y = pca.inverse_transform(pca.fit_transform(sample_y))
    #     if viz:
    #         fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    #         ax.scatter(sample_x, sample_y, label='Original Signal')
    #         ax.plot(sample_x, denoised_sample_y, color='r', label='Denoised Signal')
    #         plt.legend()
    #         plt.tight_layout()
    #         plt.show()

    # fft
    if not isinstance(fft_cutoff_order, type(None)):
        sample_y = denoised_sample_y 
        denoised_sample_y = denoise_fft(sample_x, sample_y, cutoff_freq=20, denoise_order=1, 
                                        sample_frequency=camera_freq, viz=viz)
    
    # median filter
    if not isinstance(median_kernel_size, type(None)):
        sample_y = denoise_median(sample_x, sample_y, kernel_size=median_kernel_size, viz=viz)

    return sample_x, sample_y


def reset_tails(ys, ratio=0.1):
    for i, y in enumerate(ys):
        num = int(len(y) * ratio)
        y[-num:] = y[-2*num:-num]
        ys[i] = y
    return ys

def linear_func(x, a, b):
    return a*x + b

from scipy.optimize import curve_fit
def remove_linear_bg(xs, ys, linear_ratio=0.8):
    for i in range(len(ys)):
        length = int(len(ys[i]) * linear_ratio)
        popt, pcov = curve_fit(linear_func, xs[i][-length:], ys[i][-length:])
        a, b = popt
        y_fit = linear_func(np.array(xs[i]), a, b=0)
        ys[i] = ys[i] - y_fit
    return xs, ys

def process_curves(xs, ys, curve_params):

    tune_tail = curve_params['tune_tail']
    trim_first = curve_params['trim_first']

    # trim tails
    if tune_tail:
        ys = reset_tails(ys)
    if trim_first != 0:
        xs_trimed, ys_trimed = [], []
        for x, y in zip(xs, ys):
            ys_trimed.append(y[trim_first:])
            xs_trimed.append(np.linspace(x[0], x[-1], len(y[trim_first:])))
        xs, ys = xs_trimed, ys_trimed

    # remove linear background
    xs, ys = remove_linear_bg(xs, ys, linear_ratio=0.8)
    return xs, ys

# def process_rheed_data(xs, ys, 
# camera_freq, savgol_window_order=(15, 3), pca_component=10, 
#                         fft_cutoff=20, fft_order=1, median_kernel_size=51):    

#     """Processes RHEED data by interpolating, denoising, and applying dimensionality reduction.

#     Args:
#         xs (list): List of x-values for each partial curve.
#         ys (list): List of y-values for each partial curve.
#         length (int, optional): The desired length for interpolation. Defaults to 500.
#         savgol_window_order (tuple, optional): The order of the Savitzky-Golay filter window. Defaults to (15, 3).
#         pca_component (int, optional): The number of components for PCA. Defaults to 10.

#     Returns:
#         tuple: A."""

#     # interpolate the data to same size 
#     if length == None:
#         length = int(np.mean([len(x) for x in xs]))

#     xs_processed = []
#     ys_processed = []
#     for x, y in zip(xs, ys):
#         x_sl = np.linspace(np.min(x), np.max(x), length)
#         y_sl = np.interp(x_sl, x, y)
#         xs_processed.append(x_sl)
#         ys_processed.append(y_sl)
#     xs_processed, ys_processed = np.array(xs_processed), np.array(ys_processed)

#     # denoise the data
#     if savgol_window_order:
#         ys_processed = savgol_filter(ys_processed, savgol_window_order[0], savgol_window_order[1])

#     # apply PCA
#     if pca_component:
#         pca = PCA(n_components=pca_component)
#         ys_processed = pca.inverse_transform(pca.fit_transform(ys_processed))

#     # fft
#     if fft_cutoff and fft_order:
#         denoised_sample_y = denoise_fft(sample_x, sample_y, cutoff_freq=20, denoise_order=1, sample_frequency=camera_freq)
    
#     # median filter
#     if median_kernel_size:
#         denoised_sample_y = denoise_median(sample_x, sample_y, kernel_size=median_kernel_size)

#     # trim tails
#     trim_first = fit_settings['trim_first']
#     if fit_settings['tune_tail']:
#         ys = reset_tails(ys)
#     if trim_first != 0:
#         xs_trimed, ys_trimed = [], []
#         for x, y in zip(xs, ys):
#             ys_trimed.append(y[trim_first:])
#             xs_trimed.append(np.linspace(x[0], x[-1], len(y[trim_first:])))
#         xs, ys = xs_trimed, ys_trimed

#     # remove linear background
#     xs, ys = remove_linear_bg(xs, ys, linear_ratio=0.8)
    
#     return xs_processed, ys_processed


def fit_exp_function(xs, ys, growth_name,
        fit_settings = {'I_diff': None, 'unify':True, 'bounds':[0.01, 1], 'p_init':(1, 0.1, 0.4), 'n_std':1}):
    """Fits a""n exponential function to the given data.

    Args:
        xs (list): List of x-values for each partial curve.
        ys (list): List of y-values for each partial curve.
        growth_name (str): Name of the growth.
        fit_settings (dict, optional): Setting parameters for fitting function. Defaults to {'I_diff': 5000, 'unify': True, 'bounds':[0.01, 1], 'p_init':[0.1, 0.4, 0.1]}.

    Returns:
        tuple: A tup""le containing the fitted parameters, and a list of processed RHEED data.
    """
    # use simplified version to avoid overfitting to wrong fitting function
    def exp_func_inc_simp(x, b1, relax1):
        return b1*(1 - np.exp(-x/relax1))
    def exp_func_dec_simp(x, b2, relax2):
        return b2*np.exp(-x/relax2)
    
    # real function to show
    def exp_func_inc(x, a1, b1, relax1):
        return (a1*x+b1)*(1 - np.exp(-x/relax1))
    def exp_func_dec(x, a2, b2, relax2):
        return (a2*x+b2)*np.exp(-x/relax2)
  

    I_diff = fit_settings['I_diff']
    bounds = fit_settings['bounds']
    p_init = fit_settings['p_init']
    unify = fit_settings['unify']

    print(unify)
    parameters = []
    ys_nor, ys_nor_fit, ys_nor_fit_failed, ys_fit = [], [], [], []
    labels, losses = [], []
    
    for i in range(len(xs)):
        
        # section: normalize the curve
        x = np.linspace(1e-5, 1, len(ys[i])) # use second as x axis unit
        n_avg = len(ys[i])//100+3
        I_end = np.mean(ys[i][-n_avg:])
        I_start = np.mean(ys[i][:n_avg])
        y_nor = normalize_0_1(ys[i], I_start, I_end, I_diff, unify)
        
        if unify:
            params, params_covariance = curve_fit(exp_func_inc, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
            a, b, relax = params
            y_nor_fit = exp_func_inc(x, a, b, relax)
            labels.append(f'{growth_name}-index {i+1}:\ny=({np.round(a, 2)}t+{np.round(b, 2)})*(1-exp(-t/{np.round(relax, 2)}))')
            parameters.append((a, b, relax))
            losses.append((0, 0))
            y_nor_fit_failed = y_nor_fit

        else:
            # determine fitting function with simplified functions
            params, params_covariance = curve_fit(exp_func_inc_simp, x, y_nor, p0=p_init[1:], bounds=bounds, absolute_sigma=False) 
            b1, relax1 = params
            y1_nor_fit = exp_func_inc_simp(x, b1, relax1)

            params, params_covariance = curve_fit(exp_func_dec_simp, x, y_nor, p0=p_init[1:], bounds=bounds, absolute_sigma=False) 
            b2, relax2 = params
            y2_nor_fit = exp_func_dec_simp(x, b2, relax2)

            loss1 = ((y_nor - y1_nor_fit)**2).mean()
            loss2 = ((y_nor - y2_nor_fit)**2).mean()

            # calculate the real fitting parameters
            params, params_covariance = curve_fit(exp_func_inc, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
            a1, b1, relax1 = params
            y1_nor_fit = exp_func_inc(x, a1, b1, relax1)

            params, params_covariance = curve_fit(exp_func_dec, x, y_nor, p0=p_init, bounds=bounds, absolute_sigma=False) 
            a2, b2, relax2 = params
            y2_nor_fit = exp_func_dec(x, a2, b2, relax2)
            
            if loss1 < loss2:
                y_nor_fit = y1_nor_fit
                labels.append(f'{growth_name}-index {i+1}:\ny1=({np.round(a1, 2)}t+{np.round(b1, 2)})*(1-exp(-t/{np.round(relax1, 2)}))')
                parameters.append((a1, b1, relax1))
                y_nor_fit_failed = y2_nor_fit

            else:
                y_nor_fit = y2_nor_fit
                labels.append(f'{growth_name}-index {i+1}:\ny2=({np.round(a2, 2)}t+{np.round(b2, 2)})*(exp(-t/{np.round(relax2, 2)}))')
                parameters.append((a2, b2, relax2))
                y_nor_fit_failed = y1_nor_fit

            losses.append((loss1, loss2))

        y_fit = de_normalize_0_1(y_nor_fit, I_start, I_end, I_diff, unify)
        ys_fit.append(y_fit)
        ys_nor.append(y_nor)
        ys_nor_fit.append(y_nor_fit)
        ys_nor_fit_failed.append(y_nor_fit_failed)
    return np.array(parameters), [xs, ys, ys_fit, ys_nor, ys_nor_fit, ys_nor_fit_failed, labels, losses]


def analyze_curves(dataset, growth_dict, spot, metric, interval=1000, fit_settings={'step_size':5, 'prominence':0.1, 'length':500, 'savgol_window_order': (15,3), 'pca_component': 10, 'I_diff': 8000, 'unify':True, 'bounds':[0.01, 1], 'p_init':(1, 0.1)}):

    """
    Analyzes RHEED curves for a given spot and metric.

    Args:
        dataset (str): Name of the dataset.
        growth_dict (dict): Names of the growth index and corresponding frequency.
        spot (str): Name of the RHEED spot to collect, choice of "spot_1", "spot_2" or "spot_3".
        metric (str): Name of the metric to analyze the RHEED spot.
        interval (int, optional): Number of RHEED curves to analyze at a time. Defaults to 1000.
        fit_settings (dict, optional): Setting parameters for fitting function. Defaults to {'savgol_window_order': (15,3), 'pca_component': 10, 'I_diff': 8000, 'unify':True, 'bounds':[0.01, 1], 'p_init':(1, 0.1)}.

    Returns:
        tuple: A tuple containing the fitted parameters for all RHEED curves, the laser ablation counts for all RHEED curves, and a list of processed RHEED data.

    """

    parameters_all, x_list_all = [], []
    xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_failed_all = [], [], [], [], [], []
    labels_all, losses_all =  [], []
    
    x_end = 0
    for growth in list(growth_dict.keys()):

        # load data
        sample_x, sample_y = dataset.load_curve(growth, spot, metric, x_start=x_end)
        # sample_x, sample_y = load_curve(h5_para_file, growth_name, 'spot_2', 'img_intensity', camera_freq=500, x_start=0)

        # detect peaks
        x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=dataset.camera_freq, 
                                       laser_freq=growth_dict[growth], step_size=fit_settings['step_size'], prominence=fit_settings['prominence'])
        
        xs, ys = process_rheed_data(xs, ys, length=fit_settings['length'], savgol_window_order=fit_settings['savgol_window_order'], 
                                    pca_component=fit_settings['pca_component'])        

        # fit exponential function
        parameters, info = fit_exp_function(xs, ys, growth, fit_settings=fit_settings)        
        parameters_all.append(parameters)
        xs, ys, ys_fit, ys_nor, ys_nor_fit, ys_nor_fit_failed, labels, losses = info
        xs_all.append(xs)
        ys_all.append(ys)
        ys_fit_all+=ys_fit
        ys_nor_all+=ys_nor
        ys_nor_fit_all+=ys_nor_fit
        ys_nor_fit_failed_all+=ys_nor_fit_failed
        labels_all += labels
        losses_all += losses

        x_list = x_peaks[:-1] + x_end
        x_end = round(x_end + (len(sample_x)+interval)/dataset.camera_freq, 2)
        x_list_all.append(x_list)
        
    parameters_all = np.concatenate(parameters_all, 0)
    x_list_all = np.concatenate(x_list_all)[:len(parameters_all)]
    xs_all = np.concatenate(xs_all)
    ys_all = np.concatenate(ys_all)
    ys_nor_all = np.array(ys_nor_all)
    ys_nor_fit_all = np.array(ys_nor_fit_all)
    losses_all = np.array(losses_all)
    ys_nor_fit_all_failed = np.array(ys_nor_fit_failed_all)
    
    return parameters_all, x_list_all, [xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_all_failed, labels_all, losses_all]