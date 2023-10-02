import glob, re
import numpy as np
import matplotlib.pyplot as plt
from m3_learning.viz.layout import layout_fig
from m3_learning.RHEED.Viz import Viz
from m3_learning.RHEED.Analysis import detect_peaks, process_rheed_data , fit_exp_function, process_curves, remove_linear_bg
seq_colors = ['#00429d','#2e59a8','#4771b2','#5d8abd','#73a2c6','#8abccf','#a5d5d8','#c5eddf','#ffffe0']
import numpy as np
import matplotlib.pyplot as plt

import csv
def read_txt_to_numpy(filename):
    # Load data using numpy.loadtxt
    data = np.loadtxt(filename, dtype=float, skiprows=1, comments=None)

    # Extract header from the first row
    with open(filename, 'r') as file:
        header = file.readline().strip().split()
    return header, data


def select_range(data, start, end):
    x = data[:,0]
    y = data[:,1]
    x_selected = x[(x>start) & (x<end)]
    y_selected = y[(x>start) & (x<end)]
    data = np.stack([x_selected, y_selected], 1)
    return data


def fit_curves(xs, ys, x_peaks, sample_x, normalize_params):

    x_end = 0
    parameters_all, x_list_all = [], []
    xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_failed_all = [], [], [], [], [], []
    labels_all, losses_all =  [], []

    # fit exponential function
    parameters, info = fit_exp_function(xs, ys, growth_name='sample_growth', fit_settings=normalize_params)        
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
    x_end = round(x_end + (len(sample_x)+0)/30, 2)
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


def analyze_rheed_data(data, camera_freq, laser_freq, 
        denoise_params = {'savgol_window_order': (51,3), 'pca_component': None, 'fft_cutoff_order':(20, 1), 'median_kernel_size':51},
        curve_params = {'trim_first':0, 'tune_tail':True, 'convolve_step':5, 'prominence':0.8, 'mode':'full', 'linear_ratio':0.8},
        normalize_params = {'I_diff': None, 'unify':True, 'bounds':[0.01, 1], 'p_init':(1, 0.1, 0.4), 'n_std':1},
        viz_params = {'viz_denoise': True, 'viz_curves': False, 'viz_fittings': False, 'viz_ab': False}):

    if isinstance(data, str):
        data = np.loadtxt(data)
    sample_x, sample_y = data[:,0], data[:,1]
    
    # denoise
    sample_x, sample_y = process_rheed_data(sample_x, sample_y, camera_freq, denoise_params, viz_params['viz_denoise'])       

    # detect peaks
    x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=camera_freq, laser_freq=laser_freq, curve_params=curve_params)

    # denoise
    xs, ys = process_curves(xs, ys, curve_params)        

    # remove linear background
    xs, ys = remove_linear_bg(xs, ys, linear_ratio=curve_params['linear_ratio'])
    
    if viz_params['viz_curves']:
        xs_sample, ys_sample = xs[::5], ys[::5]
        fig, axes = layout_fig(len(ys_sample), mod=6, figsize=(12,2*len(ys_sample)//6+1), layout='compressed')
        Viz.show_grid_plots(axes, xs_sample, ys_sample, labels=None, xlabel=None, ylabel=None, ylim=None, legend=None, color=None)

    # fit exponential function
    parameters_all, x_list_all, info = fit_curves(xs, ys, x_peaks, sample_x, normalize_params)
    [xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_failed_all, labels_all, losses_all] = info
    
    if viz_params['viz_fittings']:
        Viz.plot_fit_details(xs_all[::5], ys_nor_all[::5], ys_nor_fit_all[::5], None, labels=labels_all[::5], 
                            mod=5, figsize=(12, 1.5*len(x_peaks[::5])//4+1), style='presentation')
        
    # remove outliers
    n_std = normalize_params['n_std']
    x_list_all = np.array(x_list_all) + sample_x[0]

    tau = parameters_all[:,2]/laser_freq
    x_clean = x_list_all[np.where(tau < np.mean(tau) + n_std*np.std(tau))[0]]
    tau = tau[np.where(tau < np.mean(tau) + n_std*np.std(tau))[0]]

    x_clean = x_clean[np.where(tau > np.mean(tau) - n_std*np.std(tau))[0]]
    tau = tau[np.where(tau > np.mean(tau) - n_std*np.std(tau))[0]]
    # print('mean of tau:', np.mean(tau))
    
    if viz_params['viz_ab']:
        fig, axes = layout_fig(4, 1, figsize=(12, 3*4))
        Viz.plot_curve(axes[0], sample_x, sample_y, plot_type='lineplot', xlabel='Time (s)', ylabel='Intensity (a.u.)', yaxis_style='sci')
        Viz.plot_curve(axes[1], x_list_all, parameters_all[:,0], plot_type='lineplot', xlabel='Time (s)', ylabel='Fitted a (a.u.)')
        Viz.plot_curve(axes[2], x_list_all, parameters_all[:,1], plot_type='lineplot', xlabel='Time (s)', ylabel='Fitted b (a.u.)')
        Viz.plot_curve(axes[3], x_clean, tau, plot_type='lineplot', xlabel='Time (s)', ylabel='Characteristic Time (s)')
        plt.show()

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 2.5), layout='compressed')
    ax1.scatter(sample_x, sample_y, color='k', s=1)
    Viz.set_labels(ax1, xlabel='Time (s)', ylabel='Intensity (a.u.)', ticks_both_sides=False)

    ax2 = ax1.twinx()
    ax2.scatter(x_clean, tau, color=seq_colors[0], s=3)
    ax2.plot(x_clean,  tau, color='#bc5090', markersize=3)
    Viz.set_labels(ax2, ylabel='Characteristic Time (s)', yaxis_style='lineplot', ticks_both_sides=False)
    ax2.tick_params(axis="y", color='k', labelcolor=seq_colors[0])
    ax2.set_ylabel('Characteristic Time (s)', color=seq_colors[0])
    plt.title('mean of tau: '+str(np.mean(tau)))
    plt.show()
    return parameters_all, x_list_all, info, tau


def plot_activation_energy(temp_list, tau_list, save_path=None):
    tau_mean_list = [np.mean(t_list) for t_list in tau_list]

    fig, axes = plt.subplots(1, 2, figsize=(8,2.5))

    tau_mean = np.array(tau_mean_list)

    axes[0].scatter(temp_list, tau_mean, color='k', s=10)
    axes[0].set_xlabel('T (C)')
    axes[0].set_ylabel('tau (s)')
    # axes[0].set_ylim(0,0.2)

    T = np.array(temp_list) + 273
    x = 1/(T)
    y = -np.log(tau_mean)
    m, b = np.polyfit(x, y, 1)

    axes[1].scatter(x, y, color='k', s=10)
    axes[1].plot(x, y, 'yo', x, m*x+b, '--k')
    axes[1].set_xlabel('1/T (1/K))')
    axes[1].set_ylabel(r'-ln($\tau$)')
    # axes[1].set_title('Ea=' + str(round(m*-8.617e-5, 2)) + ' eV')
    # axes[1].set_ylim(1.8,2.5)

    text = f'Ea={round(m*-8.617e-5, 2)}eV, b={b}'
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white")
    # axes[1].text(0.25, 0.1, text, transform=axes[1].transAxes, fontsize=10, verticalalignment="center", horizontalalignment="center", bbox=bbox_props)
    plt.title(text)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()