import glob, re
import numpy as np
import matplotlib.pyplot as plt
from m3_learning.viz.layout import layout_fig
from m3_learning.RHEED.Viz import Viz
from m3_learning.RHEED.Analysis import detect_peaks, process_rheed_data , fit_exp_function
seq_colors = ['#00429d','#2e59a8','#4771b2','#5d8abd','#73a2c6','#8abccf','#a5d5d8','#c5eddf','#ffffe0']
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


def fit_curves(xs, ys, x_peaks, sample_x, fit_settings):

    x_end = 0
    length_list = []
    for xi in xs:
        length_list.append(len(xi))

    xs, ys = process_rheed_data(xs, ys, length=int(np.mean(length_list)), savgol_window_order=fit_settings['savgol_window_order'], 
                                pca_component=fit_settings['pca_component'])        

    parameters_all, x_list_all = [], []
    xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_failed_all = [], [], [], [], [], []
    labels_all, losses_all =  [], []

    # fit exponential function
    parameters, info = fit_exp_function(xs, ys, growth_name='growth_1', fit_settings=fit_settings)        
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


def analyze_rheed_data(data, camera_freq, laser_freq, detect_param={'step_size':3, 'prominence':10}, viz_curves=False, 
                       viz_fittings=False, viz_ab=False, n_std=3, trim_first=0, fit_settings={'savgol_window_order': (15, 3), 'pca_component': 10, 'I_diff': None, 
                                                                                              'unify':False, 'bounds':[0.001, 1], 'p_init':[0.1, 0.4, 0.1]}):
    if isinstance(data, str):
        data = np.loadtxt(data)
    sample_x, sample_y = data[:,0], data[:,1]
    
    step_size = detect_param['step_size']
    prominence = detect_param['prominence']

    # plt.plot(sample_x, sample_y)
    # plt.show()
    
    x_peaks, xs, ys = detect_peaks(sample_x, sample_y, camera_freq=camera_freq, laser_freq=laser_freq, step_size=step_size, prominence=prominence)
    
    if trim_first != 0:
        xs_trimed, ys_trimed = [], []
        for x, y in zip(xs, ys):
            ys_trimed.append(y[trim_first:])
            xs_trimed.append(np.linspace(x[0], x[-1], len(y[trim_first:])))
        xs, ys = xs_trimed, ys_trimed
    
    if viz_curves:
        xs_sample, ys_sample = xs[::1], ys[::1]
        fig, axes = layout_fig(len(ys_sample), mod=6, figsize=(12,2*len(ys_sample)//6+1), layout='compressed')
        Viz.show_grid_plots(axes, xs_sample, ys_sample, labels=None, xlabel=None, ylabel=None, ylim=None, legend=None, color=None)

    parameters_all, x_list_all, info = fit_curves(xs, ys, x_peaks, sample_x, fit_settings)
    [xs_all, ys_all, ys_fit_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_failed_all, labels_all, losses_all] = info
    
    if viz_fittings:
        fig, axes = layout_fig(len(xs_all), mod=6, layout='compressed')
        # Viz.plot_fit_details(xs_all, ys_nor_all, ys_nor_fit_all, ys_nor_fit_failed_all, index_list=range(len(xs_all)), figsize=(12,0.3*len(x_peaks)//6+1))
        Viz.show_grid_plots(axes, xs_all, ys_nor_all, ys_nor_fit_all, labels=labels_all, xlabel=None, ylabel=None, ylim=None, legend=None, color=None)

    # n_std = 3
    tau = parameters_all[:,2]/laser_freq
    x_clean = x_list_all[np.where(tau < np.mean(tau) + n_std*np.std(tau))[0]]
    tau = tau[np.where(tau < np.mean(tau) + n_std*np.std(tau))[0]]

    x_clean = x_clean[np.where(tau > np.mean(tau) - n_std*np.std(tau))[0]]
    tau = tau[np.where(tau > np.mean(tau) - n_std*np.std(tau))[0]]
    # print('mean of tau:', np.mean(tau))
    
    if viz_ab:
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


def plot_activation_energy(temp_list, tau_list, ylim1=None, ylim2=None, save_path=None):
    tau_mean_list = [np.mean(t_list) for t_list in tau_list]

    fig, axes = plt.subplots(1, 2, figsize=(8,2.5))

    tau_mean = np.array(tau_mean_list)

    axes[0].scatter(temp_list, tau_mean, color='k', s=10)
    axes[0].set_xlabel('T (C)')
    axes[0].set_ylabel('tau (s)')
    if ylim1 is not None:
        axes[0].set_ylim(*ylim1)

    T = np.array(temp_list) + 273
    x = 1/(T)
    y = -np.log(tau_mean)
    m, b = np.polyfit(x, y, 1)

    axes[1].scatter(x, y, color='k', s=10)
    axes[1].plot(x, y, 'yo', x, m*x+b, '--k')
    axes[1].set_xlabel('1/T (1/K))')
    axes[1].set_ylabel(r'-ln($\tau$)')
    # axes[1].set_title('Ea=' + str(round(m*-8.617e-5, 2)) + ' eV')
    if ylim2 is not None:
        axes[1].set_ylim(*ylim2)

    text = f'Ea={round(m*-8.617e-5, 2)}eV'
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="white", facecolor="white")
    axes[1].text(0.25, 0.1, text, transform=axes[1].transAxes, fontsize=10, verticalalignment="center", horizontalalignment="center", bbox=bbox_props)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()