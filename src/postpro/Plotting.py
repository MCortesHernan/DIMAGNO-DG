'''Class with all plots as methods.'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.ticker import AutoLocator, FormatStrFormatter, LogFormatter, LogLocator, StrMethodFormatter
import os
import numpy as np
import matplotlib.pyplot as plt

import nmmn.plots 
import matplotlib.colors as colors
parula = nmmn.plots.parulacmap()

class plot_class:
    '''This class defines methods to make different kinds of plots.'''

    def __init__(self, dim, title,
                 z_title=r'$z$ [cm]', x_title=r'$r$ [cm]',
                 fig=None, ax=None):

        plt.rcParams['font.family'] = 'sans-serif'
        self.dim = dim
        self.title = title
        self.z_title = z_title
        self.x_title = x_title

        # Use existing fig/ax or create new
        if fig is None or ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = fig
            self.ax = ax
        if dim == 2:
            ax.set_box_aspect(1)

        for label in self.ax.get_xticklabels():
            label.set_fontsize(30)

        for label in self.ax.get_yticklabels():
            label.set_fontsize(30)

        self.ax.tick_params(labelsize=30)

 

        self.ax.xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
        self.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:g}'))

    def plot_map(self, X, Z, data, clip=None, log=False, add_colorbar=False, cax=None, write_label_x=False, write_label_y=False):
        '''2D colormap. Optionally attach colorbar.'''

        data = np.array(data, copy=True)

        # Sanitize
        data[~np.isfinite(data)] = np.nan

        if log:
            data[data <= 0] = np.nan
            if clip is None:
                vmin = np.nanmin(data)
                vmax = np.nanmax(data)
                normalize = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                normalize = colors.LogNorm(clip[0], clip[1])
        else:
            if clip is None:
                normalize = None
            else:
                normalize = colors.Normalize(clip[0], clip[1])

        pcol = self.ax.pcolormesh(X, Z, data, cmap=parula,
                                  norm=normalize, shading='auto')

        if add_colorbar:
            cbar = self.fig.colorbar(pcol, cax=cax)
            cbar.ax.tick_params(labelsize=30)

            if self.title is not None:
                self.ax.set_title(self.title, fontsize=30, pad=10)
        
        self.ax.tick_params('x', direction= 'in', length = 5, top = True)
        self.ax.tick_params('y', direction= 'in', length = 5, right = True)
        self.ax.set_xlim(np.nanmin(X), np.nanmax(X))
        self.ax.set_ylim(np.nanmin(Z), np.nanmax(Z))
        
        if write_label_x:
            self.ax.set_xlabel(self.z_title, fontsize=30)
        if write_label_y:
            self.ax.set_ylabel(self.x_title, fontsize=30)

        return pcol

    def plot_line(self, x_s, magnitude, label, color=['black'], linestyle=['solid'], linewidth=[1.], logy=False, write_label_x=False, write_label_y=False, add_legend=True, lim_x=None, lim_y=None):
        if not isinstance(magnitude, list):
            magnitude = [magnitude]
            label = [label]

        for i in range(len(magnitude)):
            self.ax.plot(x_s, magnitude[i], label=label[i], color=color[i], linestyle=linestyle[i], linewidth=linewidth[i])
        
        if logy:
            # Avoid zeros or negatives
            all_values = np.hstack(magnitude)
            min_val = max(np.min(all_values[all_values > 0]), 1e-12)  # avoid log(0)
            max_val = np.max(all_values)

            self.ax.set_yscale('log')
            self.ax.set_ylim(min_val, max_val)

            # Format y-axis labels as powers of 10
            from matplotlib.ticker import LogFormatterMathtext

            formatter = LogFormatterMathtext(base=10)
            self.ax.yaxis.set_major_formatter(formatter)

        self.ax.tick_params('x', direction= 'in', length = 5, top = True)
        self.ax.tick_params('y', direction= 'in', length = 5, right = True)

        if write_label_x:
            self.ax.set_xlabel(self.z_title, fontsize=30)
        if write_label_y:
            self.ax.set_ylabel(self.x_title, fontsize=30)

        if add_legend:
            self.ax.legend(fontsize=30)

        #set lims
        if lim_y is not None:
            self.ax.set_ylim(lim_y)
        else:
            self.ax.set_ylim([np.min(magnitude), np.max(magnitude) + 0.06*np.max(magnitude)])
        if lim_x is not None:
            self.ax.set_xlim(lim_x)
        else:
            self.ax.set_xlim([np.min(x_s), np.max(x_s)])


def Scalar_Colormap(Z, X, scalar_quantity, levels, contour, cont_lev = [-1,0,1], title = None, set_ticks = None, tick_labels = None):

    scalar_quantity_masked = np.clip(scalar_quantity, levels[0], levels[-1])

    cmap = 'viridis'
    fig = plt.figure(figsize = (10, 8))
    ax = fig.add_axes([0, 0, 1.2, 0.8])
    pcol = ax.contourf(Z,X, scalar_quantity_masked, levels, cmap = cmap)
    cbar = fig.colorbar(pcol)

    if set_ticks is not None:
        cbar.set_ticks(set_ticks)
        cbar.set_ticklabels(tick_labels)  # Show actual values
    
    if contour is not None:
        plt.contour(Z,X ,contour, [cont_lev[0]], colors = 'black',
        linestyles = 'solid', linewidths = 2.)
        plt.contour(Z,X ,contour, [cont_lev[len(cont_lev)//2]], colors = 'black',
        linestyles = 'solid', linewidths = 2.)
        plt.contour(Z,X ,contour, [0], colors = 'black',
        linestyles = 'dashed', linewidths = 2.) #plot the separatrix
        plt.contour(Z,X ,contour, [cont_lev[-1]], colors = 'black',
        linestyles = 'solid', linewidths = 2.)
        plt.contour(Z,X ,contour, cont_lev[1:-1], colors = 'black',
        linestyles = 'solid', linewidths = 1.)

    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r'z', fontsize = 18)
    ax.set_ylabel(r'x', fontsize = 18)
    ax.set_title(title, fontsize = 18)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=16)  # Increase font size

    ax.minorticks_on()  # Enable minor ticks on both axes
    import matplotlib.ticker as ticker
    # Set minor tick intervals
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # Adds 2 minor ticks between major ticks
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))  # Adds 2 minor ticks between major ticks

    # Customize minor tick appearance
    ax.tick_params(which='minor', length=4, width=1, color='black', direction = 'in')  # Smaller, lighter ticks


    ax.tick_params(axis='both', labelsize=16, width=2, length=8, color='black', direction='in')  # Customize ticks
    ax.tick_params(bottom=True, top=True, left=True, right=True)  # Enable ticks on all sides
    ax.tick_params(which='minor', bottom=True, top=True, left=True, right=True)  # Enable minor ticks on all sides
    ax.spines['top'].set_visible(True)  # Show top border
    ax.spines['right'].set_visible(True)  # Show right border

    return fig

def Add_Stream_Plot(Z, X, v_z, v_x, seed = None, color = 'magn', cb = True, cbar_label = None):

    '''
    Add a streamplot from z and x components of a vector field to an existing fig object:

    - seed: [N, 2] list with the starting points of the lines.

    - color: a string with one of the colors supported by matplotlib, see
    (https://matplotlib.org/stable/tutorials/colors/colors.html). Default value
    is a varying color depicting the magnitude of the vector field.

        - cb: Add a colorbar with the values of the magnitude of the vector field.

            - cbar_label: Add a label to the colorbar of the magnitude of field.
    '''
    if color == 'magn':
        color = np.sqrt(v_z**2 + v_x**2)

    if seed is None:
        stream = plt.streamplot(Z, X, v_z, v_x, color= color, 
        cmap= cm.rainbow, linewidth= 1.4, density= 1)
    else:
        stream = plt.streamplot(Z, X, v_z, v_x, start_points=seed,
        color= color, cmap= cm.rainbow, linewidth= 1.4, density= 30)

    if cb == True:
        cbar = plt.colorbar(stream.lines)
        cbar.set_label(cbar_label, fontsize = 12)
