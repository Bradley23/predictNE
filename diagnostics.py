#%%
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

#%%
class Report:
    def __init__(self, filename):
        # change default plot settings
        plt.rcParams.update({
            "font.size": 12
        })
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['ytick.minor.size'] = 3

        # initialize pdf
        self.pdf = PdfPages(filename)
    
    def save_fig(self, fig=None):
        if fig is None:
            fig = plt.gcf()
            self.pdf.savefig(fig)
            plt.close(fig)
        else:
            self.pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def add_title_page(self, title, info=None):
        fig = plt.figure(figsize=(8, 10))
        plt.axis('off')

        text = title + '\n\n'
        if info:
            for k, v in info.items():
                text += f"{k}: {v}\n"
        
        plt.text(0.05, 0.95, text, va='top', fontsize=14)
        self.save_fig(fig)

    def plot_loss(self, train_loss, val_loss, title=None):
        fig, ax = plt.subplots()
        ax.plot(train_loss, label='train')
        ax.plot(val_loss, label='validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # self.pdf.savefig(fig)

        return fig
        # plt.close(fig)

    def plot_pred(self, actual, predicted, r=None):
        N = actual.shape[0]
        fig, axes = plt.subplots(N, 1, figsize=(6, N+0.5))
        fig.text(0.4, 0.9, 'Actual', color='C0', ha='right')
        fig.text(0.5, 0.9, 'vs.', ha='center')
        fig.text(0.6, 0.9, 'Predicted', color='C1', ha='left')

        for i in range(N):
            axes[i].plot(actual[i], label='actual', linewidth=1)
            axes[i].plot(predicted[i], label='predicted', linewidth=1)
            # axes[i].legend()
            if r is not None:
                axes[i].text(1, 0.5, f"r = {r[i]:.2f}", va='center', ha='left', transform=axes[i].transAxes)

            axes[i].axis('off')
        
        return fig

    def close(self):
        self.pdf.close()