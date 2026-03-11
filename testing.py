#%%
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

#%%

x = np.linspace(0,10,100)

with PdfPages("test_plots.pdf") as pdf:
    y = np.sin(x)
    plt.figure()
    plt.plot(x, y)
    plt.title('Sine wave')
    pdf.savefig()
    plt.close()

    y = np.sin(x)*2
    plt.figure()
    plt.plot(x, y)
    plt.title('2 x Sine wave')
    pdf.savefig()
    plt.close()