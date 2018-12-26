from shutil import copyfile
import pandas as pd
from PIL import Image
from src import Helper as helper
import matplotlib.pyplot as plt

PLOT_FILE = './data/plot.png'


class Visualizer:
    def __init__(self):
        self.file = './data/visualize.gz'

        # Copy the original file
        copyfile('./data/stats.gz', self.file)

    def save_plot(self):

        # Load the stats Pickle
        df = pd.read_pickle(self.file, compression='gzip')

        # Draw and save the Plot
        plt.figure()
        plot = df.plot(x='Episode', y=['WinPct', 'LossPct', 'DrawPct'])
        fig = plot.get_figure()
        fig.savefig(PLOT_FILE)
        plt.close('all')

        # Close MS Photos (ugly)
        # todo: update
        helper.kill_process('Microsoft.Photos.exe')

        # Show the Plot
        img = Image.open(PLOT_FILE)
        img.show()


