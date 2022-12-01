import tkinter as tk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def logger(func):
    def wrapped_func(*args, **kwargs):
        print(func.__name__, 'start')
        func(*args, **kwargs)
        print(func.__name__, 'end')
    return wrapped_func


def lorentzian(x, intensity, X0, HWHM):
    return intensity * HWHM ** 2 / ((x - X0) ** 2 + HWHM ** 2)


class DataLoader:
    def __init__(self, filename, measurement):
        self.filename = filename
        self.measurement = measurement
        self.df_ref = None
        self.data_type = None
        self.need_center = False
        self.df_selected = None
        self.picked = self.zooming = False
        self.line_list = []
        self.rect_list = []

        self.magnification = 1

        self.load()
        self.check_shape()
        if self.need_center:
            self.input_center()

        self.run_GUI()

    def load(self):
        extension = self.filename.split('.')[-1]
        if extension in ['csv', 'asc']:
            self.df_ref = pd.read_csv(self.filename)
        else:
            print('Invalid extension.')

    def check_shape(self):
        if self.df_ref is None:
            return

        if self.df_ref.shape[0] == 1023:
            print('Andor Data')
            self.data_type = 'Andor'
        elif self.df_ref.shape[0] == 1014:
            print('Renishaw Data')
            self.data_type = 'Renishaw'
        else:
            print('Unexpected data type.')

        if self.df_ref.shape[1] == 1:
            self.need_center = True
            self.df_ref.columns = ['y']
        elif self.df_ref.shape[1] == 2:
            self.need_center = False
            self.df_ref.columns = ['x', 'y']

    def input_center(self):
        # center = int(input('Center Wavelength [nm]: '))
        # width = int(input('Wavelength Range [nm] (Default 130): '))
        center = 630
        width = 130
        x = np.linspace(center - width / 2, center + width / 2, self.df_ref.shape[0])
        self.df_ref['x'] = x
        self.df_ref.sort_index(axis='columns', inplace=True)

    def run_GUI(self):
        self.root = tk.Tk()
        frame = tk.Frame(master=self.root)
        frame.grid(row=0, column=0)
        button_submit = tk.Button(master=frame, text='OK', command=self.submit)
        button_submit.grid(row=1, column=0, sticky='NESW')
        fig=plt.figure(figsize=(6, 3))
        self.ax = fig.add_subplot(111)
        self.spec = self.ax.plot(self.df_ref['x'].values, self.df_ref['y'].values, color='k', linewidth=0.5)
        self.ax.set_ylim(self.df_ref['y'].values.min() * 0.9, self.df_ref['y'].values.max() * 1.1)
        self.canvas = FigureCanvasTkAgg(fig, frame)
        self.canvas.get_tk_widget().grid(row=0, column=0)
        fig.canvas.mpl_connect('scroll_event', self.zoom)
        fig.canvas.mpl_connect('pick_event', self.delete_line)
        fig.canvas.mpl_connect('button_press_event', self.add_line)
        fig.canvas.mpl_connect('motion_notify_event', self.reset_state)
        self.canvas.start_event_loop()

    @logger
    def add_line(self, event):
        if self.picked:
            return
        x_clicked = event.xdata
        line = self.ax.vlines(x_clicked, *self.ax.get_ylim(), color='red', linewidth=0.5, picker=5)
        self.line_list.append(line)
        self.ax.set_ylim(self.df_ref['y'].values.min() * 0.9, self.df_ref['y'].values.max() * 1.1)
        self.update_canvas()

    @logger
    def zoom(self, event):
        print(event)
        self.picked = False
        self.zooming = True
        if event.button == 'down':
            self.magnification = 1.2
        elif event.button == 'up':
            self.magnification = 0.8
        else:
            return
        x = event.xdata
        xmin, xmax = self.ax.get_xlim()
        new_xmin = x - (x - xmin) / self.magnification
        new_xmax = (xmax - x) / self.magnification + x
        self.ax.set_xlim(new_xmin, new_xmax)
        self.update_canvas()
        print(len(self.line_list))

    @logger
    def delete_line(self, event):
        self.picked = True
        self.line_list.remove(event.artist)
        event.artist.remove()
        # self.update_canvas()

    @logger
    def update_canvas(self):
        self.picked = self.zooming = False
        # 選択範囲の描画
        for rect in self.rect_list:
            rect.remove()
        self.rect_list = []
        self.line_list.sort(key=lambda x: x.properties()['segments'][0][0][0])
        for i, line2 in enumerate(self.line_list):
            if i % 2 == 1:
                line1 = self.line_list[i - 1]
                x1 = line1.properties()['segments'][0][0][0]
                x2 = line2.properties()['segments'][0][0][0]
                width = x2 - x1
                y1, y2 = self.ax.get_ylim()
                height = y2 - y1
                r = patches.Rectangle(xy=(x1, y1), width=width, height=height, fc='red', alpha=0.3)
                self.ax.add_patch(r)
                self.rect_list.append(r)
        self.canvas.draw()

    def reset_state(self, event):
        # self.picked = False
        pass

    def submit(self):
        self.root.destroy()

        peak_detected = []
        for i, line2 in enumerate(self.line_list):
            if i % 2 == 1:
                line1 = self.line_list[i - 1]
                x1 = line1.properties()['segments'][0][0][0]
                x2 = line2.properties()['segments'][0][0][0]
                df_selected = self.df_ref[(x1 < self.df_ref['x']) & (self.df_ref['x'] < x2)]

                x = df_selected['x'].values
                y = df_selected['y'].values
                indices_found, _ = find_peaks(y, prominence=40, distance=5)
                if len(indices_found) == 0:
                    print(f'Failed to detect around {x.min()} ~ {x.max()}')
                    continue
                pini = [y[indices_found[0]], x[indices_found[0]], 1]
                popt, pcov = curve_fit(lorentzian, x, y, p0=pini)
                peak_detected.append(popt[1])
        print(peak_detected)
        # self.pf = PolynomialFeatures(degree=3)
        # x_detected_cubic = self.pf.fit_transform(x_detected_t)
        #
        # model = LinearRegression()  # 線形回帰
        # model.fit(x_detected_cubic, self.peaks_target)




def main():
    filename = '/Volumes/GoogleDrive/共有ドライブ/Laboratory/Individuals/kaneda/Data_M1/220616/data_630/calibration_630.asc'
    # filename = '/Volumes/GoogleDrive/共有ドライブ/Laboratory/Individuals/kaneda/Data_M1/221013/Raman/BNCNT532_x20_1_247__X_0.1__Y_-94.5425__Time_421.asc'
    dl = DataLoader(filename, 'Rayleigh')


if  __name__ == '__main__':
    main()
