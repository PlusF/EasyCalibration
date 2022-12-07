import os
import json
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class MainWindow(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        with open('./data/data.json', 'r') as f:
            self.database = json.load(f)

        self.df_ref = None
        self.dict_df = {}
        self.dict_df_calibrated = {}

        self.create_widgets()

    def create_widgets(self):
        self.width = 300
        self.height = 200
        dpi = 100
        fig = plt.figure(figsize=(self.width / dpi, self.height / dpi), dpi=dpi)
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, self.master)
        self.canvas.get_tk_widget().grid(row=1, column=1)

        frame_msg = tk.LabelFrame(self.master, text='Message', width=self.width, height=self.height)
        frame_ref = tk.LabelFrame(self.master, text='Reference', width=self.width, height=self.height)
        frame_help = tk.LabelFrame(self.master, text='Help', width=self.width, height=self.height)
        frame_before = tk.LabelFrame(self.master, text='Data to calibrate', width=self.width, height=self.height)
        frame_after = tk.LabelFrame(self.master, text='Calibrated data', width=self.width, height=self.height)
        frame_msg.grid(row=0, column=0)
        frame_ref.grid(row=0, column=1)
        frame_help.grid(row=0, column=2)
        frame_before.grid(row=1, column=0)
        frame_after.grid(row=1, column=2)
        frame_msg.pack_propagate(False)
        frame_ref.grid_propagate(False)
        frame_help.pack_propagate(False)
        frame_before.pack_propagate(False)
        frame_after.pack_propagate(False)

        # frame_msg
        self.msg = tk.StringVar(value='This is for Renishaw Raman data for now.\nSulfur, naphthalene and acetonitrile are supported.')
        label_msg = tk.Label(master=frame_msg, textvariable=self.msg)
        label_msg.pack()

        # frame_ref
        self.filename_ref = tk.StringVar()
        entry_ref = tk.Entry(frame_ref, textvariable=self.filename_ref, width=40)
        label_material = tk.Label(frame_ref, text='material:')
        self.material = tk.StringVar(value=list(self.database.keys())[0])
        optionmenu_material = tk.OptionMenu(frame_ref, self.material, *self.database.keys())
        label_degree = tk.Label(frame_ref, text='degree:')
        self.degree = tk.StringVar(value='1 (Linear)')
        optionmenu_degree = tk.OptionMenu(frame_ref, self.degree, '1 (Linear)', '2 (Quadratic)', '3 (Cubic)')
        self.button_train = tk.Button(frame_ref, text='TRAIN', command=self.train, state=tk.DISABLED)
        entry_ref.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
        label_material.grid(row=1, column=0, sticky=tk.EW)
        optionmenu_material.grid(row=1, column=1, sticky=tk.EW)
        label_degree.grid(row=2, column=0, sticky=tk.EW)
        optionmenu_degree.grid(row=2, column=1, sticky=tk.EW)
        self.button_train.grid(row=3, column=0, columnspan=2, sticky=tk.EW)

        # frame_msg
        self.help = tk.StringVar(value='sulfur: 86 ~ 470 cm-1\nnaphthalene: 514 ~ 1576 cm-1\nacetonitrile: 2254 ~ 2940 cm-1\n\n参照ピークが・・・\n2本か3本しかないとき:Linear\n中心波長付近にあるとき: Quadratic\n全体に分布しているとき: Cubic')
        label_help = tk.Label(master=frame_help, textvariable=self.help)
        label_help.pack()

        # frame_before
        self.listbox_before = tk.Listbox(frame_before, height=8, width=40)
        self.button_calibrate = tk.Button(frame_before, text='CALIBRATE', command=self.calibrate, state=tk.DISABLED)
        self.listbox_before.pack()
        self.button_calibrate.pack()

        # frame_after
        self.listbox_after = tk.Listbox(frame_after, height=8, width=40)
        self.button_download = tk.Button(frame_after, text='DOWNLOAD', command=self.download, state=tk.DISABLED)
        self.listbox_after.pack()
        self.button_download.pack()

    def train(self):
        self.ax.clear()

        x_ref_true_list = self.database[self.material.get()]
        # 範囲外のピークは除外
        x_ref_true_list = np.array(x_ref_true_list)
        x_min, x_max = self.df_ref.x.min(), self.df_ref.x.max()
        x_ref_true_list = x_ref_true_list[(x_ref_true_list > x_min) & (x_ref_true_list < x_max)]
        # ピークを中心に十分広い範囲を探索する
        peak_ranges = [[x-10, x+10] for x in x_ref_true_list]

        # find peak from range nearby the right peaks
        first_index_list = []
        found_peak_list = []
        for peak_range in peak_ranges:
            partial = (peak_range[0] < self.df_ref.x) & (self.df_ref.x < peak_range[1])
            first_index = np.where(partial)[0][0]  # 切り取った範囲の開始インデックス
            first_index_list.append(first_index)

            df_partial = self.df_ref[partial]
            found_peaks, properties = find_peaks(df_partial.y, prominence=50)
            if len(found_peaks) != 1:
                self.msg.set('Training failed. Check the file or the condition.')
                return
            found_peak_list.append(found_peaks[0] + first_index)

        self.ax.plot(self.df_ref.x, self.df_ref.y, color='k')
        ymin, ymax = self.ax.get_ylim()
        for peak_range in peak_ranges:
            self.ax.vlines(peak_range[0], ymin, ymax, color='k')
            self.ax.vlines(peak_range[1], ymin, ymax, color='k')

        for first_index, found_peak in zip(first_index_list, found_peak_list):
            self.ax.vlines(self.df_ref.x[found_peak], ymin, ymax, color='r', linewidth=1)

        x_ref_extracted = self.df_ref.x.loc[found_peak_list]
        self.pf = PolynomialFeatures(degree=int(self.degree.get()[0]))
        x_ref_extracted_poly = self.pf.fit_transform(x_ref_extracted.values.reshape(-1, 1))

        self.lr = LinearRegression()
        self.lr.fit(x_ref_extracted_poly, np.array(x_ref_true_list).reshape(-1, 1))

        self.canvas.draw()
        self.button_calibrate.config(state=tk.ACTIVE)
        self.msg.set('Successfully trained.\nYou can now calibrate.')

    def calibrate(self):
        for filename, df in self.dict_df.items():
            x = self.pf.fit_transform(df.x.values.reshape(-1, 1))
            x_calibrated = self.lr.predict(x)
            data = np.hstack([x_calibrated, df.y.values.reshape(-1, 1)])
            df_calibrated = pd.DataFrame(data=data, columns=['x', 'y'])

            filename_new = '.'.join(filename.split('.')[:-1]) + f'_calibrated_by_{self.material.get()}.' + filename.split('.')[-1]

            self.dict_df_calibrated[filename_new] = df_calibrated

        self.update_listbox()
        self.button_download.config(state=tk.ACTIVE)

        self.msg.set('Successfully calibrated.\nYou can now download the calibrated data.')

    def drop(self, event=None):
        master_geometry = list(map(int, self.master.winfo_geometry().split('+')[1:]))
        dropped_place = ((event.y_root - master_geometry[1]) // self.height, (event.x_root - master_geometry[0]) // self.width)

        filenames = [f.replace('{', '').replace('}', '') for f in event.data.split('} {')]
        loaded_df_list = self.load(filenames)
        if dropped_place == (0, 1):  # reference data
            self.filename_ref.set(filenames[0])
            self.df_ref = loaded_df_list[0]
            self.button_train.config(state=tk.ACTIVE)
        elif dropped_place == (1, 0):  # data to calibrate
            for filename, df in zip(filenames, loaded_df_list):
                self.dict_df[filename] = df
            self.update_listbox()

    def load(self, filenames):
        msg = ''
        df_list = []
        for filename in filenames:
            df = pd.read_csv(filename, sep='[:\t]', header=None, engine='python')
            if df.shape[0] > 1024:  # sifからbatch conversionで変換したascファイルのとき
                msg += 'Solis data loaded.\n'
                df = df.loc[26:, 0:1]
                df = df.reset_index(drop=True)
            elif df.shape[0] == 1024:  # AutoRayleighで出力したascファイルのとき
                msg += 'RAS data loaded.\n'
            elif df.shape[0] == 1015:
                msg += 'Renishaw Raman data loaded.\n'
            else:
                msg += 'Unsupported file．\n'
            df.columns = ['x', 'y']
            df = df.astype(float)

            msg += f'directory: {os.path.dirname(filename)}\n'
            msg += f'filename: {os.path.basename(filename)}\n'

            df_list.append(df)

        self.msg.set(msg)

        return df_list

    def update_listbox(self):
        self.listbox_before.delete(0, tk.END)
        for filename in self.dict_df.keys():
            self.listbox_before.insert(0, filename)

        self.listbox_after.delete(0, tk.END)
        for filename in self.dict_df_calibrated.keys():
            self.listbox_after.insert(0, filename)

    def delete_from_listbox(self):
        pass

    def download(self):
        msg = ''
        for filename, df in self.dict_df_calibrated.items():
            df.to_csv(filename, sep='\t', index=False, header=False)
            msg += 'Successfully downloaded.\n'
            msg += f'directory: {os.path.dirname(filename)}\n'
            msg += f'filename: {os.path.basename(filename)}\n'
        self.msg.set(msg)

    def quit(self):
        self.master.quit()


def main():
    root = TkinterDnD.Tk()
    app = MainWindow(master=root)
    root.protocol('WM_DELETE_WINDOW', app.quit)
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', app.drop)
    app.mainloop()


if __name__ == '__main__':
    main()
