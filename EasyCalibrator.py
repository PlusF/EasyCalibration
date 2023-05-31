import numpy as np
from dataloader import DataLoader
from calibrator import Calibrator


class EasyCalibrator(Calibrator, DataLoader):
    def __init__(self, *args, **kwargs):
        Calibrator.__init__(self, *args, **kwargs)
        DataLoader.__init__(self)
        self.filename_raw_list = []
        self.filename_ref = ''

        self.set_measurement('Raman')
        self.set_material('sulfur')

        self.center: float = 630
        self.wavelength_range = 134

    def load_raw_list(self, filenames):
        ok_dict = self.load_files(filenames)
        for filename, success in ok_dict.items():
            if success:
                self.filename_raw_list.append(filename)

    def load_ref(self, filename):
        self.filename_ref = filename
        self.load_file(filename)
        self.set_data(self.spec_dict[filename].xdata, self.spec_dict[filename].ydata)

    def set_initial_xdata(self, center: float):
        self.center = center
        self.xdata = np.linspace(center - self.wavelength_range / 2,
                                 center + self.wavelength_range / 2,
                                 self.spec_dict[self.filename_ref].xdata.shape[0])

    def reset_data(self):
        if self.filename_raw_list == [] or self.filename_ref == '':
            raise ValueError('Load data before reset.')
        self.set_data(self.spec_dict[self.filename_ref].xdata, self.spec_dict[self.filename_ref].ydata)

    def save_raw_files(self):
        for filename in self.filename_raw_list:
            self.save(filename)

