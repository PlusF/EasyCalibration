import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def main():
    # Raman
    filename_ref = input('reference spectrum:')
    # filename_raw = input('input spectrum:')
    df_ref = pd.read_csv(filename_ref, sep='\t', header=None)
    # df_raw = pd.read_csv(filename_raw, sep='\t', header=None)
    df_ref.columns = ['x', 'y']
    # df_raw.columns = ['x', 'y']

    data_id = {
        0: 'sulfur',
        1: 'naphthalene',
        2: 'acetonitrile'
    }
    database = {
        'sulfur': [85.1, 153.8, 219.1, 473.2],
        'naphthalene': [513.8, 763.8, 1021.6, 1147.2, 1382.2, 1464.5, 1576.6, 3056.4],
        'acetonitrile': [2253.7, 2940.8]
    }
    x_ref_true_list = database[data_id[int(input('(sulfur: 0, naphthalene: 1, acetonitrile: 2):'))]]
    peak_ranges = [[x-10, x+10] for x in x_ref_true_list]

    # find peak from range nearby the right peaks
    first_index_list = []
    found_peak_list = []
    for peak_range in peak_ranges:
        partial = (peak_range[0] < df_ref.x) & (df_ref.x < peak_range[1])
        first_index = np.where(partial == True)[0][0]  # 切り取った範囲の開始インデックス
        first_index_list.append(first_index)

        df_partial = df_ref[partial]
        found_peaks, properties = find_peaks(df_partial.y, prominence=50)
        if len(found_peaks) != 1:
            print('failed to find peak')
        found_peak_list.append(found_peaks[0] + first_index)

    plt.plot(df_ref.x, df_ref.y, color='k')
    ymin, ymax = plt.ylim()
    for peak_range in peak_ranges:
        plt.vlines(peak_range[0], ymin, ymax, color='k')
        plt.vlines(peak_range[1], ymin, ymax, color='k')

    for first_index, found_peak in zip(first_index_list, found_peak_list):
        plt.vlines(df_ref.x[found_peak], ymin, ymax, color='r', linewidth=1)

    # plt.show()

    x_ref_extracted = df_ref.x.loc[found_peak_list]
    degree = int(input('calibration dimension(1~3):'))
    pf = PolynomialFeatures(degree=degree)
    x_ref_extracted_poly = pf.fit_transform(x_ref_extracted.values.reshape(-1, 1))

    lr = LinearRegression()
    lr.fit(x_ref_extracted_poly, np.array(x_ref_true_list).reshape(-1, 1))

    x_ref = pf.fit_transform(df_ref.x.values.reshape(-1, 1))
    x_ref_corrected = lr.predict(x_ref)
    data = np.hstack([x_ref_corrected, df_ref.y.values.reshape(-1, 1)])
    df_ref_corrected = pd.DataFrame(data=data, columns=['x', 'y'])

    plt.plot(df_ref_corrected.x, df_ref_corrected.y, color='r')
    # plt.xlim(2200, 3000)
    # plt.ylim(ymin, ymax)
    plt.show()


if __name__ == '__main__':
    main()
