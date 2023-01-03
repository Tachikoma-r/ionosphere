from math import factorial

import matplotlib.pyplot as plt
from scipy import special
from typing import List, Tuple
import numpy as np
import georinex as gr
import pandas as pd
import glob
import itertools
import logger

from constants import SPEED_OF_LIGHT, EARTH_RADIUS, factor_TEC
import logging
from tec import find_files, sat_list, driver

custom_print = logger.get_logger(__name__).info

GPS_list = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
            'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24',
            'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32']
h = 500.
rad = 180. / np.pi
tecu2ns = 1e9 / SPEED_OF_LIGHT / factor_TEC
order = 4

def norm_p(n: int, m: int) -> float:
    """
    Возвращает константу нормализации для соответствующего полинома Лежандра.

    :param n: Порядок сферической гармоники
    :type n: int
    :param m: Порядок сферической гармоники
    :type m: int
    :return: Константа нормализации для связанного полинома Лежандра.
    """
    if m == 0:
        return np.sqrt(factorial(n - m) * (2 * n + 1) / factorial(n + m))
    else:
        return np.sqrt(factorial(n - m) * (4 * n + 2) / factorial(n + m))


def get_coef(b: float, s: float, order: int) -> np.ndarray:
    """
    Расчитывает коэффициенты сферической гармоники.

    :param b: Геоцентрическая широта в IPP
    :type b: float
    :param s: Долгота в IPP
    :type s: float
    :param order: Порядок сферической гармоники
    :type order: int
    :return: Коэффициенты сферических гармоник.
    """
    cof_p = np.zeros((order + 1) * (order + 1))
    ms = np.linspace(s, order * s, order)
    i = 0
    x = np.sin(b)
    for n in range(order + 1):
        for m in range(n + 1):
            p = special.lpmv(m, n, x)
            if m == 0:
                cof_p[i] = p * norm_p(n, m)
            else:
                cof_p[i] = p * norm_p(n, m) * np.cos(ms[m - 1])
                i = i + 1
                cof_p[i] = p * norm_p(n, m) * np.sin(ms[m - 1])
            i = i + 1
    return cof_p


def Get_coef(b,s,order):
    import numpy as np
    from scipy import special

    cof_P = np.zeros((len(b), (order+1)*(order+1)))
    ms = np.zeros((len(s),order))

    for i in range(len(s)):
        ms[i,:] = np.linspace(s[i], order*s[i], order)

    i = 0
    x = np.sin(b)
    for n in range(order+1):
        for m in range(n+1):
            P = special.lpmv(m,n,x)
            if m==0:
                cof_P[:, i] = P*norm_p(n,m)
            else:
                cof_P[:, i] = P*norm_p(n,m)*np.cos(ms[:,m-1])
                i=i+1
                cof_P[:, i] = P*norm_p(n,m)*np.sin(ms[:,m-1])
            i=i+1
    return cof_P


def get_ipp(el: float, az: float, lat_r: float, lon_r: float, z: float, t_r):
    """
    Возвращает широту и долготу в IPP

    :param el: Угол возвышения в рад
    :type el: float
    :param az: Азимут в рад
    :type az: float
    :param lat_r: Широта приемника
    :type lat_r: float
    :param lon_r: Долгота приемника
    :type lon_r: float
    :param z: Зенитный угол
    :type z: float
    :param t_r: Время восхода спутника
    :return: b, s: Широта и долгота точки в IPP.
    """
    t = np.pi / 2 - el - z
    b = np.arcsin(np.sin(lat_r) * np.cos(t) + np.cos(lat_r) * np.sin(t) * np.cos(az))
    s = lon_r + np.arcsin(np.sin(t) * np.sin(az) / np.cos(t))
    s = s + t_r - np.pi
    return b, s


def calc_fit_dcb_old(stec: np.ndarray, mapf: np.ndarray, lat_i: np.ndarray, lon_i: np.ndarray,
                     k: int, n_r: int, order: int, interval: int) -> Tuple[List[np.ndarray], List]:
    sat_num, time_idx = stec.shape
    stec[np.where(stec == 0)] = np.nan
    tmx = int(time_idx)
    ep = np.linspace(0, tmx - 1, tmx) * interval / 3600
    d_hour = 12
    t_r = ep * 2 * np.pi / 24
    t_r = t_r[None, :] * np.ones((sat_num, time_idx))
    yun = tmx // d_hour
    h_matrix = []
    l_matrix = []
    ith = k
    s_rad = lon_i / rad - np.pi + t_r
    b_rad = lat_i / rad
    for i in range(d_hour):
        for j, k in itertools.product(range(sat_num), range(yun * i, yun * (i + 1))):
            if np.isnan(stec[j, k]):
                continue
            h_elem = np.zeros(((order + 1) * (order + 1) * 12 + 32 + n_r))
            map_func_elem = mapf[j, k].copy()
            st = (order + 1) * (order + 1) * i + 32 + n_r
            ed = (order + 1) * (order + 1) * (i + 1) + 32 + n_r - 1
            h_elem[ith] = (-9.52437) * map_func_elem
            h_elem[j + n_r] = (-9.52437) * map_func_elem
            h_elem[st:ed + 1] = get_coef(b_rad[j, k], s_rad[j, k], order)
            l_elem = stec[j, k] * map_func_elem
            l_matrix.append(l_elem)
            h_matrix.append(h_elem)
    return h_matrix, l_matrix


def calc_fit_dcb_o1(stec: np.ndarray, az: np.ndarray, el: np.ndarray, receiver_lon: int, receiver_lat: int,
                 k: int, n_r: int, order: int, interval: int) -> Tuple[list[np.ndarray], list]:
    """
        Вычисляет матрицу коэффициентов и вектор наблюдения для модели DCB.

        :param stec: Значения STEC для каждого спутника и времени
        :param az: Азимутальный угол
        :param el: Угол возвышения
        :param receiver_lon: Долгота приемника
        :param receiver_lat: Широта приемника
        :param k: Индекс приемника, DCB которого рассчитывается
        :param n_r: Количество приемников
        :param order: Порядок сферических гармоник
        :param interval: Временной интервал между каждым наблюдением в секундах
        :return: Кортеж из двух списков. Первый список содержит матрицу плана, второй список содержит наблюдения.
        """
    sat_num, time_idx = stec.shape

    az[np.where(stec == 0)] = np.nan
    el[np.where(stec == 0)] = np.nan
    stec[np.where(stec == 0)] = np.nan
    tmx = int(time_idx)

    zp = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + h) * np.sin((0.9782 * (90 - el)) * np.pi / 180))
    # plt.plot(zp[0])
    # print('tgyh')
    map_func = np.cos(zp)
    ep = np.linspace(0, tmx - 1, tmx) * interval / 3600
    d_hour = 1
    t_r = ep * 2 * np.pi / 24
    t_r = t_r[None, :] * np.ones((sat_num, time_idx))
    yun = tmx // d_hour
    h_matrix = []
    l_matrix = []
    ith = k
    sbm = receiver_lat * np.ones((sat_num, time_idx)) / rad
    slm = receiver_lon * np.ones((sat_num, time_idx)) / rad
    b_rad, s_rad = get_ipp(el / rad, az / rad, sbm, slm, zp, t_r)
    for i in range(d_hour):
        for j, k in itertools.product(range(sat_num), range(yun * i, yun * (i + 1))):
            if (np.isnan(stec[j, k])) or (stec[j, k] == 0):
                continue
            h_elem = np.zeros(((order + 1) * (order + 1) * d_hour + 32 + n_r))
            map_func_elem = map_func[j, k].copy()
            st = (order + 1) * (order + 1) * i + 32 + n_r
            ed = (order + 1) * (order + 1) * (i + 1) + 32 + n_r - 1
            h_elem[ith] = (-1) * map_func_elem
            h_elem[j + n_r] = (-1) * map_func_elem
            h_elem[st:ed + 1] = get_coef(b_rad[j, k], s_rad[j, k], order)
            # print(s_rad[j, k])
            # print(h_elem[st:ed + 1])
            l_elem = stec[j, k] * map_func_elem
            l_matrix.append(l_elem)
            h_matrix.append(h_elem)
    # print(l_matrix)
    # plt.plot(l_matrix)
    # plt.show()
    return h_matrix, l_matrix


def calc_fit_dcb(df, order: int) -> Tuple[list[np.ndarray], list]:
    h_matrix = []
    l_matrix = []
    n_r = len(df.STATION.unique())
    for i in range(n_r):
        df0 = df.loc[df.STATION == df.STATION.unique()[i]]
        print(df.STATION.unique()[i])
        stec = df0.groupby('PRN')['stec'].apply(np.array) * tecu2ns
        el = df0.groupby('PRN')['el'].apply(np.array)
        az = df0.groupby('PRN')['az'].apply(np.array)
        ix = df0.groupby('PRN')['ix'].apply(np.array)
        receiver_lat = df0.groupby('PRN')['rlat'].apply(np.array)
        receiver_lon = df0.groupby('PRN')['rlon'].apply(np.array)
        for k in range(len(stec)):
            # print(stec[k])
            time_idx = len(stec[k])
            zp = np.arcsin(EARTH_RADIUS / (EARTH_RADIUS + h) * np.sin((0.9782 * (90 - el[k])) * np.pi / 180))
            map_func = np.cos(zp)
            ep = ix[k] * 60 / 3600
            t_r = ep * 2 * np.pi / 24
            sbm = receiver_lat[k] * np.ones((time_idx)) / rad
            slm = receiver_lon[k] * np.ones((time_idx)) / rad
            b_rad, s_rad = get_ipp(el[k] / rad, az[k] / rad, sbm, slm, zp, t_r)
            h_elem = np.zeros((time_idx, (order + 1) * (order + 1) + 32 + n_r))
            st = 32 + n_r
            ed = (order + 1) * (order + 1) + 32 + n_r - 1
            h_elem[:, i] = (-1) * map_func
            h_elem[:, k + n_r] = (-1) * map_func
            h_elem[:, st:ed + 1] = Get_coef(b_rad, s_rad, order)
            l_elem = stec[k] * map_func
            l_matrix.append(l_elem)
            h_matrix.append(h_elem)
    cc = list(itertools.chain.from_iterable(h_matrix))
    gg = list(itertools.chain.from_iterable(l_matrix))
    return cc, gg


def solve_matrix_old(h_matrix: np.ndarray, l_matrix: np.ndarray) -> np.ndarray:
    r, resid, rank, s = np.linalg.lstsq(h_matrix, l_matrix, rcond=None)
    return r * 10 ** 9 / 299792458


def solve_matrix(h_matrix: np.ndarray, l_matrix: np.ndarray) -> np.ndarray:
    """
    Решение матричного уравнения методом наименьших квадратов

    :param h_matrix: Матрица правой части уравнения = матрица коэффициентов
    :type h_matrix: np.ndarray
    :param l_matrix: Матрица левой части уравнения = вектор наблюдения
    :type l_matrix: np.ndarray
    :return: Возвращаемое значение является решением матричного уравнения = значения DCB
     для приемников (0..n_r), для спутников (n_r..n_r+число спутников) и ионосферные коэффициенте
    """
    r, resid, rank, s = np.linalg.lstsq(h_matrix, l_matrix, rcond=None)
    return r


def df_dcb(r: np.ndarray, n_r: int, sv: np.ndarray, recv_name: List[str]) -> pd.DataFrame:
    """
    Решение матричного уравнения для dcb записывается в pandas. Составляется списсок из полного числа спутников (32)
    и сравнивается с имеющимися спуниками по файлу. Не имеющие значения получают dcb 0

    :param r: Список оценочных значений
    :type r: List[float]
    :param n_r: Количество приемников
    :type n_r: int
    :param sv: Список спутников
    :type sv: List[str]
    :param recv_name: Имя приемника
    :type recv_name: List[str]
    """
    df = pd.DataFrame({'ESTIMATED_VALUE': r[:len(sv) + n_r]})
    # TODO ?вероятно, тут будет ошибка, если в первом элементе obs 30 спутников, а потом 32

    index_values = np.arange(n_r, 32 + n_r)
    pd_gps = pd.Series(GPS_list, index=index_values)
    result = pd_gps.str.match('|'.join(sv))
    inverse = result[~result]
    result = result[result]
    df.loc[n_r:len(sv) + n_r, 'PRN'] = pd_gps.loc[result.index].values
    df.loc[:n_r, 'STATION'] = pd.Series(recv_name)

    # заполнили 0
    line = pd.DataFrame({"ESTIMATED_VALUE": 0.0, "PRN": pd_gps.loc[inverse.index].values, 'STATION': ''})
    df = pd.concat([df, line], ignore_index=True)
    df = df.sort_values(['PRN', 'STATION'], ignore_index=True)
    return df


def save2file_dcb(df: pd.DataFrame, recv_name: List[str], path_out: str, date: str, first_start: bool = True):
    """
    Из датафрейма создаем эталонный файл, где 0 значения спутников -> среднее по всем спутникам.
    Для файла dcb заполняем 0ли из эталонного файла и дополняем приемниками, если появились новые.
    Если есть все данные, то ничего не обновляем и файл становится новым эталонным

    :param df: DataFrame данных, который нужно сохранить
    :type df: pd.DataFrame
    :param recv_name: Список имен приемников
    :type recv_name: List[str]
    :param path_out: Путь к папке, в которой будет сохранен файл DCB
    :type path_out: str
    :param date: Дата данных, которые загружаются
    :type date: str
    :param first_start: Если True, то при запуске программы будет создан первый эталонный файл
    :type first_start: bool (optional)
    """
    # если первый запуск , то создадим реф с усреднением
    if first_start:
        prn_rows = df[~df['PRN'].isna()]
        df3 = df.mask(df == 0).fillna(prn_rows.mean())
        df3.to_csv(f'{path_out}/DCBf.csv', index=False)
    # прочитали реф
    dcb_file_ref = glob.glob(f'{path_out}/DCBf.csv')
    df_ref = pd.read_csv(dcb_file_ref[0])
    # заполняем 0 из рефа
    stan_ref = df_ref['STATION'].dropna().values
    inter1 = list(set(stan_ref) - set(recv_name))
    rows = df_ref[df_ref['STATION'].isin(inter1)]
    df1 = df.where(df != 0, df_ref)
    df1 = df1.append(rows, ignore_index=True)
    print(f'{path_out}/kkk')
    df1.to_csv(f'{path_out}/DCB111-{date}.csv',
               index=False)
    df1.to_csv(f'{path_out}/DCBf.csv', index=False)


#TODO сделать выбор спутника опциональным?, вынести по функциям
def write_dcb_o1(path_out: str, dot_o: List[str], pattern: List[str], time_step: int = 60, order: int = 4,
              type_sat: str = 'G', first_start: bool = True) -> None:
    """
    Принимает путь к файлам RINEX, список файлов наблюдений и шаблон расширения (например, ['22o.gz', '22n']).
    Производит расчет DCB и записывает в файл со структурой ESTIMATED_VALUE, PRN, STATION .
    При первом запуске, если еще не создан референсный файл, необходимо установить first_start = True, после False.

    :param path_out: Путь к папке, в которой будет сохранен файл DCB
    :type path_out: str
    :param dot_o: Список путей к файлам наблюдения
    :type dot_o: List[str]
    :param pattern: Шаблон расширения (например, ['22o.gz', '22n'])/ первый - файла наблюдения, второй — файла навигации
    :type pattern: List[str]
    :param time_step: Количество cекунд между наблюдениями
    :type time_step: int
    :param order: Порядок полинома Лежандра
    :type order: int
    :param type_sat: Тип спутника - "G" для GPS, "R" для ГЛОНАСС, "E" для Galileo
    :type type_sat: str (optional)
    :param first_start: Если True, то при запуске программы будет создан первый эталонный файл
    :type first_start: bool (optional)
    """
    try:
        obs_prob = gr.load(dot_o[0], use=type_sat, useindicators=True)
        s_t_p = sat_list(obs_prob, type_sat)
        # TODO sat_list аналогичен obs_prob.sv.values, но может быть нужен, если убрать type_sat из выгрузки
    except Exception as e:
        logging.exception(e)

    n_r = len(dot_o)
    h_final = np.zeros((0, (order + 1) ** 2 * 1 + 32 + n_r))
    l_final = np.zeros(0)
    recv_name = []
    tmp_dataframe = pd.DataFrame()
    for k in range(n_r):
        obs = gr.load(dot_o[k], tlim=None, use=type_sat, useindicators=True)
        obs['time'] = pd.DatetimeIndex(obs['time'].values)
        obs = obs.resample(time=f'{time_step}S').ffill()
        r_lat, r_lon, r_alt = obs.position_geodetic
        recv_name_one = dot_o[k][-len(pattern[0]) - 9:-len(pattern[0]) - 5]
        recv_name.append(recv_name_one)
        path_obs = (dot_o[k][:-len(pattern[0]) - 1])
        path_n = glob.glob(f'{path_obs}.{pattern[1]}')[0]
        nav = gr.load(path_n)
        s_t = sat_list(obs, type_sat)
        stec = np.nan * np.ones((s_t.shape[0], obs.time.values.shape[0]))
        mapf = stec.copy()
        lat_i = stec.copy()
        lon_i = stec.copy()
        az_i = stec.copy()
        el_i = stec.copy()
        for i, sv in enumerate(s_t):
            stec[i, :], mapf[i, :], lat_i[i, :], lon_i[i, :], az_i[i, :],\
                el_i[i, :], driver_df = driver(obs, nav, None, i, sv, recv_name=recv_name_one)
            tmp_dataframe = pd.concat([tmp_dataframe, driver_df], ignore_index=True)
        # print(tmp_dataframe)
        tmp_dataframe.to_csv('out.csv')
        b1, l1 = calc_fit_dcb(stec * tecu2ns, az_i, el_i, r_lon, r_lat, k, n_r, order, time_step)
        h_final = np.concatenate((h_final, b1), axis=0)
        l_final = np.concatenate((l_final, l1), axis=0)

    date_str = f'{pattern[0][:2]}-{dot_o[0][-len(pattern[0]) - 5:-len(pattern[0]) - 1]}'
    ################
    # date_int = int(date_str[3:])-10
    # csv_file = glob.glob(f'E:/codes/result/*_{date_int}_prob.csv')
    # df_csv = pd.read_csv(csv_file[0])
    # print(df_csv)
    # b1, l1 = calc_fit_dcb(df_csv.stec * tecu2ns, df_csv.az, df_csv.el, df_csv.rlon, df_csv.rlat, k, n_r, order, time_step)
    ################
    solve = solve_matrix(h_final, l_final)
    df = df_dcb(solve, n_r, s_t_p, recv_name)
    print(df)
    save2file_dcb(df, recv_name, path_out, date_str, first_start)


def write_dcb(path_csv, path_out, date_str, first_file=True, divide=None):
    global order
    # [!0]
    if divide==None:
        searched_files = glob.glob(f'{path_csv}/*_{date_str}0*_*.csv')
        print(f'{path_csv}/*_{date_str}0*_*.csv')
    else:
        searched_files = glob.glob(f'{path_csv}/*_{date_str}[!0]*_*.csv')
    print(searched_files)
    if len(searched_files) > 1:
        print('on')
        df = pd.concat([pd.read_csv(f) for f in searched_files], ignore_index=True)
        df.sort_values(["STATION","PRN"], ascending=[False,True],inplace=True)
        df.to_csv('terr')
    else:
        print('yu')
        df = pd.read_csv(searched_files[0])
    prn_unique = df.PRN.unique()
    recv_n = df.STATION.unique()
    n_r = len(recv_n)
    cc, gg = calc_fit_dcb(df, order)
    solve = solve_matrix(cc, gg)
    # print(solve)
    df = df_dcb(solve, n_r, prn_unique, recv_n)
    save2file_dcb(df, recv_n, path_out, date_str, first_file)


def writer_dcb(path_csv: str, path_out: str, num_of_day: str,
               pattern: List[str], time_step: int = 60, one_folder: bool = 1, first_start: bool = True) -> None:
    """
      Функция принимает путь к папке, дату, список паттернов и размер шага и записывает файл DCB.

      :param path_in: Путь к папке, в которой находятся файлы RINEX
      :type path_in: str
      :param path_out: Путь к папке, в которой вы хотите сохранить файл DCB
      :type path_out: str
      :param num_of_day: Дата файлов, которые вы хотите обработать в виде дня года [e.g. '1250' - 125 день года]
      :type num_of_day: str
      :param pattern: Шаблон расширения (например, ['22o.gz', '22n'])/ первый - файла наблюдения, второй — навигации
      :type pattern: List[str]
      :param time_step: Количество cекунд между наблюдениями
      :type time_step: int
      :param one_folder: Если вы надо записать файл DCB для файлов из одной папки, установите значение 1
      :type one_folder: bool (optional)
      :param first_start: Еcли True, то при запуске программы будет создан первый эталонный файл
      :type first_start: bool (optional)
      """
    write_dcb(path_csv,path_out,num_of_day)
