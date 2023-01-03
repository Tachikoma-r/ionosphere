import fnmatch
import glob
import os
import zipfile
import multiprocessing
from multiprocessing.dummy import Pool

import xarray
from pymap3d import ecef2geodetic, ecef2aer, aer2geodetic
from pandas import Timestamp, DataFrame
from itertools import accumulate, groupby
from typing import List, Union, Optional, Tuple, Any
import georinex as gr
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from io import StringIO
import datetime

import constants
import logger
from constants import factor_TEC, factor, factor_mw, SPEED_OF_LIGHT, EARTH_RADIUS, EARTH_GM, EARTH_ROTATION_RATE
custom_print = logger.get_logger(__name__).info

load_dotenv()
header = ['Datetime', 'STATION', 'PRN', 'lat', 'lon', 'vtec', 'stec', 'az', 'el', 'rlat','rlon','ix']
out = os.getenv('OUTPUT_PATH')
fN = int(os.getenv('FREQ_CHANNEL_NUMBER'))

conversionFactor = 0.463 * 6.158  # ns to tec (GPS)

error_multiplier = 10000
DIFF_TEC_MAX = 0.05
p_mean = 0.0
p_dev = DIFF_TEC_MAX * 4.0


def correct_phases(RTEC, MWLC,
                   l1_values, l2_values,
                   c1_values, p2_values, i):
    """
    Функция корректирует значения фаз сигналов L1 и L2, вычитая разницу между текущими
    и предыдущими значениями TEC и MWLC

    :param RTEC: Значения RTEC
    :param MWLC: Значения MWLC
    :param l1_values: Значения псевдодальности L1
    :param l2_values: Значения псевдодальности L2
    :param c1_values: Значения псевдодальности C1
    :param p2_values: Значения псевдодальности P2
    :param i: индекс текущей эпохи
    :return: Скорректированные значения l1 и l2.
    """
    from constants import GPS_L1, GPS_L2
    l1 = l1_values[i]
    l2 = l2_values[i]
    c1 = c1_values[i]
    p2 = p2_values[i]

    diff_tec = RTEC[i] - RTEC[i - 1]
    diff_mwlc = MWLC[i] - MWLC[i - 1]

    diff_2 = np.round((diff_tec - (diff_mwlc * SPEED_OF_LIGHT
                                   / GPS_L1)) * factor_mw)

    diff_1 = diff_2 + np.round(diff_mwlc)

    corr_1 = l1 - diff_1
    corr_2 = l2 - diff_2

    RTEC[i] = ((corr_1 / GPS_L1) - (corr_2 / GPS_L2)) * SPEED_OF_LIGHT
    MWLC[i] = ((corr_1 - corr_2) - (GPS_L1 * c1 + GPS_L2 * p2) * factor)

    for num in range(i, len(l1_values)):
        l1_values[num] = l1_values[num] - diff_1
        l2_values[num] = l2_values[num] - diff_2

    return l1_values, l2_values


def cycle_slip_corrector(time, l1_values, l2_values,
                         c1_values, p2_values,
                         l1lli_values, l2lli_values):
    """
      Корректирует значения фаз для сбоев цикла

      :param time: Время наблюдения
      :param l1_values: Значения псевдодальности L1
      :param l2_values: Значения псевдодальности L2
      :param c1_values: Значения псевдодальности C1
      :param p2_values: Значения псевдодальности P2
      :param l1lli_values: L1 Индикатор блокировки
      :param l2lli_values: L2 Индикатор блокировки
      :return: Скорректированные значения L1 и L2, а также значения RTEC.
      """
    from constants import GPS_L1, GPS_L2
    index_start = 0
    size = 10

    RTEC = np.zeros(len(time))
    MWLC = np.zeros(len(time))  # Comb. Melbourne–Wübbena

    for index in range(len(time)):

        l1 = l1_values[index]
        l2 = l2_values[index]
        c1 = c1_values[index]
        p2 = p2_values[index]

        RTEC[index] = ((l1 / GPS_L1) - (l2 / GPS_L2)) * SPEED_OF_LIGHT
        MWLC[index] = ((l1 - l2) - (GPS_L1 * c1 + GPS_L2 * p2) * factor)

        if time[index] - time[index - 1] > datetime.timedelta(minutes=15):
            index_start = index

        l_slip1 = l1lli_values[index] % 2
        l_slip2 = l2lli_values[index] % 2

        if l_slip1 == 1 or l_slip2 == 1:
            l1_values, l2_values = correct_phases(RTEC, MWLC,
                                                  l1_values, l2_values,
                                                  c1_values, p2_values,
                                                  index)

        p_mean = 0.0
        p_dev = DIFF_TEC_MAX * 4.0

        if index - index_start >= 11:

            add_tec = 0
            add_tec_2 = 0

            for elem in range(1, size):
                add_tec = add_tec + RTEC[index - elem] - RTEC[index - 1 - elem]

                add_tec_2 = add_tec_2 + np.power(RTEC[index - elem] -
                                                 RTEC[index - 1 - elem], 2)

            p_mean = add_tec / size

            p_dev = max(np.sqrt(add_tec_2 / size - np.power(p_mean, 2)),
                        DIFF_TEC_MAX)

        p_min_tec = p_mean - p_dev * 5.0
        p_max_tec = p_mean + p_dev * 5.0

        diff_tec = RTEC[index] - RTEC[index - 1]

        if not (p_min_tec <= diff_tec < p_max_tec):
            l1_values, l2_values = correct_phases(RTEC, MWLC,
                                                  l1_values, l2_values,
                                                  c1_values, p2_values,
                                                  index)

    return l1_values, l2_values, RTEC


def relative_tec(time, c1, p2, r_tec):
    """
    Возвращает значения RTEC с удаленным скачками в данных

    :param time: Время наблюдения
    :param c1: Значения псевдодальности C1
    :param p2: Значения псевдодальности P2
    :param r_tec: Значения RТЕС
    :return: Возвращается относительный TEC
    """
    narc = 1
    index_last = 0
    if not p2.size:
        return np.nan
    a1 = p2[0] - c1[0]
    b = r_tec[0] - a1

    for index in range(1, len(time)):

        if time[index] - time[index - 1] > datetime.timedelta(minutes=15):

            b = b / narc

            for elem in range(index_last, index):
                r_tec[elem] = factor_TEC * (r_tec[elem] - b)

            index_last = index
            narc = 1
            a1 = p2[index] - c1[index]

            b = r_tec[index] - a1

        else:
            narc += 1
            a1 = p2[index] - c1[index]
            b = b + (r_tec[index] - a1)

    b = b / narc

    for elem in range(index_last, len(time)):
        r_tec[elem] = factor_TEC * (r_tec[elem] - b)
    return r_tec


def mean2eccentric(m: float, ecc: float) -> float:
    """
    Функция принимает среднюю аномалию и эксцентриситет в качестве входных данных и возвращает эксцентрическую аномалию

    :param m: Средняя аномалия
    :type m: float
    :param ecc: Эксцентриситет
    :type ecc: float
    :return: em: Эксцентрическая аномалия
    """
    if ecc > .999 or ecc < 0.:
        m = np.nan
        print("Not circular or elliptical")
    twopi = 2. * np.pi
    tol = 1.0e-12
    maxiter = 20
    m = np.remainder(m, twopi)
    m = (m < 0.) * twopi + m
    sinm = np.sin(m)
    em = m + (ecc * sinm) / (1. - np.sin(m + ecc) + sinm)
    count = 1
    err = 1
    while abs(err) > tol and count <= maxiter:
        err = (em - ecc * np.sin(em) - m) / (1. - ecc * np.cos(em))
        em = em - err
        count += 1
        if count > maxiter:
            print('Iterations maxed out in mean2eccentric')
    return em


def get_gps_time(dt: datetime) -> int:
    """
    Преобразует объект datetime в количество секунд с начала недели GPS.

    :param dt: Дата и время
    :return total: Секунды
    """
    from constants import SECS_IN_HR, SECS_IN_MIN
    total = 0
    days = (dt.weekday() + 1) % 7
    total += days * SECS_IN_HR * 24
    total += dt.hour * SECS_IN_HR
    total += dt.minute * SECS_IN_MIN
    total += dt.second
    return total


def gps_sat_position(nav: xarray.Dataset, dt: np.ndarray, sv: Optional[str] = None,
                     rx_position: Optional[List[float]] = None, cords: str = 'xyz') -> np.ndarray:
    """
      По навигационному файлу RINEX расчитывает положения спутников GPS и возвращает в координатах ECEF.
      Алгоритм по https://gssc.esa.int/navipedia/index.php/GPS_and_Galileo_Satellite_Coordinates_Computation
      Основано на aldebaran1/ Sebastijan Mrak gsit pyGps.py

      :param nav: Объект xarray.Dataset, содержащий прочитанные навигационные данные
      :type nav: xarray.Dataset
      :param dt: Дата и время
      :type dt: datetime
      :param sv: Название спутника [e.g. 'G01']
      :type sv: str (optional)
      :param rx_position: Положение приемника в координатах ECEF
      :type rx_position: Optional[List[float]] (optional)
      :param cords: Координаты выводить в xyz или aer
      :type cords: str (optional)
      :return: Положение спутника в координатах ECEF.
      """

    nav_data = nav.sel(sv=sv)
    times_array = np.asarray(dt, dtype='datetime64[ns]')
    nav_times = nav_data.time.values
    id_nan = np.isfinite(nav_data['Toe'].values)
    nav_times = nav_times[id_nan]
    best_ep_hind = []
    for t in times_array:
        idt = abs(nav_times - t).argmin()
        best_ep_hind.append(idt)
    gpstime = np.array([get_gps_time(t) for t in dt])
    t = gpstime - nav_data['Toe'][id_nan][best_ep_hind].values
    ecc = nav_data['Eccentricity'][id_nan][best_ep_hind].values
    m_k = nav_data['M0'][id_nan][best_ep_hind].values + \
          t * (np.sqrt(EARTH_GM / nav_data['sqrtA'][id_nan][best_ep_hind].values ** 6) +
               nav_data['DeltaN'][id_nan][best_ep_hind].values)
    e_k = np.zeros(len(m_k))
    for i in range(len(m_k)):
        e_k[i] = mean2eccentric(m_k[i], ecc[i])
    v_k = np.arctan2(np.sqrt(1.0 - ecc ** 2) * np.sin(e_k), np.cos(e_k) - ecc)
    phi_k = v_k + nav_data['omega'][id_nan][best_ep_hind].values
    delta_uk = nav_data['Cuc'][id_nan][best_ep_hind].values * np.cos(2.0 * phi_k) + \
               nav_data['Cus'][id_nan][best_ep_hind].values * np.sin(2.0 * phi_k)
    u_k = phi_k + delta_uk

    delta_rk = nav_data['Crc'][id_nan][best_ep_hind].values * np.cos(2.0 * phi_k) + \
               nav_data['Crs'][id_nan][best_ep_hind].values * np.sin(2.0 * phi_k)
    r_k = nav_data['sqrtA'][id_nan][best_ep_hind].values ** 2 * (1.0 - ecc * np.cos(e_k)) + delta_rk

    delta_ik = nav_data['Cic'][id_nan][best_ep_hind].values * np.cos(2.0 * phi_k) + \
               nav_data['Cis'][id_nan][best_ep_hind].values * np.sin(2.0 * phi_k)
    i_k = nav_data['Io'][id_nan][best_ep_hind].values + \
          nav_data['IDOT'][id_nan][best_ep_hind].values * t + delta_ik

    omega_k = nav_data['Omega0'][id_nan][best_ep_hind].values + \
              (nav_data['OmegaDot'][id_nan][best_ep_hind].values - EARTH_ROTATION_RATE) * t - \
              (EARTH_ROTATION_RATE * nav_data['Toe'][id_nan][best_ep_hind].values)
    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)
    X = x_k_prime * np.cos(omega_k) - (y_k_prime * np.sin(omega_k) * np.cos(i_k))
    Y = x_k_prime * np.sin(omega_k) + (y_k_prime * np.cos(omega_k) * np.cos(i_k))
    Z = y_k_prime * np.sin(i_k)

    if cords == 'xyz':
        return np.array([X, Y, Z])
    elif cords == 'aer':
        assert rx_position is not None
        rec_lat, rec_lon, rec_alt = ecef2geodetic(rx_position[0], rx_position[1], rx_position[2])
        A, E, R = ecef2aer(X, Y, Z, rec_lat, rec_lon, rec_alt)
        return np.array([A, E, R])


def get_intervals(l_1: np.ndarray, l_2: np.ndarray, p_1: np.ndarray, p_2: np.ndarray, max_gap: int = 3,
                  max_jump: int = 2):
    from constants import SPEED_OF_LIGHT, A, GPS_L1, GPS_L2
    """
     Cканирует фазу спутника и определяет, где "хорошие" интервалы начинаются и заканчиваются
     Основано на aldebaran1/ Sebastijan Mrak gsit pyGps.py

     :param l_1: Псевдодальность на частоте L1
     :type l_1: np.ndarray
     :param l_2: Псевдодальность на частоте L2
     :type l_2: np.ndarray
     :param p_1: Фаза сигнала на частоте L1
     :type p_1: np.ndarray
     :param p_2: Фаза сигнала на частоте L2
     :type p_2: np.ndarray
     :param max_gap: Максимальное количество nan перед началом нового интервала
     :type max_gap: int (optional)
     :param max_jump: Максимальный скачок фазового TEC перед началом нового интервала
     :type max_jump: int (optional)
     """
    r = np.array(range(len(p_1)))
    idx = np.isfinite(l_1) & np.isfinite(l_2) & np.isfinite(p_1) & np.isfinite(p_2)
    r = r[idx]
    intervals = []
    if len(r) == 0:
        return idx, intervals
    freq_m = (GPS_L1 ** 2 * GPS_L2 ** 2)/(GPS_L1 ** 2 - GPS_L2 ** 2)
    phase_tec = freq_m * SPEED_OF_LIGHT / A / pow(10, 16) * (l_1 / GPS_L1 - l_2 / GPS_L2)
    beginning = r[0]
    last = r[0]
    for i in r[1:]:
        if (i - last > max_gap) or (abs(phase_tec[i] - phase_tec[last]) > max_jump):
            intervals.append((beginning, last))
            beginning = i
        last = i
        if i == r[-1]:
            intervals.append((beginning, last))
    return idx, intervals


def sat_position(fnc: xarray.Dataset, f_nav: xarray.Dataset, sv: str, el_mask: int = 10,
                 t_lim: Optional[List[str]] = None, sat_pos: bool = False, ipp: bool = False,
                 ipp_alt: Optional[float] = None) -> xarray.Dataset:
    """
    Возвращает положение спутника с учетом отсечки по углу возвышения в точке IPP.

    :param fnc: Файл наблюдений
    :type fnc: xarray.Dataset
    :param f_nav: Навигационный файл
    :type f_nav: xarray.Dataset
    :param sv: Название спутника [e.g. 'G01']
    :type sv: str
    :param el_mask: Маска на угол возвышения в градусах
    :type el_mask: int (optional)
    :param t_lim: Срок обработки. Если None, будет обработан весь файл
    :type t_lim: Optional[List[str]]
    :param sat_pos: True, если надо добавить азимут и возвышение в xarray
    :type sat_pos: bool (optional)
    :param ipp: True, если необходим расчет в IPP
    :type ipp: bool (optional)
    :param ipp_alt: Высота точки прокола ионосферы (IPP)
    :type ipp_alt: Optional[float]
    :return: df: Положением спутника по азимуту и возвышению в IPP
    """
    df = fnc.sel(sv=sv)
    df['time'] = np.array([np.datetime64(ttt) for ttt in df.time.values])
    if t_lim is not None and len(t_lim) == 2:
        t0 = t_lim[0]
        t1 = t_lim[1]
        df = df.where(np.logical_and(df.time >= np.datetime64(t0),
                                     df.time <= np.datetime64(t1)),
                      drop=True)
    obs_times_64 = df.time.values
    dt = np.array([Timestamp(t).to_pydatetime() for t in obs_times_64])
    aer = gps_sat_position(f_nav, dt, sv=sv, rx_position=df.position, cords='aer')
    id_el = (aer[1, :] >= el_mask)
    aer[:, ~id_el] = np.nan
    df['time'] = dt

    if sat_pos:
        df['az'] = aer[0, :]
        df['el'] = aer[1, :]
        df['idel'] = id_el
    if ipp:
        sat_position_ipp(ipp_alt, df, aer)
    return df


def sat_position_ipp(ipp_alt: float, df: xarray.Dataset, aer: np.ndarray) -> None:
    """
    Функция принимает высоту IPP, набор данных и координаты AER спутника и возвращает геодезические координаты IPP

    :param ipp_alt: Высота IPP. Если нет, будет установлено значение 300 км
    :type ipp_alt: float
    :param df: xarray с положенем приемника
    :type df: xarray.Dataset
    :param aer: Азимут, возвышение и дальность спутника от приемника
    :type aer: np.ndarray
    """
    if ipp_alt is None:
        print('Auto assigned altitude of the IPP: 300 km')
        ipp_alt = 300e3
    else:
        ipp_alt *= 1e3
    rec_lat, rec_lon, rec_alt = df.position_geodetic
    fm = np.sin(np.radians(aer[1]))
    r_new = ipp_alt / fm
    lla_vector = np.array(aer2geodetic(aer[0], aer[1], r_new, rec_lat, rec_lon, rec_alt))
    df['ipp_lon'] = lla_vector[1]
    df['ipp_lat'] = lla_vector[0]
    df['ipp_alt'] = ipp_alt


def mapping_function(elevation: float, h: float) -> float:
    """
    Считает функцию отображения. Принимая угол возвышению и высоту спутника -> возвращает расстояние от спутника
    до точки на поверхности Земли непосредственно под ним.

    :param elevation: Высота спутника в градусах
    :type elevation: float
    :param h: Высота спутника над поверхностью Земли
    :type h: float
    :return: Расстояние от центра земли до точки на поверхности земли на данной высоте.
    """
    rc1 = EARTH_RADIUS / (EARTH_RADIUS + h)
    return np.sqrt(1 - (np.cos(np.radians(elevation)) ** 2 * rc1 ** 2))


def read_one_sinex(file: str) -> DataFrame:
    """
    Читает файл SINEX, удаляет последние две строки, переименовывает столбцы и возвращает DataFrame, содержащий только
    спутниковые данные.

    :param file: Путь к файлу
    :type file: str
    :return: DataFrame с прочитанными данными
    """
    df = pd.read_fwf(file, skiprows=57)
    df.drop(df.tail(2).index, inplace=True)
    df.columns = ['bias', 'svn', 'prn', 'station', 'obs1', 'obs2',
                  'bias_start', 'bias_end', 'unit', 'value', 'std']
    ds = xr.Dataset()
    ds.attrs['bias'] = df['bias'].values[0]
    return df[df['station'].isnull()]


def tec_rph(f1, f2, p1, p2, l1, l2, r):
    from constants import SPEED_OF_LIGHT, A
    freq_m = ((f1 ** 2 * f2 ** 2) / (f1 ** 2 - f2 ** 2))
    range_tec = freq_m * (p2[r[0]: r[1]] - p1[r[0]: r[1]]) / A / pow(10, 16)
    phase_tec = freq_m * SPEED_OF_LIGHT * (l1[r[0]: r[1]] / f1 - l2[r[0]: r[1]] / f2) / A / pow(10, 16)
    return range_tec, phase_tec


def tec_glonass():
    return 0


def mult_get_phase_corr_tec(params: tuple):
    """
    Вычисляет значения TEC фазы и псевдодальности, а затем возвращает медианную разницу между двумя значениями.
    Основано на aldebaran1/ Sebastijan Mrak's gsit pyGps.py

    :param params: Кортеж с параметрами
    :type params: tuple
    """
    if params[0][1] - params[0][0] <= 1:
        return
    if params[10] is None:  # GPS
        freq_1 = params[1]
        if params[4] == 2:
            freq_2 = params[2]
        elif params[4] == 5:
            freq_2 = params[3]
        range_tec, phase_tec = tec_rph(freq_1, freq_2, params[5], params[6], params[7], params[8], params[0])
        custom_print("work")
    else:  # GLONASS
        from constants import GLONASS_L1, GLONASS_L2, GLONASS_L1_DELTA, GLONASS_L2_DELTA
        f_1 = GLONASS_L1 + params[10] * GLONASS_L1_DELTA
        f_2 = GLONASS_L2 + params[10] * GLONASS_L2_DELTA
        range_tec, phase_tec = tec_rph(f_1, f_2, params[5], params[6], params[7], params[8], params[0])
    tec_difference = np.array(sorted(phase_tec - range_tec))
    tec_difference = tec_difference[np.isfinite(tec_difference)]
    median_difference = tec_difference[len(tec_difference) // 2]
    return phase_tec - median_difference


def get_phase_corr_tec(l_1: np.ndarray, l_2: np.ndarray, p_1: np.ndarray, p_2: np.ndarray,
                       channel: int = 2, f_n: Optional[int] = None, max_gap: int = 3, max_jump: int = 2) -> np.ndarray:
    from constants import GPS_L1, GPS_L2, GPS_L5
    idx, ranges = get_intervals(l_1, l_2, p_1, p_2, max_gap=max_gap, max_jump=max_jump)
    tec = np.nan * np.zeros(len(l_1))

    params = [(ranges[i], GPS_L1, GPS_L2, GPS_L5, channel,
               p_1, p_2, l_1, l_2, tec, f_n) for i in range(len(ranges))]

    with Pool(multiprocessing.cpu_count()) as p:
        res = p.map(mult_get_phase_corr_tec, params)
        p.close()
        p.join()

    #TODO исправить nam в 59 минуте-элементе
    for ind, r in enumerate(ranges):
        tec[r[0]:r[1]] = res[ind]
        # print(r)
        # print('//////////****')
        # print(tec[r[0]:r[1]])
        # print('//////////')
        # print(tec)
    return tec


def sat_list(obs: xarray.Dataset, type_sat: str) -> np.ndarray:
    """
    Функция принимает набор данных наблюдения и тип спутника (например, "G") и возвращает список спутников этого типа.

    :param obs: Объект xarray.Dataset, содержащий прочитанные данные наблюдения
    :type obs: xarray.Dataset
    :param type_sat: Тип спутника - "G" для GPS, "R" для ГЛОНАСС, "E" для Galileo
    :type type_sat: str
    :return: Массив строк с именами имебщихся спутников в файле
    """
    sv_list = obs.sv.values
    sat = np.array([], dtype=object)
    for ch in sv_list:
        if ch.startswith(type_sat):
            sat = np.append(sat, ch)
    return sat


# TODO 60 минут или 30 получается?
def clear_jump_data(stec: np.ndarray) -> np.ndarray:
    """
    Удаляет из массива stec скачки значений длинном меньше 60 мин.

    :param stec: Массив stec
    :type stec: np.ndarray
    :return: Массив stec с удаленными скачками.
    """
    temp_stec = stec.copy()
    lim = 120
    for _ in range(32):
        bool_array = np.isnan(temp_stec)
        list_jump = list(accumulate(sum(1 for _ in g) for _, g in groupby(bool_array)))
        last = list_jump[0]
        interval = []
        for i in list_jump[1:]:
            temp_interval = i - last
            if temp_interval < lim:
                interval.extend((i, last))
            last = i
        interval.append(last)
        interval.sort()
        if interval:
            k = interval[0]
            for i in interval[1:]:
                if i - k > lim:
                    k = i
                    continue
                temp_stec[k:i + 1] = np.nan
                k = i
        return temp_stec


#TODO  переделать header как параметр для записи или еще как
def driver(obs: xarray.Dataset, nav: xarray.Dataset, t_lim: Optional[List[str]], i: int, sv: str,
           recv_name: Optional[str] = None, df: Optional[DataFrame] = None, recv_bias: Optional[float] = 0,
           el_mask: int = 10, h: float = 506.7, mode=0):
    """
       Считывает файлы наблюдения и навигационные. Определяет интервалы без скачков данных.
       Если mode = 0, то проивозит расчет без учета dcb и возвращает ndarray, если mode=1,
       то учитывает dcb для stec/vtec и записывает строки str в порядке header

       :param obs: объект xarray.Dataset, содержащий прочитанные данные наблюдения
       :type obs: xarray.Dataset
       :param nav: объект xarray.Dataset, содержащий прочитанные навигационные данные
       :type nav: xarray.Dataset
       :param t_lim: Срок обработки. Если None, будет обработан весь файл
       :type t_lim: Optional[List[str]]
       :param i: Индекс спутника в списке спутников
       :type i: int
       :param sv: Название спутника [e.g. 'G01']
       :type sv: str
       :param recv_name: Имя приемника
       :type recv_name: Optional[str]
       :param df: DataFrame с dcb
       :type df: Optional[DataFrame]
       :param recv_bias: Смещение приемника. 0 для расчета в mode=0
       :type recv_bias: Optional[float] (optional)
       :param el_mask: Маска на угол возвышения в градусах
       :type el_mask: int (optional)
       :param h: Высота IPP
       :type h: float
       :param mode: 0 для расчета dcb, 1 для возврата строки
       """
    final_result = ''
    stec = np.nan * np.ones((len(obs.sv.values), obs.time.values.shape[0]))
    D = sat_position(obs, nav, sv=sv, t_lim=t_lim, el_mask=el_mask, sat_pos=True,
                     ipp=True, ipp_alt=h)
    r_lat, r_lon, r_alt = obs.position_geodetic
    id_el = D['idel'].values
    dt = D.time.values
    ts = pd.to_datetime(dt)
    d = ts.strftime('%Y-%m-%d %H:%M:%S')
    el = D.el.values
    az = D.az.values
    lat, lon = D.ipp_lat.values, D.ipp_lon.values
    c1 = D['C1'].values
    c1[~id_el] = np.nan
    # c2 = D['C2'].values
    # c2[~id_el] = np.nan
    l1 = D['L1'].values
    l1[~id_el] = np.nan
    l2 = D['L2'].values
    l2[~id_el] = np.nan
    p2 = D['P2'].values
    p2[~id_el] = np.nan

    l1_lli_value = D['L1lli']
    l1_lli_value[~id_el] = np.nan
    l2_lli_value = D['L2lli']
    l2_lli_value[~id_el] = np.nan
    l1_lli_value[np.isnan(l1_lli_value)] = 0
    l2_lli_value[np.isnan(l1_lli_value)] = 0
    ixin, intervals = get_intervals(l1, l2, c1, p2)
    tec = np.nan * stec[0]
    idx = np.where(~np.isnan(l1+c1+p2))
    # lyu, liu, tec[idx] = cycle_slip_corrector(ts[idx], l1[idx], l2[idx], c1[idx], p2[idx], l1_lli_value[idx], l2_lli_value[idx])
    # tec[idx] = relative_tec(ts[idx], c1[idx],
    #                          p2[idx], tec[idx])

    # for r in intervals:
    #     tec[r[0]:r[-1]] = get_phase_corr_tec(l1[r[0]:r[-1]], l2[r[0]:r[-1]], c1[r[0]:r[-1]], p2[r[0]:r[-1]])

    tec = get_phase_corr_tec(l1, l2, c1, p2)
    # temp_stec = clear_jump_data(tec) #TODO необходимо более точно настроить, чтобы работало с 60 сек интервалом
    temp_stec = tec.copy()
    if mode == 0:
        data = {'stec': temp_stec,
                'az': az,
                'el': el,
                'sv': sv,
                'recv': recv_name,
                't': ts}
        df0 = pd.DataFrame.from_dict(data)
        # print(df0)
        df0.dropna(inplace=True)
        return temp_stec, mapping_function(el, h), lat, lon, az, el, df0
    else:
        vtec = stec.copy()
        if df:
            sat_bias = df['ESTIMATED_VALUE'].loc[(df['PRN'] == sv)].values
        if recv_bias and sat_bias:
            bias = (recv_bias + sat_bias[0]) * conversionFactor
            stec_correct = temp_stec + bias
        else:
            stec_correct = temp_stec
        vtec[i, :] = stec_correct * mapping_function(el, h)
        for k, tt in enumerate(d):

            result = f'{str(tt)}, {recv_name},{sv}, {str(lat[k])}, {str(lon[k])}, {str(vtec[i, k])},' \
                     f'{str(temp_stec[k])}, {str(az[k])}, {str(el[k])},{r_lat},{r_lon},{k}\n'
            final_result = final_result + result
            # print(final_result)
        return final_result


def unzipper(path_to_rx: str, pattern: str) -> List[str]:
    """
    Принимает путь к каталогу и шаблону и возвращает список подкаталогов, соответствующих шаблону.

    :param path_to_rx: Путь к папке с файлами .rnx
    :type path_to_rx: str
    :param pattern: Шаблон файла, который вы хотите разархивировать ['.rnx']
    :type pattern: str
    :return: Список подкаталогов.
    """
    subdir = []
    for root, dirs, files in os.walk(path_to_rx):
        for filename in fnmatch.filter(files, pattern):
            zipfile.ZipFile(os.path.join(root, filename)).extractall(os.path.join(root, os.path.splitext(filename)[0]))
            subdir.append(os.path.join(root, os.path.splitext(filename)[0]))
    return subdir


def find_files(path: str, num_of_day: str, pattern: List[str]) -> Tuple[List[str], List[str]]:
    """
    Принимает путь к файлам, номер дня и список шаблонов и возвращает кортеж из двух списков файлов в папке.

    :param path: Путь к каталогу, в котором находятся файлы
    :type path: str
    :param num_of_day: Номер дня, за который вы хотите найти файлы [e.g. '1250' - 125 день]
    :type num_of_day: str
    :param pattern: Шаблон расширения (например, ['22o.gz', '22n'])/ первый - файла наблюдения, второй — файла навигации
    :type pattern: List[str]
    :return: Кортеж из двух списков.
    """
    dot_o = glob.glob(f'{path}/*{num_of_day}*.{pattern[0]}')
    dot_n = glob.glob(f'{path}/*{num_of_day}*.{pattern[1]}')
    return dot_o, dot_n


def multiprocess_driver(params: tuple):
    """
    Принимает кортеж параметров и возвращает строку данных в формате CSV.

    :param params: Парамерты из process_driver
    :type params: tuple
    :return: Строка данных csv
    """
    final_result = ''
    obs = gr.load(params[0], use=params[3], tlim=params[5], useindicators=True)
    obs['time'] = pd.DatetimeIndex(obs['time'].values)
    # obs = obs.resample(time=f'{params[6]}Min').ffill()
    obs = obs.resample(time=f'{params[6]}S').ffill()
    recv_name = params[0][-len(params[4][0]) - 9:-len(params[4][0]) - 5]
    if params[2]:
        recv_bias = params[2]['ESTIMATED_VALUE'].loc[(params[2]['STATION'] == recv_name)].values
    else:
        recv_bias = None
    sv_list = obs.sv.values
    path_obs = (params[0][:-len(params[4][0]) - 1])
    path_n = glob.glob(f'{path_obs}.{params[4][1]}')[0]
    nav1 = gr.load(path_n)
    nav_data = nav1.sel(sv=sv_list)
    for i, sv in enumerate(sv_list):
        csv_data = driver(obs, nav_data, params[5], i, sv, recv_name=recv_name, df=params[2], recv_bias=recv_bias,
                          mode=1)
        final_result += csv_data
    return final_result


def process_driver(dot_o, dot_n, df, type_sat, pattern, t_lim=None, time_step=60):
    """
    Принимает параметры для многопроцессорного вычисления TEC и сохранения в csv

    :param dot_o: Список файлов .o
    :param dot_n: Список файлов .n
    :param df: DataFrame, содержащий данные DCB
    :param type_sat: Тип спутника - "G" для GPS, "R" для ГЛОНАСС, "E" для Galileo
    :param pattern: Шаблон расширения (например, ['22o.gz', '22n'])/ первый - файла наблюдения, второй — файла навигации
    :param t_lim: Срок обработки. Если None, будет обработан весь файл
    :param time_step: Количество минут между наблюдениями
    :return: DataFrame с результатами многопроцессорной обработки.
    """
    final_result = ''
    nn = pd.DataFrame()
    params = [(obs, nav, df, type_sat, pattern, t_lim, time_step) for obs, nav in zip(dot_o, dot_n)]

    with Pool(multiprocessing.cpu_count()) as pr:
        res = pr.map(multiprocess_driver, params)
        pr.close()
        pr.join()

    for r in res:
        final_result += r
        nn = pd.read_csv(StringIO(final_result), names=header)
        nn = nn.dropna(axis=0)

    return nn


def writer(dcb_path: str, output_path: str, rinex_path: str, num_of_day: str, pattern: List[str],
           load_time_lim: Optional[List[str]] = None, type_sat: str = 'G',
           divide: Optional[Union[int, str]] = None) -> None:
    """
    Функция принимает путь к каталогах с файлами dcb и RINEX. Находит файлы RINEX соответсвущие паттерну.
    Производит вычисления process_driver и divide определяет по сколько файлов считывать из каталога

    :param dcb_path: Путь к каталогу, в котором хранятся файлы DCB
    :type dcb_path: str
    :param output_path: Путь к каталогу, в котором вы хотите сохранить выходные файлы
    :type output_path: str
    :param rinex_path: Путь к файлам RINEX
    :type rinex_path: str
    :param num_of_day: Номер дня, за который вы хотите найти файлы [e.g. '1250' - 125 день]
    :type num_of_day: str
    :param pattern: Шаблон расширения (например, ['22o.gz', '22n'])/ первый - файла наблюдения, второй — файла навигации
    :type pattern: List[str]
    :param load_time_lim: Огрничения на время файла  RINEX
    :type load_time_lim: Optional[List[str]]
    :param type_sat: Тип спутника - "G" для GPS, "R" для ГЛОНАСС, "E" для Galileo
    :type type_sat: str (optional)
    :param divide: None, int x если надо разбить на блоки станции по x штук
    :type divide: Optional[Union[int, str]]
    """
    # print('jin'+num_of_day[:3])
    dcb_file = glob.glob(f'{dcb_path}/*{num_of_day[:3]}.csv')
    try:
        df = pd.read_csv(dcb_file[0])
        print(dcb_file[0])
    except:
        print('Not have a dcb file')
        df = None
    dot_o, dot_n = find_files(rinex_path, num_of_day, pattern)
    # print(num_of_day,dot_o)
    match divide:
        case None:
            dot_o3 = dot_o[0:3]
            dot_n3 = dot_n[0:3]
            nn = process_driver(dot_o3, dot_n3, df, type_sat, pattern, load_time_lim)
            os.makedirs(output_path, exist_ok=True)
            num_of_day = num_of_day+'0'
            nn.to_csv(f'{output_path}/TECV_{num_of_day}_prob.csv', index=False)
        case 'hour':
            dot_o3 = dot_o
            dot_n3 = dot_n
            nn = process_driver(dot_o3, dot_n3, df, type_sat, pattern, load_time_lim)
            os.makedirs(output_path, exist_ok=True)
            nn.to_csv(f'{output_path}/TECV_{num_of_day}_prob_byho1.csv', index=False)
        case _:
            div = np.arange(0, len(dot_o), divide)
            for j in range(1, len(div)):
                dot_o3 = dot_o[div[j - 1]:div[j]]
                dot_n3 = dot_n[div[j - 1]:div[j]]
                nn = process_driver(dot_o3, dot_n3, df, type_sat, pattern, load_time_lim)
                os.makedirs(output_path, exist_ok=True)
                nn.to_csv(f'{r""}{output_path}/TECV_{num_of_day}_{div[j - 1]}_{div[j]}.csv', index=False)
