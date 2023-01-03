import datetime
import numpy as np
import pandas as pd

def datetime_to_tow(t):
    """
    Конвертирует Python datetime обьект в GPS номер недели и время недели.

    Parameters
    ----------
    t : datetime
      Время для конвертации в GPST временной шкале.
    Returns
    -------
    week, tow : tuple (int, float)
      GPS номер недели и время.
    """
    wk_ref = datetime.datetime(2014, 2, 16, 0, 0, 0, 0, None)
    refwk = 1780
    wk = (t - wk_ref).days // 7 + refwk
    tow = ((t - wk_ref) - datetime.timedelta((wk - refwk) * 7.0)).total_seconds()
    return wk, tow


def tow_to_datetime(tow, week):
    """
    Конвертирует GPS номер недели и время недели в Python datetime обьект.

    Parameters
    ----------
    tow : int
      Время недели в секундах.
    weeks : int
      GPS номер недели.
    Returns
    -------
    t : datetime
      Python datetime.
    """
    #  GPS week and TOW to DateTime
    t = datetime.datetime(1980, 1, 6, 0, 0, 0, 0, None)
    t += datetime.timedelta(seconds=tow)
    t += datetime.timedelta(weeks=week)
    return t


def path_glob(path, glob_str='*.Z', return_empty_list=False):
    """
    Возвращает все файлы полным путем, если файл существует в пути,
    если нет, то возвращает FilenotFoundError

    Parameters
    ----------
    path : str
      Путь к файлам.
    glob_str : str
      Расширения  файлов.
    return_empty_list: bool
    Returns
    -------
    files_with_path : list(str)
      Python datetime.

    """
    from pathlib import Path
    #    if not isinstance(path, Path):
    #        raise Exception('{} must be a pathlib object'.format(path))
    path = Path(path)
    files_with_path = [file for file in path.glob(glob_str) if file.is_file]
    if not files_with_path and not return_empty_list:
        raise FileNotFoundError('{} search in {} found no files.'.format(glob_str,
                                                                         path))
    elif not files_with_path and return_empty_list:
        return files_with_path
    else:
        return files_with_path


def getROTI(tec, length):
    """
    Sebastijan Mrak
    Возвращает скорость индекса TEC, рассчитанную как
    стандартное отклонение предоставленного TEC в скользящем
    окне длины 'length'. Возвращает ROTI как  numpy массива.
    """
    roti = []
    for i in range(len(tec) - length):
        roti.append(np.std(tec[i:i + length]))

    return np.array(roti)


def phaseDetrend(y, order, polynom=False):
    """
    Sebastijan Mrak
    Удаление тренда необработанных фазовых данных с использованием функции
    аппроксимации N-го полинома. Выходные данные без тренда — это входные
    данные, вычтенные с приближением полинома. Выходные данные имеют ту же
    длину, что и входные данные 'y'.
    """
    x = np.arange(y.shape[0])
    mask = np.isnan(y)
    z = np.polyfit(x[~mask], y[~mask], order)
    f = np.poly1d(z)
    polyfit = f(x)
    y_d = y - polyfit

    if polynom:
        return y_d, polyfit
    else:
        return y_d


from sklearn.metrics import mean_absolute_error


def MAE(tec, length):
    """
    Sebastijan Mrak
    Возвращщает среднюю абсолютную ошибку предоставленного TEC в скользящем
    окне длины 'length'.
    """
    mae = []
    for i in range(len(tec) - length):
        mae.append(mean_absolute_error(tec[i:i + length], tec[i:i]))
    return np.array(mae)


def roti(tec, lenght):
    """
    Возвращает скорость индекса TEC, рассчитанную как
    стандартное отклонение предоставленного TEC в скользящем
    окне длины 'length'. Возвращает ROTI как  numpy массива.
    """
    return pd.Series(tec).rolling(lenght).std(ddof=0)


def moving_avg(tec, lenght):
    """
    Возвращает скользящее среднее предоставленного TEC в скользящем
    окне длины 'length'(или mse).
    """
    return pd.Series(tec).rolling(window=lenght).mean()

import pymap3d as pm
def getsatElev(recposgeo, satpos):
    """
    Из ECEF координат приемника и спутника расчитывает
    азимут, возвышение и расстояние. Возвращает возвышене и азимут.
    """
    slat, slon, shei = pm.ecef2geodetic(satpos[0], satpos[1], satpos[2], deg=True)
    az, el, r = pm.geodetic2aer(slat, slon, shei, recposgeo[0], recposgeo[1], recposgeo[2], deg=True)
    return (el, az)


from scipy import special
from scipy.special import factorial

def kron_delta(i, j):
  """
  Вовращает дельту Кронекера.
  """
  if i == j:
      return 1
  else:
      return 0


def vtec_sph_harm(longitude, latitude, cos_coeff, sin_coeff, gps_time_s,
                  order_size, degree_size):
  """
  Вовращает VTEC расчитанный через полином Лежандра.
  Возожная альтернатива для get_coef.
  """
  # s = (longitude + gps_time_s * 15.0 / 3600) * np.pi / 180.0
  s = np.mod((longitude * np.pi / 180) + ((gps_time_s - 2 * 60 * 60) * np.pi / 43200),
              2 * np.pi)

  # lat = (90-latitude)*np.pi/180
  lat = (latitude) * np.pi / 180

  vtec = np.zeros(longitude.shape)
  for n in range(order_size):
      # Coefficients with degree higher than order are always 0
      if degree_size < (n + 1):
          deg_range = degree_size
      else:
          deg_range = (n + 1)
      for m in range(deg_range):
          A = np.sqrt(factorial(n - m, exact=True) * (2.0 * n + 1)
                      * (2 - kron_delta(0, m)) / factorial(n + m, exact=True))
          vtec += A * special.lpmv(m, n, np.sin(lat)) * \
              (cos_coeff[n, m] * np.cos(m * s) +
                sin_coeff[n, m] * np.sin(m * s))

  return vtec

#см. библиотеку laika laika/ephemeris.py
def glonass_diff_eq(state, acc):
  J2 = 1.0826257e-3
  mu = 3.9860044e14
  omega = 7.292115e-5
  ae = 6378136.0
  r = np.sqrt(state[0]**2 + state[1]**2 + state[2]**2)
  ders = np.zeros(6)
  if r**2 < 0:
    return ders
  a = 1.5 * J2 * mu * (ae**2)/ (r**5)
  b = 5 * (state[2]**2) / (r**2)
  c = -mu/(r**3) - a*(1-b)

  ders[0:3] = state[3:6]
  ders[3] = (c + omega**2)*state[0] + 2*omega*state[4] + acc[0]
  ders[4] = (c + omega**2)*state[1] - 2*omega*state[3] + acc[1]
  ders[5] = (c - 2*a)*state[2] + acc[2]
  return ders

# Установка библиотеки для чтения и работы с ionex через терминал
# pip install -e git+https://github.com/gnss-lab/ionex.git#egg=ionex
import src.ionex.ionex as ix


def tec_grid(file):
    """
    ionex.reader(file)
    Возвращает читалку файла в формате IONEX. Читалка - итерируемый объект,
    который на каждой итерации возвращает экземпляр IonexMap очередной карты,
    прочитанной из файла.
    tec_grid(file)
    Возвращает значения каждого экземпляра IonexMap карты,
    значений TEC ов карты, эпоху и экспоненту
    """
    inx = ix.reader(file)
    exponent = inx.exponent
    tec = []
    epochs = []
    for i in inx:
        vals = i.tec
        epoch_val = i.epoch
        tec.append(vals)
        epochs.append(epoch_val)
    height = i.height
    print("the altitude of data measured is:", height)
    tec = np.asarray(tec)

    return inx, tec, epochs, exponent


def get_lats_lons(inx):
    """
    Возвращает широту и долготу экземпляра IonexMap карты.
    """
    lat = inx.latitude
    lon = inx.longitude
    lats = []
    lats.append(lat.lat1)
    count_lat = lat.lat1
    for i in range(int((abs(lat.lat1 - lat.lat2) / abs(lat.dlat)))):
        count_lat = count_lat + lat.dlat
        lats.append(count_lat)

    lons = []
    lons.append(lon.lon1)
    count_lon = lon.lon1
    for i in range(int((abs(lon.lon1 - lon.lon2) / abs(lon.dlon)))):
        count_lon = count_lon + lon.dlon
        lons.append(count_lon)
    return lats, lons


def get_tec(file, epoch, lat, lon):
    """
    Возвращает значения TEC для соответствующей широты/долготы в эпоху epoch.
    """
    inx, tec, epochs, exponent = tec_grid(file)
    lats, lons = get_lats_lons(inx)
    tec = tec.reshape(tec.shape[0], len(lats), len(lons))
    # closest_lat = lats[min(range(len(lats)), key = lambda i: abs(lats[i]-lat))]
    # closest_lon = lons[min(range(len(lons)), key = lambda i: abs(lons[i]-lon))]

    # print("Closest latitude:",closest_lat)
    # print("Closest Longitude:",closest_lon)

    lat_index = lats.index(lat)
    lon_index = lons.index(lon)
    print(epochs)
    epoch_index = epochs.index(epoch)

    # print("Epoch and index:",epoch,",",epoch_index)

    tec_val = tec[epoch_index, lat_index, lon_index]
    # tec_val = tec_val*10**(exponent)

    return tec_val


# Применение
# Установка значений/ которые надо прочитать
import glob
ionexs = glob.glob('/content/*.22i')
ionexs.sort()
yyyy = 2022
mm = 5
dd = 5
hh = 0
lat = 53
lon = 27.5
latt = 52.5
lonn = 25.0

# Частота в мин между картами IONEX
mins_list = [0, 15, 30, 45]

# Создание массива для перебора значений с 5 до 12 числа месяца
a = (range(5, 12))
itr1 = iter(a)
itr2 = iter(ionexs[:7])
for k in range(7):
    i = next(itr1)
    d = next(itr2)
    print(i, d)

# Сохранение всех значений tec из карт для выбранных параметров в tec_ion
tec_ion = []
for d, i in zip(a, ionexs):
    print(d, i)
    the_vals = ([h, m] for h in range(24) for m in mins_list)
    for y, z in the_vals:
        print()
        epoch = datetime.datetime(yyyy, mm, d, y, z)
        tec_val = get_tec(i, epoch, latt, lonn)
        tec_ion.append(tec_val)

# Отрисовка tec
import matplotlib.pyplot as plt
plt.plot(tec_ion)
plt.show()

# Отрисовка tec после сглаживания данных
from scipy.interpolate import UnivariateSpline
arr2 = np.arange(0,96)
temp = UnivariateSpline(arr2, tec_ion)
xnew = np.arange(0, 7 * 96)
ynew = temp(xnew)
plt.title("1-D Interpolation")
plt.plot(xnew, ynew, '-', color="green")
plt.show()


# Во всех функциях время должно задаваться  в таком же виде,
# как и в файлах с результатами TEC. т.е "2022-02-02 02:02:02"
import pandas as pd
import branca

def plotter(t, path, bonds=None):
    """
    Считывая необходимое время для отрисовки и путь к файлу с результатами
    lat, lon, t, tec. Находит файл подходящий для заданного времени и загружает
    dataframe с ограничениями  по широте/долготе. Возвращает заполненный
    контур и цветовую шкалу в branca.
    """
    if bonds is None:
        bonds = [20, 38, 42, 60]
    lon_lim0 = bonds[0]
    lon_lim1 = bonds[1]
    lat_lim0 = bonds[2]
    lat_lim1 = bonds[3]
    t = datetime.datetime.strptime(t, '%Y-%m-%d %H-%M-%S').strftime('%Y-%m-%d %H-%M-%S')
    csv_f = glob.glob(f'{path}/*{t}*.csv')
    #
    dataframe = pd.read_csv(csv_f[0])

    sub_data_frame = dataframe.loc[(dataframe['Datetime'] == t)].dropna(axis=0)
    sub_data_frame = sub_data_frame.loc[(sub_data_frame['lat'] >= lat_lim0) & (sub_data_frame['lat'] <= lat_lim1)]
    sub_data_frame = sub_data_frame.loc[(sub_data_frame['lon'] >= lon_lim0) & (sub_data_frame['lon'] <= lon_lim1)]
    x = sub_data_frame['lon'].to_numpy()
    y = sub_data_frame['lat'].to_numpy()
    z = sub_data_frame['vtec'].to_numpy()
    x_min, x_max, y_min, y_max = [min(x), max(x), min(y), max(y)]
    col = 60
    row = 60
    xi = np.linspace(x_min, x_max, col)
    yi = np.linspace(y_min, y_max, row)
    x_i, y_i = np.meshgrid(xi, yi)
    scattered_points = np.stack([x.ravel(), y.ravel()], -1)
    dense_points = np.stack([x_i.ravel(), y_i.ravel()], -1)
    from scipy.interpolate import RBFInterpolator
    interpolation = RBFInterpolator(scattered_points, z.ravel(), smoothing=0, kernel='linear', epsilon=1,
                                    degree=0)
    z_dense = interpolation(dense_points).reshape(x_i.shape)
    colors = ["#48186a", "#424086", "#33638d", "#26828e", "#1fa088", "#3fbc73", "#84d44b", "#d8e219", "#fcae1e"]
    levels = len(colors)
    import matplotlib as mpl
    divnorm = mpl.colors.TwoSlopeNorm(vmin=0., vcenter=30, vmax=60)
    cm = branca.colormap.LinearColormap(colors, vmin=0, vmax=60)
    cf = plt.contourf(y_i, x_i, z_dense, levels, alpha=0.5, norm=divnorm)
    plt.colorbar()
    return cf, cm

# pip install geojsoncontour
import geojsoncontour
import folium
from folium import plugins
def folium_layer(day, t, path_tec, path_out, bonds=None):
    """
    По заданному дню, времени, пути к файлу, содержащему результаты с
    lat, lon, t, tec и пути сохранения результата, переводит контур
    в формат geojson и возвращает folium слой карты.

    # TODO это старая верcия и необходим преобразование координат
    перед нанесением на карту с использованием  geopandas
    """
    date = day + " " + t

    cf, cm = plotter(date, path_tec, bonds=bonds)
    plt.show()
    geojson = geojsoncontour.contourf_to_geojson(
        contourf=cf,
        min_angle_deg=1.0,
        ndigits=2,
        stroke_width=1,
        fill_opacity=0.5)
    print(geojson)
    geomap = folium.Map([53.5, 28], zoom_start=7, tiles="cartodbpositron")  # [54, 30]
    folium.GeoJson(
        geojson,
        style_function=lambda x: {
            'color': x['properties']['stroke'],
            'weight': x['properties']['stroke-width'],
        }).add_to(geomap)
    cm.caption = 'TECU'
    geomap.add_child(cm)
    # Fullscreen mode
    plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)
    geomap.save(path_out + f'/folium_contour_{day}_{t}.html')

import rasterio
from rasterio.transform import Affine

def create_tiff(time, path):
    """
    В заданyое время и пути к файлу, содержащему результаты с
    lat, lon, t, tec, возвращает  растр в формате tiff.
    """
    xi, yi, col, zi = plotter(time, path)
    res = (xi[-1] - xi[0]) / col
    transform = Affine.translation(xi[0] - res / 2, yi[0] - res / 2) * Affine.scale(res, res)
    from datetime import datetime
    rt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    rt = rt.strftime('%Y-%m-%d-%H-%M-%S')
    path = r'im_{}.tif'.format(rt)
    with rasterio.open(
            path,
            'w',
            driver='GTiff',
            height=zi.shape[0],
            width=zi.shape[1],
            count=1,
            dtype=zi.dtype,
            crs='+proj=latlong',
            transform=transform,
    ) as dst:
        dst.write(zi, 1)


def plot_tec_by_csv(path):
  # Костыль, но удобно для быстрой орисовки v/stec-ов
  """
  По пути к результатам tec в csv отрисовывает значение v/stec всех спутников
  """
  from string import ascii_lowercase as alc
  GPS_list = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09', 'G10', 'G11', 'G12',
            'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24',
            'G25', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32']
  dic = 'h' # h - hour, f-full, остальное ничего
  # Если результаты в часовых вариантах 0..24 часов
  # TECV_125{y}_prob_byho1.csv соответвует названию файлов,
  # где y отвечает за необходимый час в букв англ. алфавита
  if dic=='h':
    csv_files = []
    for y in list(alc[0:24]):
      csv_files.append(f'{path}/TECV_125{y}_prob_byho1.csv')
    df_concat = pd.concat([pd.read_csv(f, index_col='Datetime', parse_dates=['Datetime']) for f in csv_files])
  elif dic=='f':
    csv_files = f'{path}/TECV_1250_prob_byho1.csv'
    df_concat = pd.read_csv(csv_files, index_col='Datetime', parse_dates=['Datetime'])
  # Отрисовка кааждого спутника
  fig, ax = plt.subplots(figsize=(8,6))
  for g in GPS_list:
    df0g1 = df_concat.loc[df_concat.PRN == g]
    # .stec можно и отслаьные столбцы из csv
    bp = df0g1.groupby('STATION').stec.plot(ax=ax)

def func(x):
  """
  Добавляет в датафрейм x столбец diff равный разнице времени между
  в мbнутах (60 с). Датафрейм должен иметь datetime64 в индексе.
  """
  x['dif'] = x.index.to_series().diff().dt.total_seconds().div(60, fill_value=0)
  return x

def del_tec_interval(path, file1, file2):
  """
  Пробная функция для удаления интервалов меньше x мин. Лучшая альтернатива
  для clear_jump_dataю при работе с часовыми файлами.
  """
  from string import ascii_lowercase as alc

  data1 = pd.read_csv(file1,index_col='Datetime', parse_dates=['Datetime'])
  data2 = pd.read_csv(file2,index_col='Datetime', parse_dates=['Datetime'])
  data2['ix']+=60
  merged = pd.concat([data1,data2])
  grouped = merged.groupby(['STATION', 'PRN'])
  for name, group in grouped:
    kk = group.ix.diff()
  grouped.apply(func)