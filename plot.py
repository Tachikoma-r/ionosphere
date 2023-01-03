import multiprocessing
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import plotly.io
from pandas import read_csv
import datetime
import itertools
import geopandas as gpd
from pykrige.ok import OrdinaryKriging
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Polygon
import glob
import logger
from dotenv import load_dotenv
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

import createdb

load_dotenv()

PATH_TO_UI_DATA = os.getenv('PATH_TO_UI_DATA_FROM_IONIC')
PATH_TO_TEC = os.getenv('PATH_TO_SCV')


def pixel2poly(x, y, z, resolution):
    """
    :param x: x координта клетки
    :param y: y координта клетки
    :param z: матрица значений для каждой (x,y)
    :param resolution: пространственное разрешение каждой клетки
    :return: Массив полигонов и значения,соответствующие им
    """
    polygons = []
    values = []
    half_res = resolution / 2
    for i, j in itertools.product(range(len(x)), range(len(y))):
        minx, maxx = x[i] - half_res, x[i] + half_res
        miny, maxy = y[j] - half_res, y[j] + half_res
        polygons.append(Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]))
        if isinstance(z, (int, float)):
            values.append(z)
        else:
            values.append(z[j, i])
    return polygons, values


def bonds(t, path):
    """"
    Из csv файлы считывает значения VTEC и координты.Адаптирует их под координаты Земли.
    Использует библиотеку pykrige для кригинга, чтобы получить растр и перевести его в полигоны

    :param t: Момент времени отрисовки
    :param path: Границы карты
    :return: geopandas.DataFrame в момент времени t
    """
    t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H-%M-%S')
    csv_f = glob.glob(f'{path}')
    print(path)
    if len(csv_f) == 0:
        raise FileNotFoundError("")

    dataframe = read_csv(csv_f[0])
    sub_data_frame = dataframe.loc[(dataframe['Datetime'] == t)].dropna(axis=0)
    # sub_data_frame = sub_data_frame.loc[(dataframe['PRN'] != "G30")].dropna(axis=0)
    # sub_data_frame = sub_data_frame.loc[(dataframe['PRN'] != "G31")].dropna(axis=0)
    # sub_data_frame = sub_data_frame.loc[(dataframe['PRN'] != "G32")].dropna(axis=0)
    x = sub_data_frame['lon'].to_numpy()
    y = sub_data_frame['lat'].to_numpy()
    z = sub_data_frame['vtec'].to_numpy()
    geo = pd.DataFrame({'value': z.tolist(), 'lon': x.tolist(), 'lat': y.tolist()})
    gm = (gpd.GeoDataFrame(geo, crs="EPSG:4326", geometry=gpd.points_from_xy(geo["lon"], geo["lat"]))
          .to_crs("EPSG:3347")
          )
    gm["Easting"], gm["Northing"] = gm.geometry.x, gm.geometry.y
    resolution = 25_000
    gridx = np.arange(gm.bounds.minx.min(), gm.bounds.maxx.max(), resolution)
    gridy = np.arange(gm.bounds.miny.min(), gm.bounds.maxy.max(), resolution)
    krig = OrdinaryKriging(x=gm["Easting"], y=gm["Northing"], z=gm["value"], variogram_model="spherical")
    z, ss = krig.execute("grid", gridx, gridy)
    polygons, values = pixel2poly(gridx, gridy, z, resolution)
    gm_model = (gpd.GeoDataFrame({"TECU": values}, geometry=polygons, crs="EPSG:3347")
                .to_crs("EPSG:4326")
                )
    return gm_model


def plotly_layer(gm_model):
    """
    Наносит значения gpd.DataFrame на карту plotly

    :param gm_model: ГеоДатафрейм состоящий иззначений TEC и полигонов соответвующим им
    """
    fig = px.choropleth_mapbox(gm_model, geojson=gm_model.geometry, locations=gm_model.index,
                               color="TECU", range_color=[0, 50], color_continuous_scale="RdYlGn_r", opacity=0.3,
                               center={"lat": 52.261, "lon": 30}, zoom=4,
                               mapbox_style="carto-positron")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=10))
    fig.update_traces(marker_line_width=0)

    return fig


def ui_plot_layer(date: str, time: str, path_to_tec, path_out: str):
    logger.Log_ui.add_log(f"Creating layer for period '{date} {time}'")
    try:
        plotly_layer(bonds(f'{date} {time}', path_to_tec))
    except FileNotFoundError as er:
        raise er


def mult_create_and_save_fig_to_db(params: tuple):
    createdb.insert_data("plotly_figures", "figures", f"{params[0]} {str(params[1].time())}",
                         plotly.io.to_json(plotly_layer(bonds(str(params[1]), file_name(str(params[1]))))))


def create_and_save_fig_to_csv(date: str):
    try:
        params = []
        for i in pd.date_range(f"{date} 0:00", f"{date} 23:00", freq="15min"):
            params.append((date, i))
        with Pool(multiprocessing.cpu_count()) as pr:
            pr.map(mult_create_and_save_fig_to_db, params)
    except FileNotFoundError as er:
        raise er


def file_name(date: str):
    """
    Находит в PATH_TO_TEC файлы  csv соответсвующего дня date

    :param date: Время за какой день был рсчитан VTEC
    :return: Путь к csv файлу
    """
    date_in = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    path = f'{PATH_TO_TEC}/' \
           f'TECV_{patterned_date(str(date_in.timetuple().tm_yday))}.csv'
    return path


def patterned_date(day_of_year: str):
    patterned_str = ""
    for i in range(1, 4 - len(day_of_year), 1):
        patterned_str += '0'
    patterned_str += f'{day_of_year}0'
    return patterned_str[:4]


def create_scatter_vtec(date: str):
    """
    Отрисовывет график vtec. При расчете быделяет область широт/долгот на 2.5 градуса отличающихся
    от координат станции ples

    :param date: День рсчита VTEC в формате 2022-01-01?
    :return: График VTEC
    """
    csv = pd.read_csv(file_name(f"{date} 00:00:00"))
    dates = []
    for i in pd.date_range(f"{date} 0:00", f"{date} 23:00", freq="15min"):
        dates.append(str(i).replace(":", "-"))
    lat, lon = 54.42096615178241, 27.827572700967604
    csv_res_per = csv.loc[csv["Datetime"].isin(dates)].dropna(axis=0)
    csv_res_per = csv_res_per.loc[(csv_res_per['lat'] <= lat + 2.5) & (csv_res_per['lat'] >= lat - 2.5) &
                                  (csv_res_per['lon'] <= lon + 2.5) & (csv_res_per['lon'] >= lon - 2.5)]
    time_m = []
    vtec_m = []
    print(csv_res_per)
    for i in pd.date_range(f"{str(date).split(' ')[0]} 0:00", f"{str(date).split(' ')[0]} 23:00", freq="15min"):
        csv_m = csv_res_per.loc[csv_res_per['Datetime'] == str(i).replace(':', '-')]
        sum_m = sum(csv_m['vtec']) / len(csv_m['vtec'])
        time_m.append(str(i).replace(':', '-'))
        vtec_m.append(sum_m)
    fig = go.Scatter(x=time_m, y=vtec_m, name="VTEC")
    return fig


def create_scatter_vtec_for_period(date_start: str, date_end: str):
    frames = []
    for date in pd.date_range(date_start, date_end, freq='D'):
        if str(date).split(' ')[0] != '2022-01-05':
            csv = pd.read_csv(file_name(f"{str(date).split(' ')[0]} 00:00:00"))
            dates = []
            for i in pd.date_range(f"{str(date).split(' ')[0]} 0:00", f"{str(date).split(' ')[0]} 23:00", freq="15min"):
                dates.append(str(i).replace(":", "-"))
            csv_res_per = csv.loc[csv["Datetime"].isin(dates)].dropna(axis=0)
            csv_res = csv_res_per.loc[csv_res_per["STATION"] == " ples"]
            csv_res_per = csv_res.sort_values(by=["Datetime"], ascending=True)
            time_m = []
            vtec_m = []
            for i in pd.date_range(f"{str(date).split(' ')[0]} 0:00", f"{str(date).split(' ')[0]} 23:00", freq="15min"):
                print(i)
                csv_m = csv_res_per.loc[csv_res_per['Datetime'] == str(i).replace(':', '-')]
                print(csv_m)
                sum_m = sum(csv_m['vtec'])/len(csv_m['vtec'])
                time_m.append(str(i).replace(':', '-'))
                vtec_m.append(sum_m)
            frames.append(pd.DataFrame({'time': time_m, 'vtec': vtec_m}))

    frame = pd.DataFrame(pd.concat(frames))
    frame = frame.loc[frame['time'] != '2022-01-01 23-15-00']
    frame = frame.loc[frame['time'] != '2022-01-10 23-15-00']
    print(frame)
    scatter = go.Scatter(x=frame['time'], y=frame['vtec'], name='VTEC')
    return scatter, frame['time']


def plotter1(time, path):
    """
    Используя значения VTEC из csv файлов и функцию scipy.interpolate.gridata рисует крту TEC без насения на карту Земли

    :param time: Время в какой момент отрисовать значение VTEC
    :param path: Путь к каталогу срезультатами вычислений в формате csv
    """
    dataframe = read_csv(path)
    sub_data_frame = dataframe.loc[(dataframe['Datetime'] == time)].dropna(axis=0)
    y = sub_data_frame['lat'].to_numpy()
    x = sub_data_frame['lon'].to_numpy()
    z = sub_data_frame['vtec'].to_numpy()
    x_min, x_max, y_min, y_max = [min(x), max(x), min(y), max(y)]
    col = round(abs(x_min - x_max) * 10)
    row = round(abs(y_min - y_max) * 10)
    xi = np.linspace(x_min, x_max, col)
    yi = np.linspace(y_min, y_max, row)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='linear', rescale=True)
    cf = plt.contourf(xi, yi, zi, levels=50, cmap='gist_rainbow', alpha=0.5, antialiased=True)
    plt.show()
