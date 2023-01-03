
# Название проекта

Программа предназначена для расчета STEC/VTEC и DCB значений по данным RINEX 2/3 для GPS


## Установка
Скачать python c https://www.python.org/downloads/ или anaconda https://www.anaconda.com/products/distribution

Для установки проекта можно установить пакеты отдельно
Для работы tec и dcb
```bash
  pip install georinex
  pip install pymap3d
  pip install xarray==0.20.2
  pip install scipy
  pip install python-dotenv
```
Для работы plot
```bash
  pip install geopandas==0.11.1
  pip install PyKrige
  pip install plotly
  
```
 или установить по requirements.txt
```bash
  pip install -r requirements.txt
```
    
## Переменные среды

Для запуска проекта необходимо настроить переменные в .env файле 

`PATH_TO_RINEX`- Путь к каталогу, в котором хранятся файлы RINEX v2/3

`PATH_TO_DCB` - Путь к каталогу, в котором хранятся или будут храниться файлы с DCB

`PATH_TO_SCV`- Путь к каталогу, в котором хранятся или будут храниться результаты вычислений в формате scv

`PATTERN` - Шаблон расширения (например, ['22o.gz', '22n']) / где первый - файла наблюдения, второй — файла навигации

`STRING_DATE` - Номер дня, за который вы хотите найти файлы [e.g. '1250' - 125 день] в str 

(Опционально)
`FILE_PATTERN`
Если файлы заархивированы в .rnx.zip


## Пример использования

После настройки параметров среды в main.py можно запустить в 2х "режимах" - для расчета DCB и STEC/VTEC.

```python
write_mod = 'dcb' #или 'tec' для STEC/VTEC
```
В расчете DCB используется функция writer_dcb из модуля dcb.py
```python
writer_dcb(path_to_rx, path_to_dcb, string_date, pattern, time_step=60)
```
В расчете TEC используется функция writer из модуля tec.py
```python
writer(path_to_dcb, path_to_scv, path_to_rx, string_date, pattern)
```
где параметры path_to_rx, path_to_dcb, string_date, pattern беруться из файла переменных среды или указваются лично в main.py

time_step отвечает за интервалы между вычислениями в секундах. Т.е при ринексе с интервалами по 15с каждое 4 значение будет отвечать 60с. Но указывание времени не кратного интервалу ринекса приведет к ошибке.

!!!
При запуске dcb в первый раз параметр first_start в функции writer_dcb установлен на True, но после необходимо указать first_start=True

```python
writer_dcb(path_to_rx, path_to_dcb, string_date, pattern, time_step=60, first_start=False)
```


## Authors

- [@Tachikoma-r](https://github.com/Tachikoma-r)
