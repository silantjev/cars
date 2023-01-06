# cars
Задача: построить модель классификации, определяющую категорию цены подержанного автомобиля в зависимости от характеристик по выборке из подержанных автомобилей, выставленных на продажу в Соединённых Штатах.
Цель проекта — тренировка построения предсказательной модели.
Источник: Skillbox.ru

Язык: Python 3

Автор: А. В. Силантьев

Файлы:
car_data.csv — данные об автомобилях
cars.ipynb — тетрадь с анализом данных и подбором модели
pipeline.py строит конвеер с выбранной моделью
cars_dill_pipe.pkl — полученный конвеер
local_api.py — локальный api-сервер, работающий на конвеере cars_dill_pipe.pkl
api_test.ipynb — тетрадь с тестом api-сервера
Папка data содержит json-файлы примеров, используемые при тесте api_test.ipynb

Используемые модули:
os, math, warnings, json, pickle, dill=0.3.6, numpy=1.24.0, scipy=1.9.3, pandas=1.5.2, matplotlib=3.6.2, requests=2.28.1, fastapi=0.88.0, pydantic=1.10.2, scikit-learn=1.2.0

Файлы с расширением py следует запускать программой-интерпретатором python3, например:
python3 pipeline.py
или
python pipeline.py

Для запуска файлов с расширением ipynb используется jupyter notebook

Установка модулей:
pip install dill numpy scipy pandas matplotlib requests pydantic scikit-learn "fastapi[all]"
