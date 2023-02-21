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
local_api.py — локальный api-сервис, работающий на конвеере cars_dill_pipe.pkl и использующий библиотеку FastAPI
local_api_flask.py — другая версия локального api-сервиса, использующая библиотеку Flask вместо FastAPI 
api_test.ipynb — тетрадь с тестом api-сервиса
Папка data содержит json-файлы примеров, используемые при тесте api_test.ipynb
requirements.txt — файл с необходимыми библиотеками для всего проекта
requirements_local_api.txt — файл с необходимыми библиотеками для local_api
Dockerfile используется для создания образа сервиса local_api

Используемые библиотеки:
os, math, warnings, json, pickle, dill=0.3.6, numpy=1.24.0, scipy=1.9.3, pandas=1.5.2, matplotlib=3.6.2, requests=2.28.1, fastapi=0.88.0, pydantic=1.10.2, uvicorn=0.20.0, scikit-learn=1.2.0

Файлы с расширением py следует запускать программой-интерпретатором python3, например:
python3 pipeline.py
или
python pipeline.py

Для запуска файлов с расширением ipynb используется jupyter notebook

Установка модулей:
pip install dill numpy scipy pandas matplotlib requests pydantic scikit-learn fastapi uvicorn
или
pip install -r requirements.txt


Команды для docker.

-Создание образа:
docker build -t cars .

-Запуск:
docker run --rm --name api -d -p 8000:8000 cars

-Остоновка:
docker stop api


Благодарности:
Автор благодарен платформе Skillbox, где он обучался на профессию Data Scientist и лично Вячеславу Чарину.
