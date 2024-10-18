Репозиторий содержит код для обучения и взаимодействия с моделью crop-transtormer, которая предназначена для предсказания параметров метеостанции в рамках конкурса Agrol.Meteo. 
Модель crop-transtormer размером 35,485,482 параметров работает на архитектуре трансформер с некоторыми модификациями для обработки регрессионных данных и занимается предсказанием временных рядов 7-ми следующих параметров:

|Параметр             | Название                |
|---------------------|-------------------------|
|SOLAR_RADIATION      | Солнечная радиация      |
|PRECIPITATION        | Осадка                  |
|WIND_SPEED           | Скорость ветра          |
|LEAF_WETNESS         | Влажность листа         |
|HC_AIR_TEMPERATURE   | Температура воздуха     |
|HC_RELATIVE_HUMIDITY | Относительная влажность |
|DEW_POINT            | Точка росы              |

Обученная версия модель доступна на платформе [Kaggle](https://www.kaggle.com/) по следующей [ссылке](https://www.kaggle.com/models/danildolgov/crop-transformer) (внутри проекта при запуске demo.py модель загружается автоматически).
Обучение данной модели происходило в течении 2.37 часов (52,398 итераций) на датасете agro_dataset.csv, который был загружен с помощью API по ссылке https://agroapi.xn--b1ahgiuw.xn--p1ai/parameter/.

# Использование
Способ использования модели продемонстрирован в файле demo.py внутри проекта. В нём показывается как подготавливать данные и получать предсказания. 

# Примеры
Большинство из параметров предсказываются моделью довольно хорошо (особенно SOLAR_RADIATION), но всё же присутствуют и те, которые предсказываются хуже (особенно PRECIPITATION и DEW_POINT).
<h3 align="center">Example 1</h3>

![0](https://github.com/user-attachments/assets/c6c956a8-6114-489f-bf26-850a0fa8135c)


<h3 align="center">Example 2</h3>

![1](https://github.com/user-attachments/assets/28bfb2f7-b23b-4db8-b45c-88908a542aac)


<h3 align="center">Example 3</h3>

![2](https://github.com/user-attachments/assets/2184847f-c800-4d3b-a908-136dc021af2f)


<h3 align="center">Example 4</h3>

![3](https://github.com/user-attachments/assets/158e95d3-6d4f-4637-ad93-257e6c0dd54b)
