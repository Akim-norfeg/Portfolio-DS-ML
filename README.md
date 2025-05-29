# Портфолио проектов

Ниже проекты с кликабельными названиями, кратким описанием и стеком.

## 1. [Определение возраста клиентов по лицу](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Определение%20возраста%20клиентов%20по%20лицу.ipynb)  
   **Описание:** Построена модель на основе предобученной ResNet50 для регрессии возраста по фотографиям клиентов с аугментацией и оценкой качества.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow.keras` (ImageDataGenerator, ResNet50, Sequential, GlobalAveragePooling2D, Dense, Adam)

## 2. [Классификация токсичных комментариев](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Классификация%20токсичных%20комментариев.ipynb)  
   **Описание:** Обучена модель классификации токсичных комментариев с лемматизацией, TF-IDF и BERT-векторизацией, гиперпараметрическим подбором и оценкой по метрике F1.  
   **Стек:** `pandas`, `re`, `warnings`, `numpy`, `matplotlib`, `seaborn`, `spacy`, `pymystem3`, `nltk`, `scikit-learn` (Pipeline, TfidfVectorizer, LogisticRegression, LinearSVC, GridSearchCV), `xgboost`, `torch`, `transformers`, `tqdm`

## 3. [Прогноз снижения активности клиентов интернет-магазина](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Прогноз%20снижения%20активности%20клиентов%20интернет-магазина.ipynb)  
   **Описание:** Построена и оптимизирована классификационная модель для предсказания снижения покупательской активности клиентов с анализом важности признаков.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`, `scikit-learn` (GridSearchCV, RandomizedSearchCV, ColumnTransformer, OneHotEncoder, OrdinalEncoder, SimpleImputer, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier)

## 4. [Анализ удовлетворённости и текучести персонала](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Анализ%20удовлетворённости%20и%20текучести%20персонала.ipynb)  
   **Описание:** Построены модели предсказания уровня удовлетворённости сотрудников и вероятности их увольнения с подбором гиперпараметров и анализом влияния признаков.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`, `phik`, `scikit-learn` (SimpleImputer, OneHotEncoder, OrdinalEncoder, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, метрики)

## 5. [Отбор коров по удою и вкусу молока](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Отбор%20коров%20по%20удою%20и%20вкусу%20молока.ipynb)  
   **Описание:** Построены регрессионная и классификационная модели для прогнозирования годового удоя и вкусовых качеств молока и отбора оптимальных коров.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (LinearRegression, LogisticRegression, OneHotEncoder, StandardScaler, метрики)

## 6. [Прогноз цены недвижимости](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Прогноз%20цены%20недвижимости.ipynb)  
   **Описание:** Построена LightGBM-модель регрессии логарифма цены недвижимости с предобработкой данных, удалением выбросов и оценкой RMSE.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly.express`, `plotly.graph_objects`, `scipy`, `scikit-learn`, `lightgbm`, `joblib`

## 7. [Оценка рисков и прибыли при разработке скважин](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Оценка%20рисков%20и%20прибыли%20при%20разработке%20скважин.ipynb)  
   **Описание:** Построена регрессионная модель для прогнозирования продуктивности скважин и проведён анализ рисков и прибыльности трёх регионов с выбором наилучшего.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (ColumnTransformer, SimpleImputer, Pipeline, GridSearchCV, LinearRegression)

## 8. [Оптимизация портфеля методом Монте-Карло](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Оптимизация%20портфеля%20методом%20Монте-Карло.ipynb)  
   **Описание:** Выполнено скачивание финансовых данных, расчёт доходностей и волатильностей, моделирование 10 000 портфелей Монте-Карло, выбор оптимального по Шарпу и оценка рисков VaR/CVaR.  
   **Стек:** `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## 9. [A/B-тестирование крупного интернет-магазина](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20data%20analyst/AB-тестирование%20крупного%20интернет-магазина.ipynb)  
   **Описание:** Проведен анализ и проверка A/B-теста: приоритизированы гипотезы по ICE/RICE, рассчитаны и визуализированы кумулятивные и ежедневные метрики групп A и B, выявлены выбросы и выполнен статистический тест Mann–Whitney.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`, `datetime`

## 10. [Анализ маркетинговой кампании приложения](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20data%20analyst/Анализ%20маркетинговой%20кампании%20приложения.ipynb)  
    **Описание:** Выполнен сбор и очистка данных по визитам, заказам и расходам, рассчитаны метрики CAC, LTV и ROI, проведён анализ конверсии и удержания пользователей по каналам, регионам и устройствам с визуализацией результатов.  
    **Стек:** `pandas`, `numpy`, `datetime`, `matplotlib.pyplot`

## 11. [Исследование данных сервиса аренды самокатов](https://github.com/Akim-norfeg/Portfolio-DS-ML/blob/main/projects%20ds%20ml/Исследование%20данных%20сервиса%20аренды%20самокатов.ipynb)  
    **Описание:** Проведен подробный разведочный анализ данных о поездках, тарифах и поведении пользователей сервиса аренды самокатов с формулировкой продуктовых гипотез.  
    **Стек:** `pandas`, `numpy`, `matplotlib`, `mplcyberpunk`, `seaborn`, `plotly.express`, `missingno`, `scipy.stats.gaussian_kde`
