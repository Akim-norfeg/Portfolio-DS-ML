# Портфолио проектов

Ниже проекты с кликабельными названиями, кратким описанием и стеком.

1. [Определение возраста клиентов по лицу](URL_проекта_3)  
   **Описание:** Построена модель на основе предобученной ResNet50 для регрессии возраста по фотографиям клиентов с аугментацией и оценкой качества.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `tensorflow.keras` (ImageDataGenerator, ResNet50, Sequential, GlobalAveragePooling2D, Dense, Adam)

2. [Классификация токсичных комментариев](URL_проекта_6)  
   **Описание:** Обучена модель классификации токсичных комментариев с лемматизацией, TF-IDF и BERT-векторизацией, гиперпараметрическим подбором и оценкой по метрике F1.  
   **Стек:** `pandas`, `re`, `warnings`, `numpy`, `matplotlib`, `seaborn`, `spacy`, `pymystem3`, `nltk`, `scikit-learn` (Pipeline, TfidfVectorizer, LogisticRegression, LinearSVC, GridSearchCV, train_test_split, f1_score), `xgboost`, `torch`, `transformers`, `tqdm`

3. [Прогноз снижения активности клиентов интернет-магазина](URL_проекта_5)  
   **Описание:** Построена и оптимизирована классификационная модель для предсказания снижения покупательской активности клиентов с анализом важности признаков.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`, `scikit-learn` (train_test_split, GridSearchCV, RandomizedSearchCV, ColumnTransformer, Pipeline, OneHotEncoder, OrdinalEncoder, SimpleImputer, StandardScaler, MinMaxScaler, RobustScaler, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, SVC, roc_auc_score, permutation_importance)

4. [Анализ удовлетворённости и текучести персонала](projects ds ml/Анализ удовлетворённости и текучести персонала.ipynb)  
   **Описание:** Построены модели предсказания уровня удовлетворённости сотрудников и вероятности их увольнения с подбором гиперпараметров и анализом влияния признаков.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `shap`, `phik`, `scikit-learn` (GridSearchCV, RandomizedSearchCV, ColumnTransformer, SimpleImputer, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler, Pipeline, train_test_split, LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, SVC, метрики)

5. [Отбор коров по удою и вкусу молока](URL_проекта_1)  
   **Описание:** Построены регрессионная и классификационная модели для прогнозирования годового удоя и вкусовых качеств молока и отбора оптимальных коров.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (train_test_split, LinearRegression, LogisticRegression, OneHotEncoder, StandardScaler, метрики)

6. [Прогноз цены недвижимости](URL_проекта_11)  
   **Описание:** Построена LightGBM-модель регрессии логарифма цены недвижимости с предобработкой данных, удалением выбросов и оценкой RMSE.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly.express`, `plotly.graph_objects`, `scipy`, `scikit-learn`, `lightgbm`, `joblib`

7. [Оценка рисков и прибыли при разработке скважин](URL_проекта_2)  
   **Описание:** Построена регрессионная модель для прогнозирования продуктивности скважин и проведён анализ рисков и прибыльности трёх регионов с выбором наилучшего.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (StandardScaler, MinMaxScaler, RobustScaler, ColumnTransformer, SimpleImputer, Pipeline, train_test_split, GridSearchCV, mean_squared_error, LinearRegression)

8. [Оптимизация портфеля методом Монте-Карло](URL_проекта_7)  
   **Описание:** Выполнено скачивание финансовых данных, расчёт доходностей и волатильностей, моделирование 10 000 портфелей Монте-Карло, выбор оптимального по Шарпу и оценка рисков VaR/CVaR.  
   **Стек:** `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`

9. [A/B-тестирование крупного интернет-магазина](URL_проекта_9)  
   **Описание:** Проведен анализ и проверка A/B-теста: приоритизированы гипотезы по ICE/RICE, рассчитаны и визуализированы кумулятивные и ежедневные метрики групп A и B, выявлены выбросы и выполнен статистический тест Mann–Whitney.  
   **Стек:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`, `datetime`

10. [Анализ маркетинговой кампании приложения](URL_проекта_8)  
    **Описание:** Выполнен сбор и очистка данных по визитам, заказам и расходам, рассчитаны метрики CAC, LTV и ROI, проведён анализ конверсии и удержания пользователей по каналам, регионам и устройствам с визуализацией результатов.  
    **Стек:** `pandas`, `numpy`, `datetime`, `matplotlib.pyplot`

11. [Исследование данных сервиса аренды самокатов](URL_проекта_7_eda)  
    **Описание:** Проведен подробный разведочный анализ данных о поездках, тарифах и поведении пользователей сервиса аренды самокатов с формулировкой продуктовых гипотез.  
    **Стек:** `pandas`, `numpy`, `matplotlib`, `mplcyberpunk`, `seaborn`, `plotly.express`, `missingno`, `scipy.stats.gaussian_kde`
