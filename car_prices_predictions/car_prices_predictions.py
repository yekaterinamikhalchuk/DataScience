#!/usr/bin/env python
# coding: utf-8

# # Определение стоимости автомобилей

# ## Постановка задачи

# Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Необходимо построить модель для определения стоимости. 
# 
# Заказчику важны:
# 
# - качество предсказания;
# - скорость предсказания;
# - время обучения.
# 
# 
# **Признаки**
# 
# - `DateCrawled` — дата скачивания анкеты из базы
# - `VehicleType` — тип автомобильного кузова
# - `RegistrationYear` — год регистрации автомобиля
# - `Gearbox` — тип коробки передач
# - `Power` — мощность (л. с.)
# - `Model` — модель автомобиля
# - `Kilometer` — пробег (км)
# - `RegistrationMonth` — месяц регистрации автомобиля
# - `FuelType` — тип топлива
# - `Brand` — марка автомобиля
# - `NotRepaired` — была машина в ремонте или нет
# - `DateCreated` — дата создания анкеты
# - `NumberOfPictures` — количество фотографий автомобиля
# - `PostalCode` — почтовый индекс владельца анкеты (пользователя)
# - `LastSeen` — дата последней активности пользователя
# 
# **Целевой признак**
# 
# - `Price` — цена (евро)

# ## Подготовка данных

# ## Импорт необходимых библиотек и данных

# In[1]:


import pandas as pd 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import seaborn as sns
from sklearn.metrics import make_scorer
from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None 
#Чтобы не появлялось предупреждение SettingWithCopy при масштабировании

import warnings
warnings.filterwarnings('ignore')

import time
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder


# Установим библиотеку lightgbm для дальнейшего построения модели градиентного бустинга

# In[2]:


pip install lightgbm


# Проверка установки:

# In[3]:


import lightgbm as lgb
print(lgb.__version__)


# In[4]:


df = pd.read_csv('/datasets/autos.csv')
df.head()


# ## Обработка датасета: удаление дубликатов, приведение к нижнему регистру 

# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


# print('Количество дублирующих строк в датасете:', df.duplicated().sum())


# In[8]:


df.drop_duplicates(inplace=True)
print('Количество дублирующих строк в датасете:',df.duplicated().sum())


# In[9]:


df.columns = df.columns.str.lower()
df.columns


# In[10]:


df.columns = ['date_crawled', 'price', 'vehicle_type', 'registration_year', 'gearbox',
       'power', 'model', 'kilometer', 'registration_month', 'fuel_type', 'brand',
       'not_repaired', 'date_created', 'number_of_pictures', 'postal_code',
       'last_seen']


# In[11]:


list_for_strlower = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand']
for column in list_for_strlower:
    df[column] = df[column].str.lower() 


# In[12]:


def nan_ratio(column):
    return print('В столбце', column, 'пропущено {:.1%}'.format(df[column].isna().value_counts() [1] / len(df)) + ' значений.')


# In[13]:


listname= ['vehicle_type', 'gearbox', 'fuel_type', 'not_repaired']
for i in listname:
    print(nan_ratio(i))


# ## Обработка пропущенных значений столбцов датасета

# ### `DateCrawled` — дата скачивания анкеты из базы

# In[14]:


#df['date_crawled'] = pd.to_datetime(df['date_crawled'], format='%Y-%m-%d %H:%M:%S')
#df['datecrawled'].dt.year() 
df.info()


# In[15]:


df.drop('date_crawled',axis=1, inplace=True)


# Данный столбец будет не нужен при дальней работе с данными, поэтому его удаляем. 

# ### `RegistrationYear` — год регистрации автомобиля

# In[16]:


df['registration_year'].describe()


# In[17]:


df['registration_year'].value_counts()


# In[18]:


np.percentile(df['registration_year'], [1,99])


# In[19]:


df = df.query('registration_year >= 1980 and registration_year <=2018')
df['registration_year'].describe()


# В столбце `RegistrationYear` присутствуют выбросы. Так как минимальное значение - 0, а максимальное - 9999. Поэтому возьмем 99% всех данных и уберем 1% выбросов. Для дальнейшего анализа **оставим период 1980 - 2018 гг.** 

# ### `Model` — модель автомобиля

# In[20]:


df[df['model'].isna()].sample(3)


# In[21]:


df['model'].isna().sum() / len(df['model'])


# In[22]:


model_mode=df.groupby(['brand'])['model'].agg(pd.Series.mode).to_frame()
model_mode.columns=['model_mode']
model_mode = model_mode.reset_index()
model_mode


# In[23]:


df=df.merge(model_mode, how='left')
df['model'] = df['model'].fillna(df.model_mode)
df['model'] = df['model'].astype('str')


# In[24]:


df.drop(columns='model_mode', inplace=True)


# In[25]:


df.head()


# In[26]:


df['model'].isna().sum()


# В столбце "Model" было пропущенно около 5% строк. Чтобы не терять важные данные, пропуски были заменены на замые частотные значения в зависимости от бренда автомобиля. Была создана отдельная таблица с сгруппированными данными, а затем датасет был объединен с этой таблицей. После замены пропусков лишние столбцы были удалены. 
# 
# Пропуски могли появиться из-за того, что пользователь не указал модель при регистрации. 

# ### `VehicleType` — тип автомобильного кузова

# In[27]:


df['vehicle_type'].value_counts()


# In[28]:


df['vehicle_type'].isna().sum() / len(df['vehicle_type'])


# In[29]:


vehicle_type_mode=df.groupby(['brand','model'])['vehicle_type'].agg(pd.Series.mode).to_frame()
vehicle_type_mode.columns=['vehicle_type_mode']
vehicle_type_mode = vehicle_type_mode.reset_index()
vehicle_type_mode


# In[30]:


df=df.merge(vehicle_type_mode, how='left')
df['vehicle_type'] = df['vehicle_type'].fillna(df.vehicle_type_mode)
df['vehicle_type'] = df['vehicle_type'].astype('str')


# In[31]:


df.drop(columns='vehicle_type_mode', inplace=True)


# In[32]:


df['vehicle_type'].isna().sum() / len(df['vehicle_type'])


# Найдено около 10% пропущенным значений. Все проупски были заменены исходя из наиболее часто встречающихся значений соответствующих моделей и брендов.  

# ### `Gearbox` — тип коробки передач

# In[33]:


df['gearbox'].value_counts(normalize=True)


# In[34]:


df['gearbox'].isna().sum() / len(df['gearbox'])


# In[35]:


gearbox_mode=df.groupby(['brand','model'])['gearbox'].agg(pd.Series.mode).to_frame()
gearbox_mode.columns=['gearbox_mode']
gearbox_mode = gearbox_mode.reset_index()
gearbox_mode


# In[36]:


gearbox_mode.info()


# In[37]:


gearbox_mode['gearbox_mode'].isna().sum()


# In[38]:


df=df.merge(gearbox_mode, how='left')
df['gearbox'] = df['gearbox'].fillna(df.gearbox_mode)
df['gearbox'] = df['gearbox'].astype('str')


# In[39]:


df['gearbox'].isna().sum()


# In[40]:


df.drop(columns='gearbox_mode', inplace=True)


# Обнаружено 5% пропущенных значений. Это важная информация, поэтому решено было заменить пропуски значениями. Замена была произведена наиболее часто встречающимися значениями в зависимости от бренда и модели. После совершения замены ненужные столбцы были удалены. Пропуски могли возникнуть из-за того, что владелец забыл указать данный признак при регистрации.
# Интересно, что 80% всех машин - с механической коробкой передач, верный признак того, что машины достаточно старые. 

# ### `Power` — мощность (л. с.)

# In[41]:


df['power'].hist(range=(0, 600))
plt.title('Гистаграмма частот параметра Power')

# In[42]:


df2=df.groupby(['brand','model'])['power'].median().reset_index()
df.loc[df['power'] < 10, 'power'] = df.loc[df['power'] < 10].merge(df2, on=['brand', 'model'], how='left')['power_y']


# In[43]:


print('Доля значений больше 1000: {:.2%}'.format(df.query('power > 1000')['power'].count() / len(df)))


# In[44]:


df = df.query('power <= 1000') 


# In[45]:


df['power'].describe()


# Было найдено много нулевых значений. Так как такие маленькие значения выглядят нереалистичными, заменим их, а также все, что меньше 10 медианным значением.
# Также видно, что максимальное значение составляет 20000. Достаточно большое значение. 
# Так как средняя мощность составляет 124 лошадиных сил и 110 - по медиане, а мощность всей массы данных не превышает 800 лошадиных сил, ограничим наши данные до 1000 лошадиных сил (к тому же они составляют меньше 1%).

# ### `Kilometer` — пробег (км)

# In[46]:


df['kilometer'].describe()


# In[47]:


df['kilometer'].isna().sum() / len(df)


# Пропусков не обнаружено, также как и необычных значений.

# ### `RegistrationMonth` — месяц регистрации автомобиля

# In[48]:


df['registration_month'].value_counts()


# Столбец "RegistrationMonth" содержит 37352 ячейки со значением 0. Это может означать то, что продавец не указал месяц при публикации объявления. Так как машины не новые и прослужили много лет, месяц не будет оказывать сильного влияния, оставим как есть. 

# In[ ]:





# ### `FuelType` — тип топлива

# In[49]:


fuel_type_mode=df.groupby(['brand','model'])['fuel_type'].agg(pd.Series.mode).to_frame()
fuel_type_mode.columns=['fuel_type_mode']
fuel_type_mode = fuel_type_mode.reset_index()

df=df.merge(fuel_type_mode, how='left')
df['fuel_type'] = df['fuel_type'].fillna(df.fuel_type_mode)
df['fuel_type'] = df['fuel_type'].astype('str')
df.drop(columns='fuel_type_mode', inplace=True)


# ### `Brand` — марка автомобиля

# In[50]:


df['brand'].value_counts().to_frame()


# In[51]:


df['brand'].isna().sum()


# В данном столбце не оказалось строк с пропущенными значениями. Видим, что самыми популярными машинами на сайте стали *Volkswagen, BMW и Opel*. А меньше всего встречались - *Lancia, Trabant, Lada*. Стоит заметить, что встретилась группа машин под названием sonstige_autos, что в переводе с немецкого означает "другие автомобили". Можно предположить, что в эту группу включены все остальные марки, которые не встречаются в таблице. 

# ### `NotRepaired` — была машина в ремонте или нет

# In[52]:


df['not_repaired'].value_counts(normalize=True)


# In[53]:


df['not_repaired'].isna().sum() / len(df)


# In[54]:


not_repaired_mode=df.groupby(['brand','model'])['not_repaired'].agg(pd.Series.mode).to_frame()
not_repaired_mode.columns=['not_repaired_mode']
not_repaired_mode = not_repaired_mode.reset_index()

df=df.merge(not_repaired_mode, how='left')
df['not_repaired'] = df['not_repaired'].fillna(df.not_repaired_mode)
df['not_repaired'] = df['not_repaired'].astype('str')
df.drop(columns='not_repaired_mode', inplace=True)


# Заменим пропуски тем же способом, что и ранее 

# ### `NumberOfPictures` — количество фотографий автомобиля

# In[55]:


df['number_of_pictures'].value_counts()


# In[56]:


df.drop('number_of_pictures', axis=1, inplace=True) 


# ### `PostalCode` — почтовый индекс владельца анкеты (пользователя)

# In[57]:


df.drop('postal_code', axis=1, inplace=True) 


# Удалим данный столбец, так как он не нужен при обучении модели. Он будет лишь мешать её обучению

# ### `Price` — цена (евро)

# In[58]:


df['price'].describe()


# Существуют строки со нулевым значением цены. Выглядит нереалистично. Посмотрим сколько их 

# In[59]:


df['price'].value_counts()

px.histogram(df, 
             x="price",
             marginal = 'box', 
             color_discrete_sequence=['indianred'], 
             title='Гистограмма частот цен на автомобили', 
             labels=dict(price='Цена')
             ).show()


# Примем начальное значение цены 500 евро. Вряд ли машина может стоить меньше.

# In[60]:


df.query('price < 500')['price'].count()


# In[61]:


df = df.query('price >= 500')


# In[62]:


df.info()


# ### `LastSeen` — дата последней активности пользователя

# Данный столбец также можно удалить, так как он не нужен для обучения модели и в дальнейшем использоваться не будет

# In[63]:


df.drop(columns='last_seen', inplace=True)


# ### `DateCreated` — дата создания анкеты

# In[64]:


df.drop(columns='date_created', inplace=True)


# ## Проверка мультиколлинеарности признаков

# Используем кодирование категориальных данных, чтобы далее построить матрицу мультиколлинеарности 
# 

# In[65]:


df.info()


# In[66]:


fig, ax = plt.subplots()
sns.heatmap(df.corr(), vmin=-1, vmax=1, center=0, square=True, annot=True, fmt='.1f')
fig.set_figwidth(10)
fig.set_figheight(10)
plt.title('График мультиколлинеарности признаков датасета', fontsize=14)
plt.show()


# После построения корреляционной модели для количественных факторов, можно сделать вывод, что нет сильно скоррелированных между собой признаков. Видим, что наблюдается небольшая корреляция между ценой и годом регистрации автомобиля, что вполне логично. Но коэффициент корреляции не настолько высок, чтобы нужно было удалять данные. Поэтому оставляем всё, как есть. Данные готовы!

# In[67]:


enc = OrdinalEncoder()
list_enc = ['vehicle_type', 'gearbox', 'model', 'fuel_type', 'brand', 'not_repaired']
enc.fit(df[list_enc])
df[list_enc]= enc.transform(df[list_enc])
df.head()


# In[68]:


df.info()


# ## Вывод по пункту "Подготовка данных"

# <div class="alert alert-block alert-info">
# 
# **После вывода общей информации видим:** 
#  1. 16 столбцов, семь из которых имеют тип данных integer, остальные девять - object.
#  2. В некоторых столбцах пропущено много значений.
# 
#     
# **Было выполнено:**
#   1. Удалены столбцы number_of_pictures, postal_code, last_seen, date_created по причине того, что они не нужны в обучении модели. 
#   2. Удалены 4 дубликата, которые встретились в исходных данных.
#   3. Значение пропусков в столбцах vechile_type, gearbox, not_repaired были заменены наиболее часто встречающимся значением соответствующего столбца в зависимости от модели и бренда автомобиля. 
#   4. Значение пропусков в столбце model были заменены наиболее часто встречающимся значением модели в зависимости от бренда автомобиля. 
#   5. Много встретившись нулевых значением в столбце power, были заменены медианным значение соответствующей модели и бренда автомобиля. Таким способом были заменены все значения до 10 лошадиных сил. Также был создан срез, в котором значение лошадиных сил не превышает 1000. 
#   6. Так как машина не может стоить меньше 500 евро, был создан срез, в который не входят строки с ценой меньше 500 евро. 
#   7. Так как в столбце год регистрации встретились необычные значения, создан срез 1980-2018 гг, такоц период был выбран исходя из 1% выбросов. 
#   8. Категориальные данные подверглись кодированию для построения корреляционной модели. Сильной зависимости признаков друг от друга не выявлено. 
#   9. Было удалено около 18% данные, которые составили выбросы и нулевые значения. 
#   10. Из 16 Столбцов оставили 10
#     
# `Данные готовы к построению моделей!`
#     

# # Обучение моделей

# ## Подготовка данных

# Перед тем, как обучать различные модели, необходимо выделить целевой признак и признаки для обучения, а также разделить данные на две выборки.

# In[69]:


df.head()
df.info()


# In[70]:


target = df['price']
features = df.drop('price', axis=1)


# In[71]:


target
features


# In[72]:


features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345)


# In[73]:


features_train


# In[74]:


len(features_train) / len(features_valid)


# In[75]:


features_name=features_train.columns


# Проверка пройдена, данные разделились в отношении 3:1.

# Произведем масштабирование данных, так как это будет необходимо для обучения линейной регрессии.

# In[76]:


scaler = StandardScaler()
scaler.fit(features_train) 
features_train_st = scaler.transform(features_train)

features_valid_st = scaler.transform(features_valid) 

features_train_st = pd.DataFrame(features_train_st)
features_train_st.head()


# ## Случайный лес

# In[77]:


start_time_rf = time.time()
rmse_rf = 4000
best_est=0
best_model_rf = None
for est in range (40, 50, 5):
    model_rf = RandomForestRegressor(random_state=12345, n_estimators=est, max_depth=3)
    mse = cross_val_score(model_rf, features_train, target_train, cv=5, scoring=make_scorer(mean_squared_error))
    rmse = (mse**0.5).mean()
    if rmse < rmse_rf:
        best_model_rf = model_rf
        rmse_rf = rmse
        best_est = est




print('Метрика RMSE у RandomForestRegressor равна {:.1f} евро'.format(rmse_rf))
print('Количество деревьев равно {:.0f}'.format(best_est))
print('-----------------------------------------------------------------------------')



finish_time_rf = time.time()
time_rf = finish_time_rf - start_time_rf
print('Time spent: {:.2f} sec'.format(time_rf))


# In[ ]:





# ## Дерево решений

# In[78]:


start_time_df_1 = time.time()
model_df_1 = DecisionTreeRegressor(random_state=12345)
start_time_df_1 = time.time()
mse_df_1 = cross_val_score(model_df_1, features_train, target_train, cv=5, scoring=make_scorer(mean_squared_error))
rmse_df_1 = ((mse_df_1)**0.5).mean()
print('Метрика RMSE у DecisionTreeRegressor равна {:.1f} евро'.format(rmse_df_1))
print('-----------------------------------------------------------------------------')
finish_time_df_1 = time.time()
time_df_1 = finish_time_df_1 - start_time_df_1
print('Time spent: {:.2f} sec'.format(time_df_1))


# ## LightGBM

# In[79]:


start_time_gbm = time.time()

rmse_gbm = 4000
best_model_gbm = None
best_est_gbm = 0
best_count_iteration = 0 
for est in range (80, 100, 10):
    model_gbm = LGBMRegressor(random_state=12345, n_estimators=est, depth=3)
    mse_gbm = cross_val_score(model_gbm, features_train, target_train, cv=5, scoring=make_scorer(mean_squared_error), n_jobs=-1)
    rmse = (mse_gbm**0.5).mean()
    if rmse < rmse_gbm:
        best_model_gbm = model_gbm
        rmse_gbm = rmse
        best_est_gbm = est




print('Метрика RMSE у LGBMRegressor равна {:.1f} евро'.format(rmse_gbm))
print('Количество деревьев равно {:.0f}'.format(best_est_gbm))
print('-----------------------------------------------------------------------------')


finish_time_gbm = time.time()
time_gbm = finish_time_gbm - start_time_gbm
print('Time spent: {:.2f} sec'.format(time_gbm))


# Посмотрим значения RMSE модели LightGBM с дефолтными параметрами

# In[80]:


start_time_gbm_1 = time.time()
model_gbm_1 = LGBMRegressor(random_state=12345)
mse_gbm_1 = cross_val_score(model_gbm_1, features_train, target_train, cv=5, scoring=make_scorer(mean_squared_error), n_jobs=-1)
rmse_gbm_1 = ((mse_gbm_1)**0.5).mean()
print('Метрика RMSE у LightGBM равна {:.1f} евро'.format(rmse_gbm_1))
print('-----------------------------------------------------------------------------')
finish_time_gbm_1 = time.time()
time_gbm_1 = finish_time_gbm_1 - start_time_gbm_1
print('Time spent: {:.2f} sec'.format(time_gbm_1))


# Модель с дефолтными параметрами выдает значения лучше и работает в разы быстрее. 

# ## Линейная регрессия

# In[81]:


model_lr_1 = LinearRegression()


# In[82]:


start_time_lr_1 = time.time()
mse_lr_1 = cross_val_score(model_lr_1, features_train, target_train, cv=5, scoring=make_scorer(mean_squared_error))
rmse_lr_1 = ((mse_lr_1)**0.5).mean()
print('Метрика RMSE у LinearRegression равна {:.1f} евро'.format(rmse_lr_1))
print('-----------------------------------------------------------------------------')
finish_time_lr_1 = time.time()
time_lr_1 = finish_time_lr_1 - start_time_lr_1
print('Time spent: {:.2f} sec'.format(time_lr_1))


# # Анализ моделей

# Напишем функцию MAPE. MAPE - средняя абсолютная ошибка в процентах (mean percentage absolute error). Функция показывает в скольки процентах в среднем может ошибаться модель. Будем использовать данную метрику вместе с метрикой RMSE для оценки модели.

# In[83]:


def mape (target, predict):
    mape = (1 / len(target) ) * (( (abs(target - predict)) / (abs(target)) ).sum()) * 100 
    return mape


# Для выявления значимых признаков напишем функцию feature_priority, которая на вход берет датасет, модель и установленное количество значимых признаков, а на выходе получаем график значимости признаков.

# In[84]:


def feature_priority(X, model, n):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns
    d_first = X.shape[1]
    plt.figure(figsize=(8, 8))
    plt.title(f"Значимость признаков модели {model}")
    plt.bar(range(d_first), importances[indices[:d_first]], align='center')
    plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
    plt.xlim([-1, d_first])
    best_features = indices[:n]
    best_features_names = feature_names[best_features]
    print(f'Первые {n} значимых признаков {list(best_features_names)} из {d_first} ')
    plt.show() 


# ## Дерево решений

# In[85]:


start_time_dt_fit = time.time()
model_dt = DecisionTreeRegressor(random_state=12345)
model_dt.fit(features_train, target_train)
finish_time_dt_fit = time.time()
time_dt_fit = finish_time_dt_fit - start_time_dt_fit
print('Time spent: {:.2f} sec'.format(time_dt_fit))


# In[86]:


feature_priority(features_train, model_dt, 3)


# In[87]:


start_time_dt_pred = time.time()
predicted_valid_dt = model_dt.predict(features_valid)
rmse_dt = mean_squared_error(target_valid, predicted_valid_dt)**0.5
print('Метрика RMSE у DesicionTreeRegressor равна {:.1f} евро'.format(rmse_dt))
print('Средняя прогнозируемая цена DesicionTreeRegressor равна {:.1f} евро'.format(predicted_valid_dt.mean()))
print('-----------------------------------------------------------------------------')
finish_time_dt_pred = time.time()
time_dt_pred = finish_time_dt_pred - start_time_dt_pred
print('Time spent: {:.2f} sec'.format(time_dt_pred))


# In[88]:


mape_dt = mape(target_valid, predicted_valid_dt)


# In[89]:


print('Метрика MAPE равна {:.2f}'.format(mape_dt))


# ## Случайный лес

# In[90]:


start_time_rf_1_fit = time.time()
model_rf_1 = RandomForestRegressor(random_state=12345)
model_rf_1.fit(features_train, target_train)

finish_time_rf_1_fit = time.time()
time_rf_1_fit = finish_time_rf_1_fit - start_time_rf_1_fit
print('Time spent: {:.2f} sec'.format(time_rf_1_fit))


# In[91]:


start_time_rf_1_pred = time.time()
predicted_valid_rf_1 = model_rf_1.predict(features_valid)
rmse_rf_1 = (mean_squared_error(target_valid, predicted_valid_rf_1)**0.5).mean()
print('Метрика RMSE у RandomForestRegressor равна {:.1f} евро'.format(rmse_rf_1))
print('Средняя прогнозируемая цена DesicionTreeRegressor равна {:.1f} евро'.format(predicted_valid_rf_1.mean()))
print('-----------------------------------------------------------------------------')
finish_time_rf_1_pred = time.time()
time_rf_1_pred = finish_time_rf_1_pred - start_time_rf_1_pred
print('Time spent: {:.2f} sec'.format(time_rf_1_pred))


# In[92]:


mape_rf = mape(target_valid, predicted_valid_rf_1)


# In[93]:


print('Метрика MAPE равна {:.2f}'.format(mape_rf))


# In[94]:


feature_priority(features_train, model_rf_1, 3)


# ## LightGBM

# In[95]:


start_time_gbm_2_fit = time.time()
model_gbm_2 = LGBMRegressor(random_state=12345)
model_gbm_2.fit(features_train, target_train)

finish_time_gbm_2_fit = time.time()
time_gbm_2_fit = finish_time_gbm_2_fit - start_time_gbm_2_fit
print('Time spent: {:.2f} sec'.format(time_gbm_2_fit))


# In[96]:


start_time_gbm_2_pred = time.time()
predicted_valid_gbm_2 = model_gbm_2.predict(features_valid)
rmse_gbm_2 = (mean_squared_error(target_valid, predicted_valid_gbm_2)**0.5).mean()
print('Метрика RMSE у LightGBM равна {:.1f} евро'.format(rmse_gbm_2))
print('Средняя прогнозируемая цена LightGBM равна {:.1f} евро'.format(predicted_valid_gbm_2.mean()))
print('-----------------------------------------------------------------------------')
finish_time_gbm_2_pred = time.time()
time_gbm_2_pred = finish_time_gbm_2_pred - start_time_gbm_2_pred
print('Time spent: {:.2f} sec'.format(time_gbm_2_pred))


# In[97]:


mape_gbm = mape(target_valid, predicted_valid_gbm_2)


# In[98]:


print('Метрика MAPE равна {:.2f}'.format(mape_gbm))


# In[99]:


feature_priority(features_train, model_gbm_2, 3)


# ## Линейная регрессия

# In[100]:


start_time_lr_fit = time.time()
model_lr = LinearRegression()
model_lr.fit(features_train_st, target_train)

finish_time_lr_fit = time.time()

time_lr_fit = finish_time_lr_fit - start_time_lr_fit
print('Time spent: {:.2f} sec'.format(time_lr_fit))


# In[101]:


start_time_lr_pred = time.time()
predicted_valid_lr = model_lr.predict(features_valid_st)
rmse_lr = (mean_squared_error(target_valid, predicted_valid_lr)**0.5).mean()
print('Метрика RMSE у линейной регрессии равна {:.1f} евро'.format(rmse_lr))
print('Средняя прогнозируемая цена линейной регрессии равна {:.1f} евро'.format(predicted_valid_lr.mean()))
print('-----------------------------------------------------------------------------')

finish_time_lr_pred = time.time()

time_lr_pred  = finish_time_lr_pred  - start_time_lr_pred 
print('Time spent: {:.2f} sec'.format(time_lr_pred))


# In[102]:


mape_lr = mape(target_valid, predicted_valid_lr)


# In[103]:


print('Метрика MAPE равна {:.2f}'.format(mape_lr))


# In[104]:


feature_names=features_train.columns


# In[105]:


importances = np.abs(model_lr.coef_)

indices = np.argsort(importances)[::-1]
d_first = features_train.shape[1]
plt.figure(figsize=(8, 8))
plt.title("Значимость признаков модели линейной регрессии")
plt.bar(range(d_first), importances[indices[:d_first]], align='center')
plt.xticks(range(d_first), np.array(feature_names)[indices[:d_first]], rotation=90)
plt.xlim([-1, d_first])
best_features = indices[:3]
best_features_names = feature_names[best_features]
print(f'Первые 3 значимых признаков {list(best_features_names)} из {d_first} ')
plt.show() 


# ## Сводная таблица

# In[106]:


result_name = ['model_name', 'rmse', 'mape', 'average_price', 'time_spent_fit', 'time_spent_predict']
result_lr = ['LinearRegression', rmse_lr, mape_lr, predicted_valid_lr.mean(), time_lr_fit, time_lr_pred]
result_dt = ['DesicionTreeRegressor', rmse_dt, mape_dt, predicted_valid_dt.mean(), time_dt_fit, time_dt_pred]
result_rf = ['RandomForestRegressor', rmse_rf_1, mape_rf, predicted_valid_rf_1.mean(), time_rf_1_fit, time_rf_1_pred]
result_gbm = ['LightGBMRegressor', rmse_gbm_2, mape_gbm, predicted_valid_gbm_2.mean(), time_gbm_2_fit, time_gbm_2_pred]

df_results = pd.DataFrame([result_lr, result_dt, result_rf, result_gbm], columns=result_name)

df_results = df_results.round(2)
df_results.style.set_caption("Results of the different machine learning models")
df_results = df_results.sort_values('rmse', ascending=True).reset_index()
df_results.drop('index', axis=1, inplace=True)

df_results


# # Вывод

# <div class="alert alert-block alert-info">
# 
# **В проекте были выполнены следующие действия:**
#     
#   1. Проведена предобработка данных, где были удалены лишние столбцы, дублирующиеся строки, а нулевые и аномальные значения были заменены. 
#   2. Данные были стандартизированы для обучения модели с помощью кодирования Encoder и масштабированы с помощью StandardScaler(для обучения линейной регрессии)
#   3. Написана функция для расчета метрики MAPE, которая показывает в скольки процентах в среднем может ошибаться модель.
#   4. Были обучены четыре регрессионные модели машинного обучения, а также подобраны оптимальные параметры для их обучения
#   5. Получены прогнозируемые значения цен на автомобили
#   6. Исходя из прогнозов были рассчитаны две метрики: RMSE, которая выражается в натуральных единицах, а также процентная метрика MAPE 
#   7. Рассчитано время на выполнения обучения и прогнозирования для каждой модели
#  
# **Исходя из полученных данных можно сделать следующий вывод:**   
#   - Наиболее эффективной моделью оказалась модель случайного леса с дефолтными параметрами. Отклонение составило 1638 евро. Это наименьший показатель среди всех моделей. MAPE у модели случайного леса также наименьший и составляет 33%. Это означает, что в 33% случаев модель случайного леса будет допускать ошибку, а в 67% прогнозировать верно. Значение довольно большое, но это лучший результат, которого удалось достичь. 
#   - Второй по эффективности моделей оказалась модель градиентного бустинга LightGBM с дефолтными параметрами. RMSE и MAPE у данной модели ниже, чем у случайного леса, однако скорость опережает почти на 2 секунды. 
