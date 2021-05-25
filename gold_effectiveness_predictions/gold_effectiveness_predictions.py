#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Цели-и-задачи-проекта" data-toc-modified-id="Цели-и-задачи-проекта-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Цели и задачи проекта</a></span></li><li><span><a href="#Подготовка-данных" data-toc-modified-id="Подготовка-данных-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Подготовка данных</a></span><ul class="toc-item"><li><span><a href="#Импорт-необходимых-библиотек-и-данных" data-toc-modified-id="Импорт-необходимых-библиотек-и-данных-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Импорт необходимых библиотек и данных</a></span></li><li><span><a href="#Обработка-тестовой-выборки" data-toc-modified-id="Обработка-тестовой-выборки-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Обработка тестовой выборки</a></span><ul class="toc-item"><li><span><a href="#Обработка-тренировочной-выборки" data-toc-modified-id="Обработка-тренировочной-выборки-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>Обработка тренировочной выборки</a></span></li></ul></li></ul></li><li><span><a href="#Анализ-данных" data-toc-modified-id="Анализ-данных-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Анализ данных</a></span><ul class="toc-item"><li><span><a href="#Анализ-изменения-концентрации-металлов-на-различных-этапах-очистки" data-toc-modified-id="Анализ-изменения-концентрации-металлов-на-различных-этапах-очистки-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Анализ изменения концентрации металлов на различных этапах очистки</a></span></li><li><span><a href="#Анализ-распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках" data-toc-modified-id="Анализ-распределения-размеров-гранул-сырья-на-обучающей-и-тестовой-выборках-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Анализ распределения размеров гранул сырья на обучающей и тестовой выборках</a></span></li></ul></li><li><span><a href="#Модель" data-toc-modified-id="Модель-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Модель</a></span><ul class="toc-item"><li><span><a href="#Написание-функции-для-вычисления-итоговой-sMAPE" data-toc-modified-id="Написание-функции-для-вычисления-итоговой-sMAPE-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Написание функции для вычисления итоговой <em>sMAPE</em></a></span></li><li><span><a href="#Поиск-наилучшей-модели-для-целевого-признака-rougher.output.recovery" data-toc-modified-id="Поиск-наилучшей-модели-для-целевого-признака-rougher.output.recovery-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Поиск наилучшей модели для целевого признака rougher.output.recovery</a></span></li><li><span><a href="#Поиск-наилучшей-модели-для-целевого-признака-final.output.recovery" data-toc-modified-id="Поиск-наилучшей-модели-для-целевого-признака-final.output.recovery-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Поиск наилучшей модели для целевого признака final.output.recovery</a></span></li><li><span><a href="#Проверка-наилучшей-модели-на-тестовой-выборке" data-toc-modified-id="Проверка-наилучшей-модели-на-тестовой-выборке-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Проверка наилучшей модели на тестовой выборке</a></span></li></ul></li></ul></div>

# # Восстановление золота из руды

# ## Цели и задачи проекта

# Цель: Подготовить прототип модели машинного обучения для «Цифры». Компания разрабатывает решения для эффективной работы промышленных предприятий.Модель должна предсказать коэффициент восстановления золота из золотосодержащей руды. Используйте данные с параметрами добычи и очистки. Модель поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.
# 
# Задачи:
# 
# 1. Подготовить данные;
# 2. Провести исследовательский анализ данных;
# 3. Построить и обучить модель.

# ## Подготовка данных

# ### Импорт необходимых библиотек и данных

# In[1]:


import pandas as pd 
from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import  cross_val_score
from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
#Импортируем необходимые библиотеки
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None 
#Чтобы не появлялось предупреждение SettingWithCopy при масштабировании


# In[2]:


df_train = pd.read_csv('/datasets/gold_recovery_train.csv')
df_test = pd.read_csv('/datasets/gold_recovery_test.csv')
df_full = pd.read_csv('/datasets/gold_recovery_full.csv')
df_train.info()
df_test.info()
#df_full.info()
#Загрузили необходимые файлы


# In[3]:


#Функция для подсчета пропущенных значений
def nan_ratio(data, column):
    return print('Пропущено {:.1%}'.format(data[column].isna().value_counts()[1] / len(data)) + ' значений.')


# ### Обработка тестовой выборки

# In[4]:


#Создали новый датафрейм, состоящий из двух недостающих столбцов в df_test и столбца date, по которому будем объединять.
df_join = df_full[['date', 
                   'rougher.output.recovery', 
                   'final.output.recovery']]


# In[5]:


#Вытянули таргеты из общей выборки в тестовую 
df_test = df_test.merge(df_join, 
              on='date', 
              how='left') 


# In[6]:


df_test['rougher.output.recovery'] = df_test['rougher.output.recovery'].where(df_test['rougher.output.recovery'] != 0, np.nan)
df_test['final.output.recovery'] = df_test['final.output.recovery'].where(df_test['final.output.recovery'] != 0, np.nan)

df_train['rougher.output.recovery'] = df_train['rougher.output.recovery'].where(df_train['rougher.output.recovery'] != 0, np.nan)
df_train['final.output.recovery'] = df_train['final.output.recovery'].where(df_train['final.output.recovery'] != 0, np.nan)

nan_ratio(df_train, 'rougher.output.recovery')


# In[7]:


#Удалим строки, где пропущено значение хотя бы одного из двух целевых признаков
df_test.dropna(subset=['rougher.output.recovery', 
                       'final.output.recovery'], axis=0, inplace=True)
df_test.shape


# In[8]:


df_test= df_test.interpolate(method='nearest')


# In[9]:


df_test.head()


# In[10]:


px.histogram(df_test, 
             x="rougher.output.recovery", 
             marginal = 'box', 
             color_discrete_sequence=['indianred']).show()


# In[11]:


df_test.info()


# In[12]:


df_test.drop('date', axis=1, inplace=True)


# In[13]:


df_test_rougher = df_test.iloc[:, 12:34]
df_test_rougher['rougher.output.recovery'] = df_test['rougher.output.recovery']
df_test_rougher.info()


# #### Обработка тренировочной выборки

# In[14]:


nan_ratio(df_train, 'rougher.output.recovery')


# Для того, чтобы проверить правильность расчета формулы, временно удалим пропущенные строки из нужных нам для расчета столбцов.

# In[15]:


df_train['rougher.output.recovery'] 


# In[16]:


#Создадим временный DataFrame для проверки расчета эффективности обогащения
df_train1 = df_train


# In[17]:


#Функция расчета эффективности обогащения
def recovery (c, f , t):
    recovery = ((c * (f-t)) / (f * (c-t))) * 100
    return recovery


# In[18]:


#Добавляем новый столбец с подсчитанной эффективностью во временный DF для дальнейшего подсчета MAE 
df_train1['rougher.output.recovery_calculated'] = recovery(df_train1['rougher.output.concentrate_au'], 
                                                           df_train1['rougher.input.feed_au'], 
                                                           df_train1['rougher.output.tail_au'])


# In[19]:


#Удаляем все строки с пропущенными значениями в столбце rougher.output.recovery для подсчета MAE
df_train1 = df_train1.dropna(axis='index', 
                             how='any', 
                             subset=['rougher.output.recovery'])


# In[20]:


#Расчитываем среднюю абсолютную ошибку 
print('Средняя абсолютная ошибка между исходными и подсчитанными данными в столбце "rougher.output.recovery" равна {:.17f}'.format(mean_absolute_error
                      (df_train1['rougher.output.recovery_calculated'], 
                       df_train1['rougher.output.recovery'])))


# - Данные тренировочной выборки были изучены и обработаны для подсчета recovery. Написана функция для подсчета. 
# - После удаления строк с пропущенными значениями во временном DataFrame была расчитана средняя абсолютная ошибка с помощью функции mean_absolute_error()
# - Средняя абсолютная ошибка между исходными и подсчитанными данными в столбце "rougher.output.recovery" равна `0.00000000000000966`
# 
# 
# `Вывод:` средняя квадратическая ошибка очень мала, следовательно, можно предположить, что эффективность обогащения рассчитана правильно

# **Изучив признаки в тестовой и тренировочной выборках, выявили, что:** 
# 
# - Существуют признаки, которые есть в тренировочной выборке но недоступны в тестовой. Все они имеют тип данных float64 (с плавающей запятой). 
# 
# - Видим, что недоступны данные, характеризующие параметры продукта (концентрацию металлов, отвальных хвостов, а также показатель recovery) на всех этапах: флотация, первый и второй этапы очистки. 
# 
# - Можем предположить, что признаки, которые есть в тренировочной выборке, но отсутсвуют в тестовой, могут быть рассчитаны до того, как мы получили финальный продукт. Значит, что их нельзя использовать в prediction, а, следовательно, и в обучении. Так как если эти признаки уже посчитаны, значит мы уже имеем финальный продукт и прогнозировать нечего. 
# 
# `Поэтому в дальнейших действих мы займемся удалением лишних столбцов из тренировочной выборки. `
# 
# * Создадим новую переменную, куда поместим очищенную тренировочную выборку

# In[21]:


#Для корректного обучения модели удалим из тренировочной выборки столбцы, которых нет в тестовой 
missing_col = set(df_train.columns) - set(df_test.columns)
df_train_final = df_train.drop(missing_col, axis=1)

print(df_train_final.shape)
print(df_test.shape)


# **Чтобы данные обучились корректно, необходимо обработать пропуски.**
# 
# - Удалим строки где хотя бы в одном из целевых признаков есть пропущенное значение. 
# 
# - Пропуски в остальных признаках заменим методом interpolate()

# In[22]:


#Удалим строки, где пропущено значение хотя бы одного из двух целевых признаков
df_train_final.dropna(subset=['rougher.output.recovery', 
                              'final.output.recovery'], 
                      inplace=True)
df_train_final.shape


# In[23]:


#Заменим пропуски 
df_train_final = df_train_final.interpolate(method='nearest')


# Так как мы имеет два целевых показателя: `rougher.output.recovery` и `final.output.recovery` **необходимо подготовить две обучающие выборки**. 
# 
# Две выборки необходимы, так как в таргет `rougher.output.recovery` входят только показатели этапа флотации, в то время как `final.output.recovery`  затрагивает все стадии процесса получения золота из руды (следовательно, используем польную тренировочную выборку - df_train).

# In[24]:


#Выделим столбцы для обучения данных с таргетом rougher.output.recovery
df_train_rougher = df_train_final.iloc[:, 13:36]
#df_train_rougher.drop('final.output.recovery', axis=1, inplace=True)
df_train_rougher.shape
df_train_rougher.info()


# **`Выводы по части "Подготовка данных":`**
# 
# В данной части проекта были загружены необходимые библиотеки и документы. После загрузки файлов, мы увидели что:
# 
# 1. Количество столбцов тестовой выборки и тренировочной отличаются. 
#     - Были удалены лишние столбцы из тренировочной выборки
# 2. Много нулевых и пропущенных значений в целевых показателях 
#     - Такие строки были удалены
# 3. Много пропущенных значений в остальных признаках 
#     - Пропуски были экстраполированы с помощью ближайшей интерполяции
# 4. Эффективность обогащения рассчитана правильно, так как средняя квадратическая ошибка очень мала
# 

# ## Анализ данных

# ### Анализ изменения концентрации металлов на различных этапах очистки

# In[25]:


concentrate = {'au': ['rougher.output.concentrate_au', 
                      'primary_cleaner.output.concentrate_au', 
                      'final.output.concentrate_au'],
              'ag': ['rougher.output.concentrate_ag', 
                      'primary_cleaner.output.concentrate_ag', 
                      'final.output.concentrate_ag'],
              'pb': ['rougher.output.concentrate_pb', 
                      'primary_cleaner.output.concentrate_pb', 
                      'final.output.concentrate_pb']}
for key in concentrate:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data = df_train[concentrate[key]])
    plt.title(f'Распределение концентрации {key} на различных этапах очистки')
    plt.show()
    


# ### Анализ распределения размеров гранул сырья на обучающей и тестовой выборках

# In[26]:


#Для построения графика distplot, временно удалим пропушенные значения в столбце 'rougher.input.feed_size'
df_test_distplot = df_test.dropna(subset = ['rougher.input.feed_size'], 
                                  axis=0)


# In[27]:


#Строим distplot, чтобы сравнить распределения размеров гранул сырья на обучающей и тестовой выборках
plt.figure(figsize=(6, 6))
sns.distplot(df_train_final['rougher.input.feed_size'], 
             label = 'тренировочная выборка')
sns.distplot(df_test_distplot['rougher.input.feed_size'], 
             label = 'тестовая выборка')
plt.title('Распределение частоты и плотности размеров гранул сырья на обучающей и тестовой выборках')
plt.xlabel('Размер гранул сырья')
plt.ylabel('Количество на интервал')
plt.legend()
plt.show()


# In[28]:


df_train_final['rougher.input.feed_size'].describe()


# In[29]:


df_test_distplot['rougher.input.feed_size'].describe()
df_full.dropna(axis=0, inplace=True)


# In[30]:


#Удалим строки, где пропущено значение хотя бы одного из двух целевых признаков
df_full.dropna(subset=['rougher.output.recovery', 
                              'final.output.recovery'], 
                      inplace=True)

df_full['rougher.concentrate'] = df_full['rougher.output.concentrate_ag'] + df_full['rougher.output.concentrate_pb'] + df_full['rougher.output.concentrate_sol'] + df_full['rougher.output.concentrate_au']
df_full['primary_cleaner.concentrate'] = df_full['primary_cleaner.output.concentrate_ag'] + df_full['primary_cleaner.output.concentrate_pb'] + df_full['primary_cleaner.output.concentrate_sol'] + df_full['primary_cleaner.output.concentrate_au']
df_full['final.concentrate'] = df_full['final.output.concentrate_ag'] + df_full['final.output.concentrate_pb'] + df_full['final.output.concentrate_sol'] + df_full['final.output.concentrate_au']


# In[31]:


x=['rougher.concentrate', 'primary_cleaner.concentrate', 'final.concentrate']
for i in x:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data = df_full[i])
    plt.title(f'Распределение концентрации на этапе {i}')
    plt.show()


# In[32]:


plt.figure(figsize=(6, 6))
sns.distplot(df_full['rougher.concentrate'], label='сырьё')
sns.distplot(df_full['primary_cleaner.concentrate'], label='черновой концентрат')
sns.distplot(df_full['final.concentrate'], label='финальный концентрат')
plt.title('Распределение частоты и плотности концентрации всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах')
plt.xlabel('Концентрат')
plt.ylabel('Количество на интервал')
plt.legend()
plt.show()


# **`Выводы по части "Анализ данных":`**
# 
# - Было изучено **как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки**. Исходя из графика видим, что:
#      - концентрация золота увеличивается после каждого из этапов очистки, что вполне логично, ведь мы этого и добиваемся
#      - концентрация серебра, наоборот, уменьшается, а свинца - увеличивается. 
# 
# 
# - Из графика **распределения частоты и плотности размеров гранул сырья** видим, что:
#      - данные тренировочной и тестовой выборок распределены нормально, 
#      - их средние значения располагаются рядом (со значениями 60 и 55 соответсвенно).
#      
#      *Так как данные распределены нормально со схожей средней, мы исключаем некорректную оценку модели ввиду сильно отличающихся друг от друга распределений. Данные готовы к обучению!*
# 
# - Построив график распределения суммарной концентрации всех веществ на разных стадиях: в сырье, в черновом и финальном концентратах, выявлено, что на на этапе сырья и чернового концентрата много нулевых значений. Принято решение оставить данные в исходном виде, чтобы не исказить конечный результат. 

# ## Модель

# ### Написание функции для вычисления итоговой *sMAPE*

# Напишем функцию вычисления метрики качества — sMAPE. Функция намеренно не считает итоговое значение, так как мы будем использовать ее при обучении моделей для двух выборок. И только после нахождения оптимальных моделей для каждого из двух целевых показателей, мы вычислим итоговую sMAPE

# In[33]:


#Функция принимает на вход таргеты и предсказания, а выдаёт вычисленный общий sMAPE
def smape (target, predict):
    smape = (1 / len(target) ) * ( (abs(target - predict) / ( (abs(target) + abs(predict) ) / 2)).sum()) * 100 
    return smape
#smape_total = 0.25 * smape_rougher + 0.75 * smape_final    


# ### Поиск наилучшей модели для целевого признака rougher.output.recovery

# Будут предприняты следующие шаги: 
# 
# - Стандартизируем выборки двух целевых признаков для более корректного обучения модели.
# 
# - Далее обучим модели линейной регрессии, случайного леса и дерева решений и оценим их качество кросс-валидацией
# 
# - Выберем лучшую модель и проверим ее качество 
# 
# - Сравним получившийся результат с дамми регрессором. 

# In[34]:


#Выделим таргет и остальные признаки
target_rougher_train = df_train_rougher['rougher.output.recovery']
features_rougher_train = df_train_rougher.drop('rougher.output.recovery', axis=1)
target_rougher_test = df_test_rougher['rougher.output.recovery']
features_rougher_test = df_test_rougher.drop('rougher.output.recovery', axis=1)


# In[35]:


numeric = df_train_rougher.drop('rougher.output.recovery', axis=1).columns.tolist()
scaler1 = StandardScaler()
scaler1.fit(features_rougher_train[numeric]) 
features_rougher_train[numeric] = scaler1.transform(features_rougher_train[numeric])

features_rougher_test[numeric] = scaler1.transform(features_rougher_test[numeric]) 

print(features_rougher_train)


# In[36]:


model_LR_rougher = LinearRegression()


r2_lr_mean = cross_val_score(model_LR_rougher, 
                            features_rougher_train, 
                            target_rougher_train,
                            cv=5).mean()

print("Mean R2 from CV of LinearRegression:", r2_lr_mean)

mae_lr_mean = cross_val_score(model_LR_rougher, 
                              features_rougher_train, 
                              target_rougher_train,
                              scoring='neg_mean_absolute_error').mean()

print("Mean MAE from CV of LinearRegression:", mae_lr_mean)

smape_lr_mean = cross_val_score(model_LR_rougher, 
                                features_rougher_train, 
                                target_rougher_train,
                                scoring=make_scorer(smape,
                                                    greater_is_better=False),
                                cv=5).mean()

print("Mean sMAPE from CV of LinearRegression:", smape_lr_mean)


# In[37]:


model_dt_rougher = DecisionTreeRegressor(random_state=12345)


r2_dt_mean = cross_val_score(model_dt_rougher, 
                            features_rougher_train, 
                            target_rougher_train,
                            cv=5).mean()

print("Mean R2 from CV of DecisionTreeRegressor:", r2_dt_mean)

mae_dt_mean = cross_val_score(model_dt_rougher, 
                              features_rougher_train, 
                              target_rougher_train,
                              scoring='neg_mean_absolute_error').mean()

print("Mean MAE from CV of DecisionTreeRegressor:", mae_dt_mean)

smape_dt_mean = cross_val_score(model_dt_rougher, 
                                features_rougher_train, 
                                target_rougher_train,
                                scoring=make_scorer(smape,
                                                    greater_is_better=False),
                                cv=5).mean()

print("Mean sMAPE from CV of DecisionTreeRegressorn:", smape_dt_mean)


# In[38]:


model_rf_rougher = RandomForestRegressor(random_state=12345)


r2_rf_mean = cross_val_score(model_rf_rougher, 
                            features_rougher_train, 
                            target_rougher_train,
                            cv=5).mean()

print("Mean R2 from CV of RandomForestRegressor:", r2_rf_mean)

mae_rf_mean = cross_val_score(model_rf_rougher, 
                              features_rougher_train, 
                              target_rougher_train,
                              scoring='neg_mean_absolute_error').mean()

print("Mean MAE from CV of RandomForestRegressor:", mae_rf_mean)

smape_rf_mean = cross_val_score(model_rf_rougher, 
                                features_rougher_train, 
                                target_rougher_train,
                                scoring=make_scorer(smape,
                                                    greater_is_better=False),
                                cv=5).mean()

print("Mean sMAPE from CV of RandomForestRegressor:", smape_rf_mean)


# ### Поиск наилучшей модели для целевого признака final.output.recovery

# In[39]:


target_final_train = df_train_final['final.output.recovery']
features_final_train = df_train_final.drop('final.output.recovery', axis=1)
target_final_test = df_test['final.output.recovery']
features_final_test = df_test.drop('final.output.recovery', axis=1)


# In[40]:


numeric = df_train_final.drop('final.output.recovery', axis=1).columns.tolist()
scaler2 = StandardScaler()
scaler2.fit(features_final_train[numeric]) 
features_final_train[numeric] = scaler2.transform(features_final_train[numeric])

features_final_test[numeric] = scaler2.transform(features_final_test[numeric]) 

print(features_final_train)


# In[41]:


model_LR_final = LinearRegression()


r2_lr_mean_f = cross_val_score(model_LR_final, 
                            features_final_train, 
                            target_final_train,
                            cv=5).mean()

print("Mean R2 from CV of LinearRegression:", r2_lr_mean_f)

mae_lr_mean_f = cross_val_score(model_LR_final, 
                              features_final_train, 
                              target_final_train,
                              scoring='neg_mean_absolute_error').mean()

print("Mean MAE from CV of LinearRegression:", mae_lr_mean_f)

smape_lr_mean_f = cross_val_score(model_LR_final, 
                                features_final_train, 
                                target_final_train,
                                scoring=make_scorer(smape,
                                                    greater_is_better=False),
                                cv=5).mean()

print("Mean sMAPE from CV of LinearRegression:", smape_lr_mean_f)


# In[42]:


model_dt_final = DecisionTreeRegressor(random_state=12345)


r2_dt_mean_f = cross_val_score(model_dt_final, 
                            features_final_train, 
                            target_final_train,
                            cv=5).mean()

print("Mean R2 from CV of DecisionTreeRegressor:", r2_dt_mean_f)

mae_dt_mean_f = cross_val_score(model_dt_rougher, 
                              features_final_train, 
                              target_final_train,
                              scoring='neg_mean_absolute_error').mean()

print("Mean MAE from CV of DecisionTreeRegressor:", mae_dt_mean_f)

smape_dt_mean_f = cross_val_score(model_dt_rougher, 
                                features_final_train, 
                                target_final_train,
                                scoring=make_scorer(smape,
                                                    greater_is_better=False),
                                cv=5).mean()

print("Mean sMAPE from CV of DecisionTreeRegressorn:", smape_dt_mean_f)


# In[43]:


model_rf_final = RandomForestRegressor(random_state=12345)


r2_rf_mean_f = cross_val_score(model_rf_final, 
                            features_final_train, 
                            target_final_train,
                            cv=5).mean()

print("Mean R2 from CV of RandomForestRegressor:", r2_rf_mean_f)

mae_rf_mean_f = cross_val_score(model_rf_final, 
                              features_final_train, 
                              target_final_train,
                              scoring='neg_mean_absolute_error').mean()

print("Mean MAE from CV of RandomForestRegressor:", mae_rf_mean_f)

smape_rf_mean_f = cross_val_score(model_rf_final, 
                                features_final_train, 
                                target_final_train,
                                scoring=make_scorer(smape,
                                                    greater_is_better=False),
                                cv=5).mean()

print("Mean sMAPE from CV of RandomForestRegressor:", smape_rf_mean_f)


# In[44]:


result_lr_name = ['ModelName', 'sMAPE_rougher', 'sMAPE_final']
result_lr = ['LinearRegression', abs(smape_lr_mean), abs(smape_lr_mean_f)]
result_dt = ['DesicionTree', abs(smape_dt_mean), abs(smape_dt_mean_f)]
result_rf = ['RandomForest', abs(smape_rf_mean), abs(smape_rf_mean_f)]
df_results = pd.DataFrame([result_lr, result_dt, result_rf], columns=result_lr_name)
df_results['sMAPE_total'] = 0.25 * df_results['sMAPE_rougher'] + 0.75 * df_results['sMAPE_final']
df_results
#df_results.index.rename


# ### Проверка наилучшей модели на тестовой выборке

# In[45]:


smape_lr_mean_rougher = cross_val_score(model_LR_rougher, 
                            features_rougher_test, 
                            target_rougher_test,
                                cv=5).mean()

mape_rf_mean_final = cross_val_score(model_LR_final, 
                                features_final_test, 
                                target_final_test,
                                scoring=make_scorer(smape,
                                                    greater_is_better=False),
                                cv=5).mean()



# In[46]:


smape_overall = 0.25 * abs(smape_lr_mean_rougher) + 0.75 * abs(mape_rf_mean_final)


# In[47]:


dummy_rougher = DummyRegressor(strategy='median')
dummy_final = DummyRegressor(strategy='median')
dummy_rougher.fit(features_rougher_train,target_rougher_train)
dummy_final.fit(features_final_train,target_final_train)
dummy_predict_rougher = dummy_rougher.predict(features_rougher_test)
dummy_predict_final = dummy_final.predict(features_final_test)
dummy_final = (0.25*smape(target_rougher_test, dummy_predict_rougher) + 0.75* smape(target_final_test, dummy_predict_final)).round(2)


    


# In[48]:


print('Метрика sMAPE лучшей модели на тестовой выборке равна', smape_overall)
print('Метрика sMAPE на DummyRegressor равна', dummy_final)
if dummy_final > smape_overall:
    print('Лучшая модель эффективнее DummyRegressor, ее можно использовать!')
else:
    print('Модель бесполезна')


# <div class="alert alert-block alert-info">
# 
# **В ходе проекта были выполнены следующие действия:**
#     
# 1. Получена общая информация о данных
# 2. Проведена предобработка данных
#     - *удалены лишние столбцы из тренировчной выборки*
#     - *удалены строки с нулевыми значениями в целевых признаках*
#     - *пропущенные значения в признаках экстраполированы*
# 3. Вычислив MAE между данной эффективностью обогощения и расссчитанной на обучающей выборке для признака `rougher.output.recovery`, выявлено, что эффективность обогащения рассчитана правильно.
# 4. Изучено как меняется концентрация металлов (Au, Ag, Pb) на различных этапах очистки
#      - *концентрация золота увеличивается после каждого из этапов очистки, что вполне логично, ведь мы этого и добиваемся*
#      - *концентрация серебра, наоборот, уменьшается, а свинца - увеличивается.* 
# 5. Построена гистограмма для сравнения распределения размеров гранул сырья на обучающей и тестовой выборках:
#     - *данные тренировочной и тестовой выборок распределены нормально*
#     - *их средние значения располагаются рядом (со значениями 60 и 55 соответсвенно)*
# 6. Обучающая и тестовая выборки были разделены на две части для двух таргетов в зависимости от того, на каком этапе расчитывается показатель. 
# 7. Данные были стандартизированы путем масштабирования для последующего более корректного обучения модели.
# 8. Написана функция для вычисления итоговой sMAPE
# 9. Обучены разные модели и бфло оценено их качество кросс-валидацией. Исходя из этого были выбраны лучшие модели для каждой из выборок.
# 10. Пройдена проверка на тестовой выборке, результат был сравнен с DummyRegressor
# 
# 
# Исходя из всех вышеописанных действий, можно сделать следующий вывод: 
# 
# - Наиболее эффективной моделью для получения целевого показателя этапа сырья является линейная регрессия, в то время как для окончательного целевого показателя был выбран случайный лес. 
# - Выбранные модели показали большую эффективности, чем случайная модель дамми. 
#  
# **`Итог: была создана модель машинного обучения, которая предсказывает коэффициент восстановления золота из золотосодержащей руды. Данная модель рекомендована к использованию. Она поможет оптимизировать производство, чтобы не запускать предприятие с убыточными характеристиками.`**
