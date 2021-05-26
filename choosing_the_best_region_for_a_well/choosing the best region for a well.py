#!/usr/bin/env python
# coding: utf-8

# # Выбор локации для скважины

# ## Цель и задачи

# Добывающая компания «ГлавРосГосНефть». Предоставлены пробы нефти в трёх регионах: в каждом 10 000 месторождений, где измерили качество нефти и объём её запасов. 
# 
# **Цель:** построить модель машинного обучения, которая поможет определить регион, где добыча принесёт наибольшую прибыль. Проанализировать возможную прибыль и риски техникой *Bootstrap.*
# 
# **Задачи:**
# - Провести предобработку данных
# - В избранном регионе найти месторождения, для каждого определить значения признаков;
# - Построить модель и оценить объём запасов;
# - Выбрать месторождения с самым высокими оценками значений.
# - Расчитать риски
# - Обозначить выводы
# 

# ## Загрузка и подготовка данных

# ### Загрузка и изучение данных

# In[1]:


import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
import seaborn as sns
from scipy import stats as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#Импортируем необходимые библиотеки
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None 
#Чтобы не появлялось предупреждение SettingWithCopy при масштабировании


# In[2]:


data1 = pd.read_csv('/datasets/geo_data_0.csv')
data2 = pd.read_csv('/datasets/geo_data_1.csv')
data3 = pd.read_csv('/datasets/geo_data_2.csv')
data1.info()
data2.info()
data3.info()


# In[3]:


data1.head()


# In[4]:


data2.head()


# In[5]:


data3.head()


# In[6]:


print(data1.duplicated().sum())
print(data2.duplicated().sum())
print(data3.duplicated().sum())


# In[7]:


data1.describe()


# In[8]:


data2.describe()


# In[9]:


data3.describe()


# <div class="alert alert-block alert-info">
# 
# В данной части проекта были загружены необходимые библиотеки и документы. После загрузки файлов, мы увидели что:
# 1. Все три файла имеют 100000 строк и 2 типа данных: `object`(столбец id) и `float` (остальные числовые данные)
# 2. Дубликатов и пропусков не обнаружено, значит можно готовить данные для обучения модели 
# 3. После вызова метода describe() выявлено, что распределение данных во втором регионе очень отличается от других. 

# ### Предобработка данных

# In[10]:


data1 = data1.drop('id', axis=1) 
data2 = data2.drop('id', axis=1) 
data3 = data3.drop('id', axis=1) 


# In[11]:


fig, ax = plt.subplots()
sns.heatmap(data1.corr(), vmin=-1, vmax=1, center=0, square=True, annot=True)
fig.set_figwidth(5)
fig.set_figheight(5)
plt.title('График мультиколлинеарности признаков в регионе №1', fontsize=14)
plt.show()


# In[12]:


fig, ax = plt.subplots()
sns.heatmap(data2.corr(), vmin=-1, vmax=1, center=0, square=True, annot=True)
fig.set_figwidth(5)
fig.set_figheight(5)
plt.title('График мультиколлинеарности признаков в регионе №2', fontsize=14)
plt.show()


# In[13]:


fig, ax = plt.subplots()
sns.heatmap(data3.corr(), vmin=-1, vmax=1, center=0, square=True, annot=True)
fig.set_figwidth(5)
fig.set_figheight(5)
plt.title('График мультиколлинеарности признаков в регионе №3', fontsize=14)
plt.show()


# <div class="alert alert-block alert-info">
# 
# - Так как в следующем пункте необходимо будет стандартизировать признаки, нам нужно было удалить столбец 'id', иначе он бы служил помехой и модель выдавала бы некорректные результаты. 
# 
# - После проверки признаков на мультиколлинеарность выявили, что сильной зависимости между признаками не наблюдается (будем считать зависимость сильной, если коэффициент корреляции >0.5 или <-0.5). 
# 
# - Во втором регионе наблюдается коэффициент корреляции = 1 между признаком f2 и целевым показателем, это поможет исследованию. 
# 
# **`Следовательно, ничего не удаляем, оставляем данные в исходном виде.`**

# ## Обучение и проверка модели

# ### Регион №1

# In[14]:


#Выделим признаки и целевой признак
features1 = data1.drop('product', axis=1)
target1 =  data1['product'] # Целевой признак

features1_train, features1_valid, target1_train, target1_valid = train_test_split(features1, 
                                                                                  target1, 
                                                                                  train_size=0.75, 
                                                                                  random_state=12345) 


# In[15]:


numeric = ['f0', 'f1', 'f2']
scaler1 = StandardScaler()
scaler1.fit(features1_train[numeric]) 
features1_train[numeric] = scaler1.transform(features1_train[numeric])

features1_valid[numeric] = scaler1.transform(features1_valid[numeric]) 

print(features1_train)


# In[16]:


model1 = LinearRegression()
#Обучили модель на тренировочной выборке
model1.fit(features1_train, target1_train) 
#Получили предсказания модели на валидационной выборке, сохранили в переменную
predictions1 = model1.predict(features1_valid)

result1 = mean_squared_error(target1_valid, predictions1) ** 0.5
#Посчитали значение метрики RMSE на валидационной выборке
print("RMSE модели линейной регрессии на валидационной выборке:", result1, 'тыс. баррелей')
#Средний запас прогнозируемого сырья'
mean_predicted_volume1 = predictions1.sum() / len(predictions1)
print("Средний запас прогнозируемого сырья:", mean_predicted_volume1, 'тыс. баррелей')


# In[17]:


geo1_predict = pd.DataFrame()
geo1_predict['product_geo1'] = target1_valid
geo1_predict['predictions_geo1'] = predictions1


# In[18]:


#Строим гистограмму частот объёмом запасов в скважинах для действительных и прогнозируемых значений
plt.figure(figsize=(20,10))
plt.figure(figsize=(15,10))
geo1_predict['product_geo1'].hist(grid=True, label='фактическое значение')
geo1_predict['predictions_geo1'].hist(grid=True, label='прогнозируемое значение')

plt.legend(fontsize=17)
plt.xlabel('Объём запасов (тыс. бареллей)',
         fontsize=15)
plt.ylabel('Частота',
         fontsize=15)
plt.title('Гистограмма частот объемов запасов в скважинах региона №1 (действительные vs прогнозируемые значения)',
         fontsize=17)


# ### Регион №2

# In[19]:


features2 = data2.drop('product', axis=1)
target2 =  data2['product'] 

features2_train, features2_valid, target2_train, target2_valid = train_test_split(features2, 
                                                                                  target2, 
                                                                                  train_size=0.75, 
                                                                                  random_state=12345) 


# scaler2 = StandardScaler()
# scaler2.fit(features2_train[numeric]) 
# features2_train[numeric] = scaler2.transform(features2_train[numeric])
# 
# features2_valid[numeric] = scaler2.transform(features2_valid[numeric]) 
# 
# print(features2_train)

# In[20]:


model2 = LinearRegression()

model2.fit(features2_train, target2_train) 

predictions2 = model2.predict(features2_valid) 

result = mean_squared_error(target2_valid, predictions2) ** 0.5
mean_predicted_volume2 = predictions2.sum() / len(predictions2)
print("RMSE модели линейной регрессии на валидационной выборке:", result, 'тыс. баррелей')
print("Средний запас прогнозируемого сырья:", mean_predicted_volume2, 'тыс. баррелей')


# In[21]:


geo2_predict = pd.DataFrame()

geo2_predict['product_geo2'] = target2_valid
geo2_predict['predictions_geo2'] = predictions2


# In[22]:


plt.figure(figsize=(20,10))
plt.figure(figsize=(20,10))
geo2_predict['product_geo2'].hist(grid=True, label='фактическое значение')
geo2_predict['predictions_geo2'].hist(grid=True, label='прогнозируемое значение')

plt.legend(fontsize=17)
plt.xlabel('Объём запасов (тыс. бареллей)',
         fontsize=15)
plt.ylabel('Частота',
         fontsize=15)
plt.title('Гистограмма частот объемов запасов в скважинах региона №2 (действительные vs прогнозируемое значения)',
         fontsize=17)


# ### Регион №3

# In[23]:


features3 = data3.drop('product', axis=1)
target3 =  data3['product'] 

features3_train, features3_valid, target3_train, target3_valid = train_test_split(features3, 
                                                                                  target3, 
                                                                                  train_size=0.75, 
                                                                                  random_state=12345) 


# In[24]:


scaler3 = StandardScaler()
scaler3.fit(features3_train[numeric]) 
features3_train[numeric] = scaler3.transform(features3_train[numeric])

features3_valid[numeric] = scaler3.transform(features3_valid[numeric]) 

print(features3_train)


# In[25]:


model3 = LinearRegression()

model3.fit(features3_train, target3_train) 

predictions3 = model3.predict(features3_valid) 

result = mean_squared_error(target3_valid, predictions3) ** 0.5
mean_predicted_volume3 = predictions3.sum() / len(predictions3)
print("RMSE модели линейной регрессии на валидационной выборке:", result, 'тыс. баррелей')
print("Средний запас прогнозируемого сырья:", mean_predicted_volume3, 'тыс. баррелей')


# In[26]:


geo3_predict = pd.DataFrame()
geo3_predict['product_geo3'] = target3_valid
geo3_predict['predictions_geo3'] = predictions3
geo3_predict


# In[27]:


plt.figure(figsize=(20,10))
geo3_predict['product_geo3'].hist(grid=True, label='фактическое значение')
geo3_predict['predictions_geo3'].hist(grid=True, label='прогнозируемое значение')

plt.legend(fontsize=17)
plt.xlabel('Объём запасов (тыс. бареллей)',
         fontsize=15)
plt.ylabel('Частота',
         fontsize=15)
plt.title('Гистограмма частот объемов запасов в скважинах региона №3 (действительные vs прогнозируемые значения)',
         fontsize=17)


# <div class="alert alert-block alert-info">
#     
# **Выводы по пункту 3:**
# 1. После подсчета **RMSE** («корень из средней квадратичной ошибки») увидели, что в первом и третьем регионах данные схожи. Наблюдается отклонение в 37 и 40 тыс. баррелей соответсвенно. В то время как во втором регионе этот показатель равен всего 0.89 тыс. баррелей.
# 2. Распределения объемов запасов первого и третьего регионов близки к нормальному, однако распределение второго региона сильно отличное от нормального. 
# 3. Средний запас прогнозируемого сырья первого и третьего региона также схожи и составляют 92 и 94 тыс. баррелей соответсвенно
# 4. У второго региона Средний запас прогнозируемого сырья равен 68 тыс. баррелей, что значительно меньше первого и второго регионов. Однако, из-за специфических данных, прогноируемые значения более точные, чем во втором и третьем регионах, так как средний разброс значений равен 0,89 тыс. баррелей

# ## Подготовка к расчёту прибыли

# In[28]:


budget = 10**10
price_per_barrel = 450000
wells = 200
research_n = 500


# In[29]:


geo1_predict['price'] = geo1_predict['product_geo1'] * price_per_barrel 
geo2_predict['price'] = geo2_predict['product_geo2'] * price_per_barrel 
geo3_predict['price'] = geo3_predict['product_geo3'] * price_per_barrel 


# In[30]:


print('Точка безубыточности равна {:.0f} денежных единиц или {:.0f} тыс. баррелей нефти'      .format((budget/wells),(budget/(price_per_barrel*wells))))


# In[31]:


#target - фактические производительности скважи
#probabilities - прогнозируемые производительности скважи
#count - количество скважин, которые выбираем
def profit(target, probabilities, count):
    probs_sorted = probabilities.sort_values(ascending=False)
    selected = target[probs_sorted.index][:count]
    revenue = (selected.sum() * price_per_barrel - budget) / 10**3
    return revenue


# В данном пункте была найдена точна безубыточности, то есть сколько баррелей нефти необходимо производить каждой скважине, чтобы покрывать все свои затраты. Из расчета того, что мы можем выбрать только 200 скважин для разработи, получается, что из выделенных 10 млрд. на каждую скважину приходится 50млн. А исходя из цены за баррель в 450000 руб., можно рассчитать, что:
# 
# **`Каждая скважина должна производить в среднем 111 тыс. баррелей нефти для покрытия возникших затрат для её разработки`**
# 
# В данном пункте также была написана функция для расчета прибыли. Прогнозированные данные были отсортированы по убыванию. Из них сделали срез 200 скважин (столько скважин для разработки выберут в конечном итоге по условию проекта). Далее расчитали прибыль для этих 200 скважин (из выручки(количество`*`цена) вычли операционные затраты)
# 
# Далее применим данную функцию для расчета прибыли по всем регионам

# ## Расчёт прибыли и рисков 

# In[32]:


max_profit1 = profit(target1_valid, geo1_predict['predictions_geo1'], wells)
max_profit2 = profit(target2_valid, geo2_predict['predictions_geo2'], wells)
max_profit3 = profit(target3_valid, geo3_predict['predictions_geo3'], wells)


# In[33]:


print('Прибыль двухсот наиболее богатых нефтью скважин региона 1 составит {:.2f} тыс. денежных единиц'.format(max_profit1))
print('Прибыль двухсот наиболее богатых нефтью скважин региона 2 составит {:.2f} тыс. денежных единиц'.format(max_profit2))
print('Прибыль двухсот наиболее богатых нефтью скважин региона 3 составит {:.2f} тыс. денежных единиц'.format(max_profit3))


# In[34]:


#target - фактические производительности скважи
#predictions - прогнозируемые производительности скважи
#well_count - количество скважин, которые выбираем
#sample_n - количество исследуемых скважин при разведке

def bootstrap(target, predictions, sample_n, well_count):
    state = np.random.RandomState(12345)

    values = []

    for i in range(1000): #Применили технику Bootstrap с 1000 выборок
        target_subsample = target.sample(n=sample_n, replace=True, random_state=state)
        probs_subsample = predictions[target_subsample.index]
        values.append(profit(target_subsample, probs_subsample, wells))
    
    values = pd.Series(values)
    
    lower = values.quantile(0.025) 
    mean = values.mean() 
    upper = values.quantile(0.975) 
    risk = st.percentileofscore(values, 0) 
    


    print('Средняя выручка региона {:.2f} тыс. денежных единиц'.format(mean))
    print('95% доверительный интервал: ({:.2f}) - ({:.2f}) тыс. денежных единиц'.format(lower, upper))
    print('Риск убытка региона: {:.2}%'.format(risk))


# In[35]:


print('Данные по региону №1')
print('')
bootstrap(target1_valid, geo1_predict['predictions_geo1'], research_n, wells)


# In[36]:


print('Данные по региону №2')
print('')
bootstrap(target2_valid, geo2_predict['predictions_geo2'], research_n, wells)


# In[37]:


print('Данные по региону №3')
print('')
bootstrap(target3_valid, geo3_predict['predictions_geo3'], research_n, wells)


# ## Общий вывод

# <div class="alert alert-block alert-info">
# 
# **В ходе проекта были выполнены следующие действия:**
#     
# 1. Получена общая информация о данных
# 2. Проведена предобработка данных
#     - *удалены лишние столбцы*
#     - *проведена проверка на дубликаты*
#     - *данные были стандартизированы методом масштабирования*
# 3. Данные были поделены на две выборки (тренировочную и валидационную) в отношении **75:25**
# 4. Вызвана и обучена на тренировочных данных модель линейной регрессии, также были получены прогнозируемые значения для каждого из трех регионов 
# 5. Высчитана метрика **RMSE** для каждого региона и среднее количество запасов для региона
# 6. Рассмотрены распределения прогнозируемых и фактических запасов скважин каждого из регионов
# 7. Написана и применена функция для расчета прибыли 200 наиболее богатых скважин в каждом регионе
# 8. С помощью процедуры **Bootstrap** написана и применена функция для расчета границ 95% доверительного интервала, средней выручки региона и вероятности убытка
# 
# 
# Исходя из всех вышеописанных действий, можно сделать следующий вывод: 
# 
# `Несмотря на специфичность данных, наиболее выгодным для разработки месторождений нефти является регион под номером 2.2`
# - В регионе самый высокий показатель средней выручки, который составляет 515,222 млн. денежных единиц, когда в других регионах он составляет 425 и 435 млн. денежных единиц.
# - В данном регионе риск убытка наименьший и составляет всего 1%, когда в остальных двух вероятность убытка равно 6 и 6,4%. 
# - После расчета прибыли 200 наиболее богатых скважин в регионах видим, что во втором регионе этот показатель является наименьшим. Однако, такой показатель достигается ввиду того, что во втором регионе данные распределены более скучено и показатель RMSE равен всего 0,89 тыс. денежных единиц. То есть отклонение составляет меньше одной тясячи ден. единиц в то время как у региона №1 и №3 данный показатель 37 и 40 тыс. денежных единиц соотвественно. 
# 
# - Регион №1 и №3 исходя из полученных данных являются достаточно схожими по показателям. Однако, вторым по перспективности регион я бы выделил регион №1 ввиду того, что риск убытка составляет 6%, что меньше 6,4 в третьем регионе. Однако, в первом регионе меньше выручка на 10 млн. денежных единиц, а средний запас прогнозируемого сырья составляет 92 тыс. баррелей, а в третьем регионе 94 тыс баррелей. Но в данных регионах корень из квадратичной ошибки также высок - 37 и 40 тыс. баррелей соответственно. (а это составляет 40% и 42% от среднего запаса прогнозируемого сырья - что достаточно много)  
# 
# **`Вывод: добывающей компании «ГлавРосГосНефть» предложено бурить новую скважину в регионе №2 ввиду вышеизложенных фактов`**
