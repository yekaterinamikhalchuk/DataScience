#!/usr/bin/env python
# coding: utf-8

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Цель-и-задача" data-toc-modified-id="Цель-и-задача-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Цель и задача</a></span></li><li><span><a href="#Импорт-необходимых-библиотек" data-toc-modified-id="Импорт-необходимых-библиотек-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Импорт необходимых библиотек</a></span></li><li><span><a href="#Получение-первичной-информации-о-данных" data-toc-modified-id="Получение-первичной-информации-о-данных-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Получение первичной информации о данных</a></span></li><li><span><a href="#Предобработка-данных" data-toc-modified-id="Предобработка-данных-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Предобработка данных</a></span></li><li><span><a href="#Преобразование-данных-для-обучения-модели" data-toc-modified-id="Преобразование-данных-для-обучения-модели-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Преобразование данных для обучения модели</a></span></li><li><span><a href="#Разделение-данных-на-выборки-и-выделение-признаков" data-toc-modified-id="Разделение-данных-на-выборки-и-выделение-признаков-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Разделение данных на выборки и выделение признаков</a></span></li><li><span><a href="#Стандартизация-числовых-данных" data-toc-modified-id="Стандартизация-числовых-данных-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Стандартизация числовых данных</a></span></li><li><span><a href="#Подсчет-метрики-accuracy-для-различных-моделей" data-toc-modified-id="Подсчет-метрики-accuracy-для-различных-моделей-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Подсчет метрики accuracy для различных моделей</a></span><ul class="toc-item"><li><span><a href="#Дерево-решений" data-toc-modified-id="Дерево-решений-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Дерево решений</a></span></li><li><span><a href="#Случайный-лес" data-toc-modified-id="Случайный-лес-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Случайный лес</a></span></li><li><span><a href="#Логистическая-регрессия" data-toc-modified-id="Логистическая-регрессия-8.3"><span class="toc-item-num">8.3&nbsp;&nbsp;</span>Логистическая регрессия</a></span></li></ul></li><li><span><a href="#Составление-матрицы-ошибок-и-расчет-метрик" data-toc-modified-id="Составление-матрицы-ошибок-и-расчет-метрик-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Составление матрицы ошибок и расчет метрик</a></span><ul class="toc-item"><li><span><a href="#Матрица-ошибок-для-дерева-решений" data-toc-modified-id="Матрица-ошибок-для-дерева-решений-9.1"><span class="toc-item-num">9.1&nbsp;&nbsp;</span>Матрица ошибок для дерева решений</a></span></li><li><span><a href="#Матрица-ошибок-для-случайного-леса" data-toc-modified-id="Матрица-ошибок-для-случайного-леса-9.2"><span class="toc-item-num">9.2&nbsp;&nbsp;</span>Матрица ошибок для случайного леса</a></span></li><li><span><a href="#Матрица-ошибок-для-логистической-регрессии" data-toc-modified-id="Матрица-ошибок-для-логистической-регрессии-9.3"><span class="toc-item-num">9.3&nbsp;&nbsp;</span>Матрица ошибок для логистической регрессии</a></span></li></ul></li><li><span><a href="#Увеличение-выборки" data-toc-modified-id="Увеличение-выборки-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Увеличение выборки</a></span><ul class="toc-item"><li><span><a href="#Дерево-решений" data-toc-modified-id="Дерево-решений-10.1"><span class="toc-item-num">10.1&nbsp;&nbsp;</span>Дерево решений</a></span></li><li><span><a href="#Случайный-лес" data-toc-modified-id="Случайный-лес-10.2"><span class="toc-item-num">10.2&nbsp;&nbsp;</span>Случайный лес</a></span></li><li><span><a href="#Логистическая-регрессия" data-toc-modified-id="Логистическая-регрессия-10.3"><span class="toc-item-num">10.3&nbsp;&nbsp;</span>Логистическая регрессия</a></span></li></ul></li><li><span><a href="#Уменьшение-выборки" data-toc-modified-id="Уменьшение-выборки-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Уменьшение выборки</a></span><ul class="toc-item"><li><span><a href="#Дерево-решений" data-toc-modified-id="Дерево-решений-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Дерево решений</a></span></li><li><span><a href="#Случайный-лес" data-toc-modified-id="Случайный-лес-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>Случайный лес</a></span></li><li><span><a href="#Логистическая-регрессия" data-toc-modified-id="Логистическая-регрессия-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>Логистическая регрессия</a></span></li></ul></li><li><span><a href="#Улучшение-модели" data-toc-modified-id="Улучшение-модели-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Улучшение модели</a></span></li><li><span><a href="#-Комментарий-ревьюера" data-toc-modified-id="-Комментарий-ревьюера-13"><span class="toc-item-num">13&nbsp;&nbsp;</span> Комментарий ревьюера</a></span></li></ul></div>

# # Отток клиентов

# ## Цель и задача

# Из «Бета-Банка» стали уходить клиенты. Каждый месяц. Немного, но заметно. Банковские маркетологи посчитали: сохранять текущих клиентов дешевле, чем привлекать новых.
# 
# **Цель:** спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. 
# 
# **Задачи:**
# - Загрузить и подготовить данные, объясняя порядок действий.
# - Исследовать баланс классов, обучить модель без учёта дисбаланса. 
# - Улучшить качество модели, учитывая дисбаланс классов. Обучить разные модели и найти лучшую. 
# - Провесьти финальное тестирование.
# 
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# **Признаки:**
#     
#  - `RowNumber` — индекс строки в данных. 
#     
#  - `CustomerId` — уникальный идентификатор клиента
#     
#  - `Surname` — фамилия
#  - `CreditScore` — кредитный рейтинг
#  - `Geography` — страна проживания
#  - `Gender` — пол
#  - `Age` — возраст
#  - `Tenure` — количество недвижимости у клиента
#  - `Balance` — баланс на счёте
#  - `NumOfProducts` — количество продуктов банка, используемых клиентом
#  - `HasCrCard` — наличие кредитной карты
#  - `IsActiveMember` — активность клиента
#  - `EstimatedSalary` — предполагаемая зарплата 
#  
#  
# **Целевой признак**
# 
#  - `Exited` — факт ухода клиента

# # Подготовка данных

# ## Импорт необходимых библиотек

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
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
# Импортируем необходимые библиотеки
pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None # Чтобы не появлялось предупреждение SettingWithCopy при масштабировании


# <div class="alert alert-success">
# <h1>Комментарий ревьюера <a class="tocSkip"></a></h1>
# Молодец, что сгруппировал все импорты в одну ячейку, так удобнее и легче вносить изменения. Пользователь тетрадки  сразу поймет, что отсутствуют какие-то библиотеки
# </div>

# ## Получение первичной информации о данных

# In[2]:


df = pd.read_csv('/datasets/Churn.csv')
df.info()
df.shape
df.head()


# In[3]:


def nan_ratio(column):
    return print('Пропущено {:.1%}'.format(df[column].isna().value_counts()[1] / len(df)) + ' значений.')

nan_ratio('Tenure')


# **После получения общей информации видим, что в нашей таблице:**
#   - 10000 строк
#   - 14 столбцов с тремя типами данных. Тип данных integer имеет 8 столбцов, float - три, object - тоже 3.
# 
# 
# **Необходимо провести предобработку данных:**
#   
#   - В столбце Tenure пропущено 9.1% значений. Пропущенные значения могут означать то, что клиент не владеет недвижимостью. Поэтому удалять или заменять медианными значениями мы их не будем, а заменим нулями. 
#   - Столбец с фамилиями нам не нужен, каждый человек имеет собственный id, к тому же при дальнейших преобразованиях множество фамилий будет только мешать. Поэтому мы удалим его. 
#   - Необходимо проверить столбец customerid	на наличие дубликатов
#   - Также удалим столбец rownumber, он бесполезен, так как номер строки можно увидеть в индексе
#   - Приведем названия столбцов к нижнему регистру, чтобы было удобнее.
#  

# ## Предобработка данных 

# In[4]:


df.columns = df.columns.str.lower() 
#Привели к нижнему регистру заголовки.

list_for_strlower = ['geography', 'gender']
for column in list_for_strlower:
    df[column] = df[column].str.lower() 
# Циклом привели значения столбцов к нижнему регистру 


# <div class="alert alert-success">
# <h1>Комментарий ревьюера <a class="tocSkip"></a></h1>
# Элегантное решение
# </div>

# In[5]:


df['tenure'] = df['tenure'].fillna(0) 

df = df.drop('surname', axis=1) 
df = df.drop('rownumber', axis=1)


# In[6]:


print('Количество дубликатов в столбце customer id:', df['customerid'].duplicated().sum())


# In[7]:


df = df.drop('customerid', axis=1) 


# Так как дубликатов нет, можем удалить и столбец id, он нам не пригодится, будет приносить лишь шум для модели. Данные прошли предобработку и готовы к дальнейшим преобразованиям для обучения модели. 

# In[8]:


df.info()
df.head()


# Посмотрим графики распределения признаков 

# In[9]:


df['creditscore'].hist()
plt.title('Гистограмма частот для признака Creditscore')


# In[10]:


df['balance'].hist()
plt.title('Гистограмма частот для признака Balance')


# In[11]:


df['tenure'].hist()
plt.title('Гистограмма частот для признака Tenure')


# In[12]:


df['age'].hist()
plt.title('Гистограмма частот для признака Tenure')


# In[13]:


df['estimatedsalary'].hist()
plt.title('Гистограмма частот для признака Estimatedsalary')


# Построим график мультиколлинеарности для количественных признаков

# In[14]:


df.columns.to_list()


# In[15]:


df_for_corr = df.copy()
df_for_corr.drop(['geography','gender','tenure','hascrcard','isactivemember','exited'], axis=1, inplace=True)


# In[16]:


fig, ax = plt.subplots()
sns.heatmap(df_for_corr.corr(), vmin=-1, vmax=1, center=0, square=True, annot=True, fmt='.1f')
fig.set_figwidth(10)
fig.set_figheight(10)
plt.title('График мультиколлинеарности признаков датасета', fontsize=14)
plt.show()


# После построения корреляционной модели, можно сделать вывод, что нет сильно скоррелированных между собой признаков. Видим, что наблюдается небольшая корреляция между балансом и количеством используемых продуктов. Но коэффициент корреляции не настолько высок, чтобы нужно было удалять данные. Поэтому оставляем всё, как есть. Данные готовы!

# ## Преобразование данных для обучения модели

# Для дальнейшего обучения модели необходимо преобразовать данные к корректному виду, чтобы не получить ошибки при обучении. Для этого преобразуем категориальные признаки в численные с **помощью техники прямого кодирования.** А чтобы не попасть в ловушку фиктивных признаков применим параметр drop_first

# In[17]:


df_ohe = pd.get_dummies(df, drop_first=True) 
df_ohe


# ## Разделение данных на выборки и выделение признаков

# Теперь разобьём данные на три выборки (тренировочная, тестовая и валидационная) в отношении 3:1:1. Также после разделения выделим признаки и целевой признак. Целевой признак - это столбец Exited, это категориальный признак, поэтому в дальнейшем мы будем использовать **метод классификации**.  

# In[18]:


features = df_ohe.drop('exited', axis=1)
target = df_ohe['exited'] 


# In[19]:


features_train, features_1, target_train, target_1 = train_test_split(features, target, train_size=0.6, random_state=12345) 


# In[20]:


features_valid, features_test, target_valid, target_test = train_test_split(features_1, target_1, test_size=0.5, random_state=12345) 


# С качественными значениями разобрались, теперь перейдем к количественным. У всех числовых столбцов разный масштаб. Чтобы привести их к корректному виду, необходимо их стандартизировать с помощью StandardScaler. 

# ## Стандартизация числовых данных

# In[21]:


numeric = ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary']
scaler = StandardScaler()
scaler.fit(features_train[numeric]) 
features_train[numeric] = scaler.transform(features_train[numeric])

features_valid[numeric] = scaler.transform(features_valid[numeric]) 
features_test[numeric] = scaler.transform(features_test[numeric])

print(features_train.head())


# # Исследование задачи

# ## Подсчет метрики accuracy для различных моделей

# ### Дерево решений

# In[22]:


model_DTC = DecisionTreeClassifier(random_state=12345)
model_DTC.fit(features_train, target_train) 

predicted_valid_DTC = model_DTC.predict(features_valid)

accuracy_valid_DTC = accuracy_score(target_valid, predicted_valid_DTC)
print('Метрика accuracy дерева решений равна', accuracy_valid_DTC)


# ### Случайный лес

# In[23]:


model_RFC = RandomForestClassifier(random_state=12345, 
                                   n_estimators = 100)
model_RFC.fit(features_train, target_train) 

predicted_valid_RFC = model_RFC.predict(features_valid)

accuracy_valid_RFC = accuracy_score(target_valid, predicted_valid_RFC)
print('Метрика accuracy случайного леса равна', accuracy_valid_RFC)


# ### Логистическая регрессия

# In[24]:


model_LR = LogisticRegression(solver = 'liblinear')
model_LR.fit(features_train, target_train) 
predicted_valid_LR = model_LR.predict(features_valid)

accuracy_valid_LR = accuracy_score(target_valid, predicted_valid_LR)
print('Метрика accuracy логистической регрессии равна', accuracy_valid_LR)


# В зависимости от модели доля правильных ответов составляет 78-86%. Исследуем целевой признак.
# Чтобы оценить адекватность модели, проверим, как часто в целевом признаке встречается класс «1» или «0». Количество уникальных значений подсчитывается методом value_counts(). Он группирует строго одинаковые величины.

# In[25]:


target_train.value_counts(normalize=True)


# In[26]:


target_valid.value_counts(normalize=True)


# Наблюдается дисбаланс классов, полученные результаты далеки от соотношения 1:1. Так как доля ячеек со значение "0" равна 80%, мы можем предположить, что наша модtль будет чаще всего предсказывать только один вариант ответа (0). 

# In[27]:


target_pred_constant = pd.Series(0, index=target_valid.index)
print(accuracy_score(target_valid, target_pred_constant)) 


# Проверку на адекватность модель не прошла. Это связано с дисбалансом классов.

# ## Составление матрицы ошибок и расчет метрик

# При дисбалансе классов метрика аccuracy не подходит, необходима новая метрика. Спачала необходимо изучить матрицу ошибок.

# ### Матрица ошибок для дерева решений

# In[28]:


print(confusion_matrix(target_valid, predicted_valid_DTC))


# Видим, что модель довольно часто ошибается. К тому же, она склонна видеть отрицательные ответы там, где и нет. Необходимо просчитать метрики, чтобы лучше разобраться. Для этого напишем функцию.  

# In[29]:


def metrics(target, prediction):
    print('Полнота:', recall_score(target, prediction))
    print('Точность:', precision_score(target, prediction))
    print('F1 мера:', f1_score(target, prediction))
    print('ROC AUC мера:', roc_auc_score(target, prediction))


# In[30]:


metrics(target_valid, predicted_valid_DTC)


# ### Матрица ошибок для случайного леса

# In[31]:


print(confusion_matrix(target_valid, predicted_valid_RFC))


# In[32]:


metrics(target_valid, predicted_valid_RFC)


# ### Матрица ошибок для логистической регрессии 

# In[33]:


print(confusion_matrix(target_valid, predicted_valid_LR))


# In[34]:


probabilities_one_valid_LR = model_LR.predict_proba(features_valid)[:, 1]
print('ROC AUC мера:', roc_auc_score(target_valid, probabilities_one_valid_LR))


# Посмотрим как сильно наша модель будет отличаться от случайной, если сбалансировать классы. 
# По горизонтали нанесём долю ложноположительных ответов (FPR), а по вертикали — долю истинно положительных ответов (TPR). Переберём значения порога логистической регрессии и проведём ROC-кривую. 

# In[35]:


model_LR_balanced = LogisticRegression(random_state=12345, solver='liblinear', class_weight='balanced')
model_LR_balanced.fit(features_train, target_train) 

probabilities_valid_LR_balanced = model_LR_balanced.predict_proba(features_valid)
probabilities_one_valid_LR_balanced = probabilities_valid_LR_balanced[:, 1]
fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid_LR_balanced)  
plt.figure()
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая")
plt.show()
print(roc_auc_score(target_valid, probabilities_one_valid_LR_balanced))


# Проанализировав три модели, можно сделать **`следующие выводы`**:
# 1. В данных наблюдается явный дисбаланс классов, поэтому метрика accuracy нам совершенно не подходит. 
# 2. Все три модели довольно часто ошибаются, а также являются пессимистичными. Видим, что модели довольно часто ошибаются. К тому же, они склонны видеть отрицательные ответы там, где и нет (склоняются к ложноотрицательным ответам - ошибка второго рода).
# 3. В связи с дисбаласом классов полнота как у модели случайного леса, так и дерева решений составляет (округлив до десятых) 0,5. Значение не близко к единицы, следовательно, модель ищет положительные объекты не так хорошо, как могла бы. Поэтому будем перепроверять и улучшать модель в следующих щагах проекта. 
# 4. Точность модели дерева решений составляет всего 0,49, а случайного леса - 0,76. Однако, метрика F1 ни у одной из моделей не близка к 1, что означает низкое качество моделей. 
# 5. ROC AUC мера логистической регрессии лучше случайно модели после балансировки классов параметром class_weight='balanced', однако значение не близко к 1, качество низкое.
# 
# **`Следовательно, необходимо преобразовать данные, чтобы дисбаланс не учитывался, а модель обучалась и предсказывала корректно. Это мы сделаем следующим шагом.`**

# # Борьба с дисбалансом

# ## Увеличение выборки

# Для того, чтобы избавиться от дисбаланса увеличим выборку в 4 раза. Сначала скопируем несколько раз положительные объекты, затем создадим новую обучающую выборку и перемешаем выборку.

# In[36]:


def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat) 

    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345) 
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train, target_train, 4)

print(target_upsampled.value_counts(normalize=True))
print(target_upsampled.shape)


# Теперь проверим показатели качества на увеличинной выборке 

# ### Дерево решений

# In[37]:


model_DTC_upsampled = DecisionTreeClassifier(random_state=12345)
model_DTC_upsampled.fit(features_upsampled, target_upsampled)

predicted_valid_DTC_upsampled = model_DTC_upsampled.predict(features_valid) 
print('Меры DTC до увеличения выборки') 
print('')
metrics(target_valid, predicted_valid_DTC)
print('')
print('Меры DTC после увеличения выборки')
print('')
metrics(target_valid, predicted_valid_DTC_upsampled)


# Модель дерева решения показала незначительный рост только польноты, что не помогло улучшить модель.  

# ### Случайный лес

# In[38]:


model_RFC_upsampled = RandomForestClassifier(random_state=12345, 
                                   n_estimators = 100)
model_RFC_upsampled.fit(features_upsampled, target_upsampled) 

predicted_valid_RFC_upsampled = model_RFC_upsampled.predict(features_valid) 
print('Меры RFC до увеличения выборки') 
print('')
metrics(target_valid, predicted_valid_RFC)
print('')
print('Меры RFC после увеличения выборки')
print('')
metrics(target_valid, predicted_valid_RFC_upsampled)


# Случайный лес показывает результаты лучше, выросли все показатели, кроме точности.  

# ### Логистическая регрессия

# In[39]:


model_LR_upsampled = LogisticRegression(solver = 'liblinear')
model_LR_upsampled.fit(features_upsampled, target_upsampled) 

predicted_valid_LR_upsampled = model_LR_upsampled.predict(features_valid) 
print('Меры LR до увеличения выборки') 
print('')
metrics(target_valid, predicted_valid_LR)
print('')
print('Меры LR после увеличения выборки')
print('')
metrics(target_valid, predicted_valid_LR_upsampled)


# 
# 
# Полнота увеличилась больше, чем уменьшилась точность, поэтому f1 мера также возросла. Но показатели все же низкие. 
# **На данный момент случайный лес показал лучшие результаты**
# 
# Попробуем еще уменьшить выборку, чтобы добиться лучших значений f1 меры.

# ## Уменьшение выборки

# In[40]:


def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones])
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones])
 
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345) 
    
    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(features_train, target_train, 0.25)

print(target_downsampled.value_counts(normalize=True))
print(target_downsampled.shape)


# ### Дерево решений

# In[41]:


model_DTC_downsampled = DecisionTreeClassifier(random_state=12345)
model_DTC_downsampled.fit(features_downsampled, target_downsampled) # Обучили модель на тренировочной выборке

predicted_valid_DTC_downsampled = model_DTC_downsampled.predict(features_valid) # Получили прогноз на валидационной выборке
print('Меры DTC до увеличения выборки') 
print('')
metrics(target_valid, predicted_valid_DTC)
print('')
print('Меры DTC после увеличения выборки')
print('')
metrics(target_valid, predicted_valid_DTC_downsampled)


# Значительное уменьшение точности обеспечило низкий рост меры f1 при высоком росте полноты, поэтому нельзя назвать данную модель качественной. 

# ### Случайный лес

# In[42]:


model_RFC_downsampled = RandomForestClassifier(random_state=12345, 
                                   n_estimators = 100)
model_RFC_downsampled.fit(features_downsampled, target_downsampled) # Обучили модель на тренировочной выборке

predicted_valid_RFC_downsampled = model_RFC_downsampled.predict(features_valid) # Получили прогноз на валидационной выборке
print('Меры RFC до увеличения выборки') 
print('')
metrics(target_valid, predicted_valid_RFC)
print('')
print('Меры RFC после увеличения выборки')
print('')
metrics(target_valid, predicted_valid_RFC_downsampled)


# Все показатели, кроме точности возросли. Точность уменьшилась значительно. Можно отметить, что при увеличенной выборке f1 на валидационной выборке было выше на 0,1. 

# ### Логистическая регрессия

# In[43]:


model_LR_downsampled = LogisticRegression(solver = 'liblinear')
model_LR_downsampled.fit(features_downsampled, target_downsampled) # Обучили модель на тренировочной выборке

predicted_valid_LR_downsampled = model_LR_downsampled.predict(features_valid) # Получили прогноз на валидационной выборке
print('Меры LR до увеличения выборки') 
print('')
metrics(target_valid, predicted_valid_LR)
print('')
print('Меры LR после увеличения выборки')
print('')
metrics(target_valid, predicted_valid_LR_downsampled)


# <div class="alert alert-block alert-info">
# 
# **`ВЫВОД:`** как при уменьшении, так и при увеличении показатели на валидационной выборке всех моделей увеличиваются, за исключением точности. Однако, на увеличенной выборке значения лучше, чем на уменьшенной. Также можно выделить модель случайного леса с лучшими показателям. Мера `F1` на увеличенной валидационной выборке модели случайного леса составляет почти 0,6. 
# 
# **Поэтому далее мы будем улучшать модель случайного леса на увеличенной выборке.**
# 
# *Приступим к улучшению модели* 

# ## Улучшение модели

# In[44]:


gbp = RandomForestClassifier()


parametrs = {'n_estimators': range(80, 100, 5), 
             'max_depth': range(1, 11, 1), 
             'criterion': ['gini', 'entropy']}
            
grid = GridSearchCV(gbp, parametrs, cv=5)
grid.fit(features_upsampled, target_upsampled)
grid.best_params_


# Подобрали новые параметры, теперь проверим нашу модель на тестовой выборке. 

# # Тестирование модели

# In[45]:


model_RFC_final = RandomForestClassifier(random_state=12345, 
                                   n_estimators = 80, max_depth=10, criterion='gini')
model_RFC_final.fit(features_upsampled, target_upsampled)

predicted_valid_final = model_RFC_final.predict(features_valid) 
print('Валидационная выборка')
print('')
print('Меры RFC после увеличения выборки')
print('')
metrics(target_valid, predicted_valid_RFC_upsampled)
print('Меры RFC после увеличения выборки и подбора параметров')
print('')
metrics(target_valid, predicted_valid_final)


# Посмотрели, что все показатели на валидационной выборке увеличились после подбора лучших параметров, теперь проверим на тестовой выборке. 

# In[46]:


test_predictions_forest = model_RFC_final.predict(features_test) 
print('')
print("Тестовая выборка:") 
print('')
print(metrics(target_test, test_predictions_forest)) 


# F1 = 0.6, мы довели метрику до максимального значения и превысили целевое 0.59. 

# # Вывод 

# <div class="alert alert-block alert-info">
# 
# 
# **`В ходе проекта были выполнены следующие действия:`**
# 1. **Данные были обработаны:** 
#    - проведена провека на дубликаты, 
#    - заменены пропуски, 
#    - все значения приведены к нижнему регистру,
#    - удалены ненужные столбцы, которые могли помешать обучению модели.
# 2. **Выделены признаки и целевой принак, а также выборка была разделена на три части в отношении 3:1:1** 
# 3. **Признаки были обработаны для приведения данных в удобный формат для дальнейшего обучения модели**
#    - Категориальные признаки были приведены в численные методом прямого кодирования (OHE).
#    - Численные признаки были масштабированы путем стандартизации 
# 4. **Подсчитана метрика accuracy для каждой из модели и проверка моделей на адекватность что показало то, что:**
#    - Наблюдается диспропоруия в данных, а именно отношение нулей к единицам составляет 4:1
# 5. **Составлена матрица ошибок для наглядного представления результатов вычислений метрик точности и полноты, из чего следует, что:**
#    - Модели пессимистичны, они часто видят отрицательные ответы там, где их на самом деле нет.
# 6. **Подсчитаны метрики полноты, точности, f1 b auc-roc и выявлено, что**
#    - У всех моделей низкое качество и их необходимо улучшить
# 7. **Построена ROC кривая для логистической регрессии:**
#    - Показатель составил 0.76, он больше случайной модели, но не дотягивает до единицы, следовательно, модель не обладает высоким качеством 
# 8. **Для исправления дисбаланса выборка была увеличена и уменьшена в 4 раза:**
#    - Увеличенная выборка показала лучшие результаты метрик на всехх моделях 
#    - Самой качественной моделью оказалась `модель случайного леса`, поэтому мы продолжили работать с ней 
# 9. **С помощью RandomForestClassifier() подобрали оптимальные гиперпараметры и запустили модель с их использованием, что позволило увеличить качество модели**
# 10. **Произведена проверка на тестовой выборке и достигнуты необходимые значения меры F1** 
# 
# Нам удалось увеличить метрику f1 модели слуйчайного леса с 0.57 до 0.61 на валидационной выборке и достигнуть значения 0.6 на тестовой выборке. 
# Также мера auc-roc увелчилась с 0.71 до 0.77 на валидационной выборке, что является показателем увеличения качества. На тестовой выборке показатель составил 0.76.
# Полнота составила 0,66. Следовательно модель с вероятностью 66% предсказывает уход клиентов из банка.
# Однако, показатель точности не очень высокий, составляет всего 0,56, то есть только в 56% модель может верно предсказать уход клиентов. 
# 
# **Данная модель поможем маркетологам прогнозировать возможный уход клиентов из банка. Так у них будет возможность принять соответствующие меры для сохранения текущих клиентов.** 
#    
