# Все неоходимое для этой лабораторной
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Задание 1. Есть два файла с данными турагенства: email.csv и username.csv. C ними нужно проделать все манипуляции, указанные в лекции 2, а именно:
# a) Группировка и агрегирование (сгруппировать набор данных по значениям в столбце, а затем вычислить среднее значение для каждой группы)
# b) Обработка отсутствующих данных (заполнение отсутствующих значений определенным значением или интерполяция отсутствующих значений)
# c) Слияние и объединение данных (соединить два DataFrames в определенном столбце)

mail_df = pd.read_csv('email.csv')
uname_df = pd.read_csv('username.csv')

grouped_data = mail_df.groupby('Trip count') # группировка с помощью groupby на основе столбца 'Trip count'
mean_values = grouped_data['Total price'].mean() # вычисляем среднее значение 'Total price' для каждой группы, полученной на предыдущем шаге
print(mean_values)

mail_df = mail_df.fillna(value=0) # обработка отсутствующих данных (замена NaN на 0)
print(mail_df) 

uname_df = uname_df.fillna(value=0)
print(uname_df)

# слияние и объединение данных
merged_info = pd.merge(uname_df, mail_df, on=['Identifier','First name','Last name','Trip count','Phone number'])
merged_info = merged_info.to_string()
print(merged_info)

# Задание 2. Преобразование данных (pivot):
# a) Нужно создать сводную таблицу так, чтобы в index были столбцы “Rep”, “Manager” и “Product”, а в values “Price” и “Quantity”. Также нужно использовать функцию aggfunc=[numpy.sum] и заполнить отсутствующие значения нулями. В итоге можно будет увидеть количество проданных продуктов и их стоимость, отсортированные по имени менеджеров и директоров.

sales_df = pd.read_csv('sales.csv')
# функция pivot_table() нужна чтобы создать таблицу, в которой данные из 1 или нескольких столбцов ДФ будут использ-ся как индексы таблицы, а данные из др. столбцов будут использ-ся как значения таблицы
# для вычисления значений таблицы используем функцию "sum"
# для заполнения пустых ячеек таблицы используем 0
table = pd.pivot_table(sales_df, index=['Rep', 'Manager', 'Product'], values=['Price', 'Quantity'], aggfunc="sum", fill_value=0)
table = table.sort_index() # сортируем индексы таблицы
print(table)

# b) Учебный файл (data.csv) + практика Dataframe.pivot. Поворот фрейма данных и суммирование повторяющихся значений.

data_df = pd.read_csv('data.csv')
table = pd.pivot_table(data_df, index='Date', columns='Product', values='Sales', aggfunc="sum", fill_value=0)
print(table)

# Задание 3. Визуализация данных (можно использовать любой из учебных csv-файлов).
# a) Необходимо создать простой линейный график из файла csv (два любых столбца, в которых есть зависимость)

cars_df = pd.read_csv('cars.csv').head(40)

cars_df.plot(kind = 'line', x = 'Horsepower', y ='MPG', color='b') # создаем линейный график синего цвета
plt.xlabel('Мощность двигателя') # название оси Х
plt.ylabel('Расход топлива') # название оси Y
plt.title('Зависимость расхода топлива от мощности двигателя') # название графика
plt.show()

# b) Создание визуализации распределения набора данных. Создать произвольный датасет через np.random.normal или использовать датасет из csv-файлов, потом построить гистограмму.

cars_df = pd.read_csv('cars.csv').head(80)

plt.hist(cars_df['Horsepower']) # строим гистограмму
plt.xlabel('Мощность двигателя')
plt.ylabel('Частотность')
plt.title('Частотность встречаемости автомобилей по мощности двигателя')
plt.show()

# c) Сравнение нескольких наборов данных на одном графике. Создать два набора данных с нормальным распределением или использовать данные из файлов. Оба датасета отразить на одной оси, добавить легенду и название.

cars_df = pd.read_csv('cars.csv').head(40)

plt.plot(cars_df['Weight'], cars_df['Acceleration'], 'g') # создаем график функции
plt.plot(cars_df['Horsepower'], cars_df['Acceleration'], 'y')
plt.xlabel(' Вес автомобиля / Мощность двигателя')
plt.ylabel('Ускорение автомобиля')
plt.title('Зависимость ускорения автомобиля от его веса и мощности двигателя')
plt.legend(['Weight / Acceleration','Horsepower / Acceleration']) # добавляем легенду
plt.show()

# d) Построение математической функции. Создать данные для x и y (где x это numpy.linspace, а y - заданная в условии варианта математическая функция). Добавить легенду и название графика.
# i. Вариант 1 - функция sin

x = np.linspace(0, 2*np.pi, 100) # создаем x (100 значений от 0 до 2*π с шагом 1/100 части интервала)
y = np.sin(x) # создаем y = синусу каждого значения x
plt.plot(x, y)
plt.legend(['sin(x)']) 
plt.title('График sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# ii. Вариант 2 - функция cos

x = np.linspace(0, 2*np.pi, 100)
y = np.cos(x)
plt.plot(x, y)
plt.legend(['cos(x)'])
plt.title('График cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# e) Моделирование простой анимации. Создать данные для x и y (где x это numpy.linspace, а y - математическая функция). Запустить объект line, ввести функцию update_graphs(i) c методом line.set_ydata() и создать анимированный объект FuncAnimation.
# a) Шаг 1: смоделировать график sin(x) (или cos(x)) в движении.
# b) Шаг 2: добавить к нему график cos(x) (или sin(x)) так, чтобы движение шло одновременно и оба графика отображались на одной оси.

x = np.linspace(0, 2 * np.pi, 100)
y_sinus = np.sin(x) # y_sinus как синус каждого значения x
y_cosinus = np.cos(x) # как косинус каждого значения x
fig, ax = plt.subplots()

sinus_line, = ax.plot(x, y_sinus, color='r') # график y_sinus от x
cosinus_line, = ax.plot(x, y_cosinus, color='g') # график y_cosinus от x
plt.legend(['sin(x)','cos(x)'])

def update_graphs(i):
    sinus_line.set_ydata(np.sin(x + i / 10)) # обновляем значения y для графика sin(x) на каждом шаге анимации
    cosinus_line.set_ydata(np.cos(x + i / 10)) # обновляем значения y для графика cos(x) на каждом шаге анимации
    return sinus_line, cosinus_line # возвращаем обновленные графики sin(x) и cos(x)

animation_res = FuncAnimation(fig, update_graphs, frames=100, interval=50, blit=True)
plt.show()


# Задание 4. Загрузка CSV-файла в DataFrame. Используя pandas, напишите скрипт, который загружает CSV-файл в DataFrame и отображает первые 5 строк df.

df = pd.read_csv('cars.csv')
df = df.head(5)
print(df)

# Задание 5. Выбор столбцов из DataFrame.
# a) Используя pandas, напишите сценарий, который из DataFrame файла sales.csv выбирает только те строки, в которых Status = presented, и сортирует их по цене от меньшего к большему.

sales_df = pd.read_csv('sales.csv')
presented = sales_df[sales_df['Status'] == 'presented']
result = presented.sort_values('Price')
print(result)

# b) Из файла climate.csv отображает в виде двух столбцов названия и коды (rw_country_code) тех стран, у которых cri_score больше 100, а fatalities_total не более 10.

climate_df = pd.read_csv('climate.csv')
clean_df = climate_df[(climate_df['cri_score'] > 100) & (climate_df['fatalities_total'] <= 10)]
result = clean_df[['rw_country_name', 'rw_country_code']]
print(result)

# c) Из файла cars.csv отображает названия 50 первых американских машин, у которых расход топлива MPG не менее 25, а частное внутреннего объема (Displacement) и количества цилиндров не более 40. Названия машин нужно вывести в алфавитном порядке.

cars_df = pd.read_csv('cars.csv')
clean_df = cars_df[(cars_df['MPG'] >= 25) & (cars_df['Displacement'] / cars_df['Cylinders'] <= 40)]
result_cars = clean_df['Car'].sort_values().head(50)
for car in result_cars:
    print(car)

# Задание 6. Вычисление статистики для массива numpy
# Используя numpy, напишите скрипт, который загружает файл CSV в массив numpy и вычисляет среднее значение, стандартное отклонение и максимальное значение массива. Для тренировки используйте файл data.csv, а потом любой другой csv-файл от 20 строк.

sales_data = np.genfromtxt('data.csv', delimiter=',', names=True)
# данные из файла 'data.csv'
# разделитель ','
# 'names=True' -> 1ая строка файла содержит названия столбцов.

mean_sales_data = np.mean(sales_data['Sales']) # вычисляем среднее значение столбца 'Sales'
std_sales_data = np.std(sales_data['Sales']) # вычисляем стандартное отклонение столбца 'Sales'
max_sales_data = np.max(sales_data['Sales']) #  вычисляем максимальное значение столбца 'Sales'

print(f"Среднее значение в дф data.csv: {mean_sales_data}")
print(f"Стандартное отклонение в дф data.csv: {std_sales_data}")
print(f"Максимальное значение в дф data.csv: {max_sales_data}")

price_data = np.genfromtxt('sales.csv', delimiter=',', names=True)

mean_price = np.mean(price_data['Price']) # среднее значение столбца 'Price'
std_price = np.std(price_data['Price']) # стандартное отклонение 'Price'
max_price = np.max(price_data['Price']) # максимальное значение 'Price'

print(f"\nСреднее значение в дф sales.csv: {mean_price}")
print(f"Стандартное отклонение в дф sales.csv: {std_price}")
print(f"Максимальное значение в дф sales.csv: {max_price}")


# Задание 7.
# Операции с матрицами: Используя numpy, напишите сценарий, который создает матрицу и выполняет основные математические операции, такие как сложение, вычитание, умножение и транспонирование матрицы.

matrix1 = np.array([[1, 2, 3], # создаем матрицу №1 из 3 столбцов и 3 строк
                     [4, 5, 6],
                     [7, 8, 9]])
print(f"Матрица №1: {matrix1}")

matrix2 = np.array([[10, 11, 12], # создаем матрицу №2 из 3 столбцов и 3 строк
                     [13, 14, 15],
                     [16, 17, 18]])
print(f"\nМатрица №2: {matrix2}")

summa = matrix1 + matrix2 # сложение матриц
print(f"\nСумма матриц №1 и №2: {summa}")

difference = matrix1 - matrix2 # вычитание матриц
print(f"\nРазность матриц №1 и №2: {difference}")

multiplication = np.dot(matrix1, matrix2) # умножение матриц
print(f"\nПроизведение матриц №1 и №2: {multiplication}")

matrix1_transpose = np.transpose(matrix1) # транспонирование матрицы №1
print(f"\nТранспонированная матрица №1: {matrix1_transpose}")

matrix2_transpose = np.transpose(matrix2) # транспонирование матрицы №2
print(f"\nТранспонированная матрица №2: {matrix2_transpose}")