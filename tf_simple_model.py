# імпортуємо клас для створення послідовної моделі
from tensorflow.keras.models import Sequential

# імпортуємо повнозв’язний (dense) шар
from tensorflow.keras.layers import Dense

# створюємо послідовну нейронну мережу
model = Sequential()

# додаємо один повнозв’язний шар
# 1 нейрон на виході
# input_shape=(1,) означає, що на вхід подається одне число
model.add(Dense(1, input_shape=(1,)))

# виводимо структуру моделі
# показує кількість шарів, параметрів та форму даних
model.summary()
