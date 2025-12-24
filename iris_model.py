# імпортуємо датасет iris з sklearn
from sklearn.datasets import load_iris

# функція для поділу даних на тренувальні та тестові
from sklearn.model_selection import train_test_split

# імпортуємо модель послідовної нейронної мережі
from tensorflow.keras.models import Sequential

# імпортуємо повнозв’язний шар
from tensorflow.keras.layers import Dense

# завантажуємо датасет iris
data = load_iris()

# x — ознаки квітки (довжина та ширина чашолистка і пелюстки)
X = data.data

# y — клас квітки (0, 1 або 2)
y = data.target

# розділяємо дані на навчальну та тестову вибірки
# 80% — для навчання, 20% — для тестування
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# створюємо послідовну нейронну мережу
model = Sequential()

# прихований шар з 10 нейронами та relu-активацією
# input_shape=(4,) — тому що маємо 4 вхідні ознаки
model.add(Dense(10, input_shape=(4,), activation='relu'))

# вихідний шар з 3 нейронами (3 класи iris)
# softmax використовується для багатокласової класифікації
model.add(Dense(3, activation='softmax'))

# компіляція моделі
# adam — оптимізатор
# sparse_categorical_crossentropy — функція втрат для цілочисельних міток класів
# accuracy — метрика точності
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# навчання моделі
# epochs=50 — кількість проходів по всій вибірці
# batch_size=5 — розмір батчу
# validation_split=0.1 — 10% тренувальних даних для валідації
model.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.1)

# оцінювання моделі на тестових даних
loss, acc = model.evaluate(X_test, y_test)

# виводимо точність класифікації
print(f"Точність на тестових даних: {acc * 100:.2f}%")
