# імпортуємо бібліотеку pytorch
import torch

# імпортуємо модуль для створення нейронних мереж
import torch.nn as nn


# створюємо власний клас нейронної мережі
class SimpleNN(nn.Module):
    def __init__(self):
        # викликаємо конструктор базового класу nn.Module
        super(SimpleNN, self).__init__()

        # визначаємо один повнозв’язний лінійний шар
        # 1 вхідний нейрон і 1 вихідний нейрон
        self.fc = nn.Linear(1, 1)

    # метод forward описує пряме поширення сигналу
    def forward(self, x):
        # передаємо вхідні дані через лінійний шар
        return self.fc(x)


# створюємо екземпляр моделі
model = SimpleNN()

# виводимо архітектуру моделі
print(model)
