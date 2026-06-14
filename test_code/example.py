def hello_world():
    """Выводит приветственное сообщение Hello World"""
    print('Hello World from Python!')


def calculate_sum(a, b):
    """Возвращает сумму двух чисел"""
    return a + b


def calculate_difference(a, b):
    """Возвращает разность двух чисел (a - b)"""
    return a - b


def calculate_product(a, b):
    """Возвращает произведение двух чисел"""
    return a * b


def calculate_division(a, b):
    """Возвращает частное от деления a на b. При b=0 возвращает None"""
    if b == 0:
        return None
    return a / b


def is_even(number):
    """Проверяет, является ли число чётным. Возвращает True или False"""
    return number % 2 == 0


def is_prime(n):
    """Проверяет, является ли число простым. Работает для n > 1"""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


def reverse_string(s):
    """Возвращает строку в обратном порядке"""
    return s[::-1]


def get_string_length(s):
    """Возвращает длину строки"""
    return len(s)


def to_uppercase(s):
    """Преобразует строку в верхний регистр"""
    return s.upper()


def to_lowercase(s):
    """Преобразует строку в нижний регистр"""
    return s.lower()


class User:
    """Класс, представляющий пользователя системы"""

    def __init__(self, name):
        """Инициализация пользователя с именем"""
        self.name = name

    def greet(self):
        """Возвращает приветствие для пользователя"""
        return f'Hello, {self.name}!'

    def get_name(self):
        """Возвращает имя пользователя"""
        return self.name

    def set_name(self, new_name):
        """Устанавливает новое имя пользователя"""
        self.name = new_name
        return self.name


class Calculator:
    """Класс-калькулятор с базовыми операциями (дублирующий calculator.py)"""

    def add(self, a, b):
        """Сложение двух чисел"""
        return a + b

    def subtract(self, a, b):
        """Вычитание двух чисел"""
        return a - b

    def multiply(self, a, b):
        """Умножение двух чисел"""
        return a * b

    def divide(self, a, b):
        """Деление двух чисел. При b=0 возвращает None"""
        if b == 0:
            return None
        return a / b


def main():
    """Главная функция для демонстрации работы калькулятора"""
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"5 - 3 = {calc.subtract(5, 3)}")
    print(f"5 * 3 = {calc.multiply(5, 3)}")
    print(f"10 / 2 = {calc.divide(10, 2)}")
