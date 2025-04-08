from pydatalog import pyDatalog

# Создание терминов (переменных)
pyDatalog.create_terms('mother, father, parents, X, Y')

# Определение фактов
+ mother('Alice', 'Bob')
+ mother('Alice', 'Charlie')
+ father('David', 'Bob')
+ father('David', 'Charlie')

# Определение правила для поиска родителей
parents(X, Y) <= mother(X, Y) | father(X, Y)

# Применение правила
print(parents('Alice', X))  # Все дети Алисы
print(parents(X, 'Bob'))    # Все родители Боба

