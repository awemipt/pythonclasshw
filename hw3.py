class CountVectorizer:
    """
    Класс, предоставляющий базовую функциональность
    для подсчета встречаемости слов в списке строк.
    """

    def __init__(self) -> None:
        """
        Конструктор класса.

        :return: Не возвращает значения.
        """
        self.count_matrix: list[list[int]] = []  # Матрица счетчиков
        self.feature_names: list[str] = []  # Уникальные слова
        self.name_indices: dict[str, int] = {}  # Индексы слов в матрице

    @staticmethod
    def remove_duplicates(input_list):
        """
        Удаление дубликатов из списка, сохраняя порядок.

        :param input_list: Входной список для обработки.
        :return: Новый список с удаленными дубликатами.
        """
        output_list = []
        for item in input_list:
            if item not in output_list:
                output_list.append(item)
        return output_list

    def fit_transform(self, data: list[str]) -> list[list[int]]:
        """
        Формирование векторизатора на входных данных
        и преобразование данных в матрицу счетчиков.

        :param data: Список строк для обработки.
        :return: Матрица счетчиков, представляющая встречаемость слов в данных.
        """
        self.feature_names = [word.lower() for sentence in data
                             for word in sentence.split()]
        self.feature_names = self.remove_duplicates(self.feature_names)
        self.name_indices = {word: i for i, word in enumerate(self.feature_names)}
        self.count_matrix = [[0 for _ in self.feature_names] for _ in data]
        for i, sentence in enumerate(data):
            for word in sentence.split():
                self.count_matrix[i][self.name_indices[word.lower()]] += 1
        return self.count_matrix

    def get_feature_names(self):
        """
        Получение списка уникальных имен признаков, полученных из обученных данных.

        :return: Список уникальных имен признаков.
        """
        return self.feature_names


corpus = ['Crock Pot Pasta Never boil pasta again',
          'Pasta Pomodoro Fresh ingredients Parmesan to taste']

vecotrizer = CountVectorizer()
count_matrix = vecotrizer.fit_transform(corpus)
print(vecotrizer.get_feature_names())
print(count_matrix)
