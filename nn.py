from neuron import Neuron

class SpikeNeuralNetwork:
    """Класс для представления нейросети из нейронов."""
    def __init__(self, num_neurons: int, input_size: int = 0, output_size: int = 0):

        # нейромедиаторы
        self.DOPHAMIN = 1.0 #влияет на обучение, стремится к одному
        self.SEROTONIN = 1.0 #влияет на концентрацию

        # супер параметры
        self._ACTIVATION_RADIUS = 10 # предел расстояния для связи нейронов
        self._ACTIVATION_DEVISOR = 5 # обратно пропорционален вероятности активации нейрона
        self._BASE_STRENGTH = 0.01 # вес новой связи по умолчанию
        self._BASE_OLD = 1 # возраст новой связи по умолчанию
        self._REFRACT_PERIOD = 0 # рефрактный период в итерациях. 0 отключает рассчёт

        # параметры обучения
        self.learning_rate = 0.01 # как быстро дофамин будет приравниваться к 1

        self.neurons = []
        self.num_neurons = num_neurons

        for _ in range(num_neurons):
            self._add_neuron()

        self.input_size = input_size
        self.output_size = output_size
        self.input_neurons = self.neurons[:input_size]
        self.output_neurons = self.neurons[-output_size:]

        self.active_neurons = []


    def __getitem__(self, index):
        return self.neurons[index]
    

    def __iter__(self):
        return iter(self.neurons)


    def __len__(self):
        return len(self.neurons)
    

    def __str__(self):
        output = "|  Индекс  | Значение  |    Среднее связей    | Средний возраст\n"
        for neuron in self.neurons:
            mid_sum = sum(link[2] for link in neuron.links) / (len(neuron.links) or 1) or "Нет"
            mid_old = round(sum(link[1] for link in neuron.links) / (len(neuron.links) or 1), 3) or "Нет"

            output += "| " + str(neuron.id) + (9 - len(str(neuron.id)))*" "
            output += "| " + str(round(float(neuron.value), 2))
            output += (10 - len(str(round(neuron.value + 0.1, 2))))*" "
            output += "| " + str(mid_sum) + (21 - len(str(mid_sum)))*" "
            output += "| " + str(mid_old) + "\n"
        return output


    def _add_neuron(self) -> None:
        """Добавляет нейрон в массив."""
        self.neurons.append(Neuron(self))



    def get_outputs(self, reLU: bool = False) -> list:
        """Возвращает сырые данные из нейросети или с применённой активацией в зависимости от параметра."""
        if reLU:
            return [1 if neuron.value > neuron.stair else 0 for neuron in self.output_neurons]
        else:
            return [neuron.value for neuron in self.output_neurons]


    def normalize_weights(self) -> None:
        """Нормализация весов для стабильности. Если сумма весов по модулю больше 1, делит каждый вес на эту сумму."""
        for neuron in self.neurons:
            total_abs_weight = sum(abs(link[2]) for link in neuron.links)
            if total_abs_weight > 1.0:
                for link in neuron.links:
                    link[2] /= total_abs_weight
    

    def iteration(self, iteration: int, inputs: list = None, reLU: bool = False) -> list:
        """
        Производит один шаг в нейросети с поддержкой нейропластичности и возвращает результат работы.
        Удалает старые связи; Подставляет входные значения;
        Создаёт связи; Расчитывает порог; Проводит спайки.
        """

        for neuron in self.neurons:
            neuron.destroy()

        if inputs is not None:
            self.set_inputs(inputs)

        for neuron in self.neurons:
            neuron.random_inhibitory()

        for neuron in self.neurons:
            neuron.hebbs_rule()

        for neuron in self.neurons:
            neuron.reLU(iteration)

        for neuron in self.neurons:
            neuron.fire()

        self.normalize_weights()
        self.DOPHAMIN += (1 - self.DOPHAMIN) * self.learning_rate

        return self.get_outputs(reLU=reLU)
    

    def get_mid_values(self) -> float:
        """Возвращает среднее значение нейронов. Для простой оценки активности нейросети."""
        return sum(neuron.value for neuron in self.neurons) / len(self.neurons)
    

    def set_inputs(self, inputs: list) -> None:
        """Применяет массив входных значений, к потенциалу входных нейронов."""
        for i, value in enumerate(inputs):
                if i < len(self.input_neurons):
                    self.input_neurons[i].value = value
                else:
                    raise RuntimeError("Value not associated with any input neuron.")