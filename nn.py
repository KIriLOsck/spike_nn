from neuron import Neuron

class SpikeNeuralNetwork:
    """Класс для представления нейросети из нейронов."""
    def __init__(self, num_neurons, input_size=0, output_size=0):

        # супер параметры
        self._DOPHAMIN = 1 #влияет на запоминание
        self._ACTIVATION_RADIUS = 10 # предел расстояния для связи нейронов
        self._ACTIVATION_DEVISOR = 5 # коофицент вероятности активации нейрона
        self._ACTIVATION_MULTIPLIER = 0.01 # влияет на шанс создания связи
        self._BASE_OLD = 5 # возраст новой связи по умолчанию
        self._REFRACT_PERIOD = 0

        # параметры обучения
        self.patience = 20  # Количество итераций без улучшения перед уменьшением lr
        self.learning_rate = 0.01
        self.min_dofamine = 0.05  # Минимальное значение коэффициента обучения
        self.max_dofamine = 0.8    # Максимальное значение коэффициента обучения

        self.neurons = []
        self.num_neurons = num_neurons

        for _ in range(num_neurons):
            self.add_neuron()

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
        output = "|  Индекс  | Значение  | Среднее связей\n"
        for neuron in self.neurons:
            output += "| " + str(neuron.id) + (9 - len(str(neuron.id)))*" "
            output += "| " + str(round(float(neuron.value), 2))
            output += (10 - len(str(round(neuron.value + 0.1, 2))))*" "
            output += "| " + str(sum(link[2] for link in neuron.links) / (len(neuron.links) or 1)) + "\n"
        return output
    
    def modify_dophamin(self, value):
        self._DOPHAMIN += value

    def get_outputs(self):
        return [neuron.value for neuron in self.output_neurons]

    def add_neuron(self):
        self.neurons.append(Neuron(self))

    def normalize_weights(self):
        """Нормализация весов для стабильности"""
        for neuron in self.neurons:
            total_abs_weight = sum(abs(link[2]) for link in neuron.links)
            if total_abs_weight > 1.0:
                for link in neuron.links:
                    link[2] /= total_abs_weight
    
    def iteration(self, inputs=None):
        """Производит один шаг в нейросети, обновляя состояние нейронов и веса."""

        for neuron in self.neurons:
            neuron.tick()

        if inputs is not None:
            for i, value in enumerate(inputs):
                if i < len(self.input_neurons):
                    self.input_neurons[i].value = value
                else:
                    raise RuntimeError("Value not associated with any input neuron.")

        for neuron in self.neurons:
            neuron.activation()
            neuron.hebbs_rule()
            neuron.reLU()
            neuron.fire()

        return [neuron.state for neuron in self.output_neurons]
    
    def get_mid_weights(self):
        return sum(neuron.value for neuron in self.neurons) / len(self.neurons)
    
    def set_inputs(self, inputs):
        for i, value in enumerate(inputs):
            if i < len(self.input_neurons):
                self.input_neurons[i].value = value