import random
from neuron import Neuron

class SpikeNeuralNetwork:
    """Класс для представления нейросети из нейронов."""
    def __init__(self, num_neurons, input_size=0, output_size=0):

        # супер параметры
        self._DOPHAMIN = 0.35 #влияет на случайную активацию нейронов
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

    def initialize_network(self):
        """Простая и надежная инициализация"""
        # Создаем базовые связи между всеми нейронами с вероятностью
        for i in range(len(self.neurons)):
            for j in range(len(self.neurons)):
                if i != j and random.random() < 0.20:
                    weight = random.uniform(0.1, 0.8)
                    self.neurons[i].add_link(j, self._BASE_OLD, weight)

    def normalize_weights(self):
        """Нормализация весов для стабильности"""
        for neuron in self.neurons:
            total_abs_weight = sum(abs(link[2]) for link in neuron.links)
            if total_abs_weight > 1.0:
                for link in neuron.links:
                    link[2] /= total_abs_weight

    def safe_iteration(self, inputs=None):
        """Производит один шаг в нейросети, оперируя только состояниями нейронов."""

        for neuron in self.neurons:
            neuron.reLU()
            neuron.fire(save=True)

        if inputs is not None:
            for i, value in enumerate(inputs):
                if i < len(self.input_neurons):
                    self.input_neurons[i].value = value
                else:
                    return []

        for neuron in self.neurons:
            neuron.reLU()
            neuron.fire(save=True)

        return [neuron.state for neuron in self.output_neurons]
    
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
    
    def stdp_learning(self, target_outputs=None):
        """STDP обучение с учетом временных корреляций"""
        for neuron in self.neurons:
            if neuron.state:  # если нейрон активировался
                for link in neuron.links:
                    target_neuron = self.neurons[link[0]]
                    
                    # Если целевой нейрон активировался ПОСЛЕ - усиливаем связь
                    if target_neuron.state and not target_neuron.prev_state:
                        link[2] += self.learning_rate * 0.1
                    
                    # Если целевой нейрон активировался ДО - ослабляем связь  
                    elif target_neuron.prev_state and not target_neuron.state:
                        link[1] -= 1
                    
                    # Ограничиваем веса
                    link[2] = max(-1.0, min(1.0, link[2]))
                    
            # Обучение выходных нейронов на основе целевых значений
            if target_outputs and neuron in self.output_neurons:
                output_idx = self.output_neurons.index(neuron)
                target = target_outputs[output_idx]
                
                if neuron.state and target < 0.5:  # Ложное срабатывание
                    for link in neuron.links:
                        link[2] -= self.learning_rate * 0.2
                elif not neuron.state and target > 0.5:  # Пропуск активации
                    # Активируем нейрон и усиливаем входящие связи
                    neuron.value = neuron.stair + 0.1
                    for link in neuron.links:
                        link[2] += self.learning_rate * 0.1
        
    def generate_iteration(self, inputs, targets, iterations):
        """Специализированная итерация для обучения"""
        # Сброс состояний
        for neuron in self.neurons:
            neuron.value = 0.0
            neuron.state = False
            neuron.prev_state = False
        
        # Установка входных значений
        for i, value in enumerate(inputs):
            if i < len(self.input_neurons):
                self.input_neurons[i].value = value
        
        # Несколько шагов для распространения сигнала
        for _ in range(iterations):
            for neuron in self.neurons:
                neuron.reLU()

                if random.random() < self._DOPHAMIN:
                    neuron.state = True
                    neuron.activates += 1
                
                neuron.fire()
                neuron.hebbs_rule()
                neuron.activation()

        for neuron in self.neurons:
                neuron.tick()
        
        # Обучение
        self.stdp_learning(targets)
        self.normalize_weights()

    def get_loss(self, inputs, outputs, iterations):
        for _ in range(iterations):
            self.safe_iteration(inputs)

        loses = [output - result for output, result in zip(outputs, self.get_outputs())]
        return sum([abs(i) for i in loses]) / len(loses), loses
    
    def get_mid_weights(self):
        return sum(neuron.value for neuron in self.neurons) / len(self.neurons)
    
    def set_inputs(self, inputs):
        for i, value in enumerate(inputs):
            if i < len(self.input_neurons):
                self.input_neurons[i].value = value

    def back_propagation(self, inputs, outputs, iterations):

        try:
            for neuron in self.neurons:
                neuron.value = 0.0
                neuron.state = False
                neuron.prev_state = False

            for i in range(iterations):
                self.set_inputs(inputs)
                self.safe_iteration(outputs)

            for out_neuron, result in zip(self.output_neurons, outputs):
                useful = out_neuron.get_useful()
                error = result - (out_neuron.value + out_neuron.stair if out_neuron.state else 0)
                step = (error / len(useful)) * self._DOPHAMIN
                for conect_neuron in useful:
                    for link in self.neurons[conect_neuron].links:
                        if link[0] == out_neuron.id:
                            link[2] += step

            for neuron in self.neurons[:-self.output_size]:
                useful = neuron.get_useful()
                error = result - (neuron.value + neuron.stair if neuron.state else 0)
                step = (error / len(useful)) * self._DOPHAMIN * self.learning_rate
                for conect_neuron in useful:
                    for link in self.neurons[conect_neuron].links:
                        if link[0] == out_neuron.id:
                            link[2] += step
                            
        except RecursionError:
            return

    def training(self, iterations: int, data: list, validation: float=0.05, batch: int = 10):
        """
        Тренируем нейронную сеть.
        Args:
            iterations: колличество итераций,
            data: лист с входными данными формата [ [[input], [output]], [[1,0], [1]] ],
            valitation: какая часть датасета будет использованна для валидации в процентах, по умолчанию 0.05,
            batch: размер батча
        """
        training_data = data[:len(data) - round(len(data) * validation)]
        validation_data = data[len(data) - round(len(data) * validation):]

        patience_counter = 0
        mid_loss = 1
        loss = 1

        train_history = []
        
        for i in range(iterations):
            # Обучаем на случайном примере
            inputs, outputs = random.choice(training_data)
            self.generate_iteration(inputs, outputs, 5)
            self.back_propagation(inputs, outputs, 1)

            # Оцениваем на валидации
            if i % batch == 0:
                loss = 0
                for val_input, val_output in validation_data:
                    loss += self.get_loss(val_input, val_output, 5)[0]

                loss /= len(validation_data)

                mid_weights = self.get_mid_weights()
                train_history.append((loss, mid_weights, mid_loss, self._DOPHAMIN))

                mid_loss = sum(data[0] for data in train_history) / len(train_history)
                
                # Динамическая регулировка коэффициента обучения
                if loss < mid_loss:
                    patience_counter = 0
                    self._DOPHAMIN = max(self._DOPHAMIN * (1 - self.learning_rate), self.min_dofamine)

                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        self._DOPHAMIN = min(self._DOPHAMIN * (1 + self.learning_rate * self.patience), self.max_dofamine)
                        patience_counter = 0

                if mid_weights > 0.5:
                    self._DOPHAMIN = max(self._DOPHAMIN * (1 - self.learning_rate), self.min_dofamine)

                elif mid_weights < -0.1:
                    self._DOPHAMIN = min(self._DOPHAMIN * (1 + self.learning_rate), self.max_dofamine)

        return train_history