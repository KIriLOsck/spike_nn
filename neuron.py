import random

class Neuron:
    """Класс для представления нейрона в нейросети."""
    def __init__(self, neurons_list, activates_history: int = 5):
        self.state = False
        self.links = []  # [индекс, возраст, вес]
        self.stair = 0.5    # Порог активации
        self.value = 0.0    # Мембранный потенциал
        self.activates = 0
        self.last_activates = []

        self.nn = neurons_list
        self.id = len(neurons_list)

        for _ in range(activates_history):
            self.last_activates.append(False)


    def __str__(self):
        out = ""
        for i in self.links:
            out += str(i) + '\n'
        out += f"Value: {self.value}, State: {self.state}"
        return out
    

    def need_correction(self):
        if random.random() < self.activates / self.nn._ACTIVATION_DEVISOR:
            self.activates -= self.nn._ACTIVATION_DEVISOR
            return True
        else:
            return False
        
    def is_in_refract(self):
        if self.nn._REFRACT_PERIOD > len(self.last_activates):
            raise RuntimeError("Impossible to determine the refract period: the activation history too short")
        
        if True in self.last_activates[-self.nn._REFRACT_PERIOD:]:
            return True
        else:
            return False
        
    def next_state(self, state: bool):
        self.last_activates.pop(0)
        self.last_activates.append(state)

    def add_link(self, index, old, weight):
        """
        Добавляет связь между нейронами.
        Если связь уже существует, обновляет вес и возраст.
        :param index: Индекс целевого нейрона
        :param old: Возраст связи
        :param weight: Вес связи
        :return: None
        """

        if index == self.id:
            return
        if self.nn[index] in self.nn.input_neurons:
            return
        if self.nn[self.id] in self.nn.output_neurons:
            return

        for i in self.links:
            if i[0] == index:
                i[2] = (weight + i[2]) / 2
                i[1] += self.nn._BASE_OLD
                return

        self.links.append([index, old, weight])


    def get_radius(self):
        if self.id < len(self.nn):
            if self.nn._ACTIVATION_RADIUS >= len(self.nn):
                return [x.id for x in self.nn if x != self.id]
            else:
                in_rad = []
                for i in range(self.id - self.nn._ACTIVATION_RADIUS, self.id + self.nn._ACTIVATION_RADIUS + 1):
                    if i != self.id:
                        if i > len(self.nn) - 1:
                            i -= len(self.nn)
                        in_rad.append(self.nn[i].id)
            return list(set(in_rad))
        else:
            raise RuntimeError("Index out of range!")


    def get_neuron(self):
        """Получает индекс случайного нейрона в пределах радиуса активации."""
        varible_neurons = self.get_radius()
        return random.choice(varible_neurons)
    

    def activation(self):
        """Проверяет, не перевозбуждается ли нейрон, и создает новые ингибирующие связи если да."""
        if not False in self.last_activates and self.need_correction():
            
            index = self.get_neuron()
            if index is None:
                return
           
            self.nn[index].add_link(self.id, self.nn._BASE_OLD, random.uniform(-0.1,0))


    def get_activation_chain(self, n1, n2):
        """Узнаёт активировались ли нейроны последовательно и возвращает порядок активации или None"""
        if n1.last_activates[:-1] == n2.last_activates[1:]: # [1 0 0 1 | 0] ... [0 | 1 0 0 1]
            return (n2 , n1) # n1 зависит от n2
        elif n2.last_activates[:-1] == n1.last_activates[1:]:
            return (n1 , n2)
        else:
            return None


    def hebbs_rule(self):
        """Правило Хебба: нейроны активирующиеся вместе, связываются."""
        connected = self.get_radius()

        for index in connected:

            if self.need_correction():
                chain = self.get_activation_chain(self, self.nn[index])
                if chain is None:
                    continue

                for n in range(len(chain) - 1): # от последнего нейрона связь не создаём
                    chain[n].add_link(chain[n + 1].id, self.nn._BASE_OLD, self.nn._BASE_STRENGTH)


    def destroy(self):
        """Обновляет состояние нейрона, уменьшая возраст связей и удаляя старые связи."""
        rem = []
        age_step = 2.0 - self.nn.DOPHAMIN
        age_step = max(0.1, min(3.0, age_step))  # защита от экстремумов
        for i in self.links:
            i[1] -= age_step
            if i[1] < 0:
                rem.append(i)
        for i in rem:
            self.links.pop(self.links.index(i))


    def reLU(self):
        """Функция активации ReLU: если значение больше порога, нейрон активируется."""

        base_decay = 0.95
        # Чем выше серотонин, тем быстрее затухание (стабилизация)
        decay_factor = base_decay * (1.0 + (self.nn.SEROTONIN - 1.0) * 0.2)
        decay_factor = max(0.8, min(0.99, decay_factor))  # затухание в пределах 1% - 20% за такт
        self.value *= decay_factor

        noise = 0.0
        if self.nn.SEROTONIN < 1.0:
            # Шум обратно пропорционален уровню серотонина
            noise_scale = (1.0 - self.nn.SEROTONIN) * 0.2  # макс 0.2 при SER = 0
            noise = random.uniform(0, noise_scale)

        effective_stair = self.stair * (1.0 + (self.nn.SEROTONIN - 1.0) * 0.5)
        effective_stair = max(0.1, effective_stair)  # не допускаем слишком низкий порог

        total_input = self.value + noise
        if total_input > effective_stair and not self.is_in_refract():
            self.next_state(True)
            self.state = True
            self.value -= effective_stair  # -= self.stair
            self.activates += 1
            if self.id not in self.nn.active_neurons:
                self.nn.active_neurons.append(self.id)
        else:
            self.next_state(False)
            self.state = False
            if self.id in self.nn.active_neurons:
                self.nn.active_neurons.remove(self.id)
    

    def fire(self, save=False):
        """Если нейрон активен, передаёт значение по связям."""
        if self.state:
            dopamine_gain = max(0.5, min(1.5, self.nn.DOPHAMIN)) #усиление от -50% до 50%
            for i in self.links:
                self.nn[i[0]].value += i[2] * dopamine_gain
                i[1] += 0 if save else 1