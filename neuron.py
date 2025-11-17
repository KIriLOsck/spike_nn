import random

class Neuron:
    """Класс для представления нейрона в нейросети."""
    def __init__(self, neurons_list):
        self.state = False
        self.links = []  # [индекс, возраст, вес]
        self.stair = 0.5    # Порог активации
        self.value = 0.0    # Мембранный потенциал
        self.activates = 0
        self.prev_state = False

        self.nn = neurons_list
        self.id = len(neurons_list)

    def __str__(self):
        out = ""
        for i in self.links:
            out += str(i) + '\n'
        out += f"Value: {self.value}, State: {self.state}"
        return out


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
        if self.prev_state and self.state and random.random() < self.activates * self.nn._DOPHAMIN / self.nn._ACTIVATION_DEVISOR:
            
            index = self.get_neuron()
            if index is None:
                return
           
            self.nn[index].add_link(self.id, self.nn._BASE_OLD, random.uniform(-0.1,0))


    def hebbs_rule(self):
        """Правило Хебба: нейроны активирующиеся вместе, связываются."""
        index = self.get_neuron()
        if random.random() < self.activates * self.nn._DOPHAMIN / self.nn._ACTIVATION_DEVISOR and self.nn[index].state and self.nn[index].prev_state:
            if index is None:
                return
            
            if self.prev_state and self.nn[index].state:
                self.add_link(index, self.nn._BASE_OLD, random.uniform(0,1) * self.nn._DOPHAMIN)
            elif self.state and self.nn[index].prev_state:
                self.nn[index].add_link(self.id, self.nn._BASE_OLD, random.uniform(0,1) * self.nn._DOPHAMIN)
            else:
                if random.random() < 0.5:
                    self.nn[index].add_link(self.id, self.nn._BASE_OLD, random.uniform(0,1) * self.nn._DOPHAMIN)
                else:
                    self.add_link(index, self.nn._BASE_OLD, random.uniform(0,1) * self.nn._DOPHAMIN)


    def tick(self):
        """Обновляет состояние нейрона, уменьшая возраст связей и удаляя старые связи."""
        rem = []
        for i in self.links:
            i[1] -= 1
            if i[1] < 0:
                rem.append(i)
        for i in rem:
            self.links.pop(self.links.index(i))


    def reLU(self):
        """Функция активации ReLU: если значение больше порога, нейрон активируется."""

        self.value *= 0.95
        if self.value > self.stair:
            self.prev_state = True if self.state else False
            self.state = True
            self.value -= self.stair
            self.activates += 1

            if self.id not in self.nn.active_neurons:
                self.nn.active_neurons.append(self.id)
        else:
            self.prev_state = True if self.state else False
            self.state = False

            if self.id in self.nn.active_neurons:
                self.nn.active_neurons.remove(self.id)

    
    def get_connect(self):
        varible_index = self.get_connect()
        connected = []
        for n in varible_index:
            for l in self.nn[n].links:
                if l[0] == self.id:
                    connected.append(n)
        return connected
    
    def get_useful(self):
        varible_index = self.get_connect()
        useful = []
        for i in varible_index:
            if self.nn[i].state:
                useful.append(i)

    def fire(self, save=False, chain_activation=False):
        """Если нейрон активен, передаёт значение по связям."""
        if self.state and not chain_activation:
            for i in self.links:
                if i[1] >= 10:
                    self.nn[i[0]].value += i[2]
                    i[1] += 0 if save else 1

        try:
            if chain_activation:
                # print(self.value)
                if self.state:
                    for i in self.links:
                        self.nn[i[0]].value += i[2]
                        i[1] += 1
                        self.nn[i[0]].fire(False, True)
                        with open("log.txt", "w") as file:
                            file.write(str(self.nn))
                            
                    self.reLU()
                    self.hebbs_rule()
                    self.activation()
                    self.tick()

        except RecursionError:
            pass
        except PermissionError:
            pass