import random

class Neuron:
    """Класс для представления нейрона в нейросети."""
    def __init__(self, neurons_list):
        self.state = False
        self.links = []  # [индекс, возраст, вес]
        self.stair = 0.5    # Порог активации
        self.value = 0.0    # Мембранный потенциал
        self.activates = 0
        self.last_activates = []
        self.spike_iteration = -1

        self.nn = neurons_list
        self.id = len(neurons_list)

        for _ in range(self.nn._REFRACT_PERIOD):
            self.last_activates.append(False)


    def __str__(self):
        out = ""
        for i in self.links:
            out += str(i) + '\n'
        out += f"Value: {self.value}, State: {self.state}"
        return out
    

    def _is_need_correction(self) -> bool:
        """
        Определяет нужна-ли коррекция нейрону.
        Чем больше активаций нейрона было без коррекции, тем выше шанс коррекции.
        """

        if random.random() < max(0, self.activates) / self.nn._ACTIVATION_DEVISOR:
            self.activates -= self.nn._ACTIVATION_DEVISOR
            return True
        else:
            return False
        

    def _next_state(self, state: bool) -> None:
        """
        Если рефрактный период > 0, добавляет в начало состояние, сохраняе длинну массива.
        Ничего не делает в противном случае.
        """

        if len(self.last_activates) > 0:
            self.last_activates.pop(0)
            self.last_activates.append(state)


    def _add_link(self, index: int, old: float, weight: float) -> None:
        """
        Добавляет связь между нейронами.
        Если связь существует, суммирует вес и возраст.
        Не позволяет создать связь к входным нейронам и от выходных. 
        """

        if index == self.id:
            return
        if self.nn[index] in self.nn.input_neurons:
            return
        if self.nn[self.id] in self.nn.output_neurons:
            return

        for i in self.links:
            if i[0] == index:
                i[2] += weight
                i[1] += old
                return

        self.links.append([index, old, weight])


    def _get_radius(self) -> set:
        """
        Закольцовывет массив нейронов и возвращает индексы ближайших к текущему.
        Может вернуть индексы входных или выходных нейронов.
        """

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


    def _get_neuron(self) -> int:
        """
        Возвращает индекс случайного нейрона в пределах радиуса активации.
        """

        varible_neurons = self._get_radius()
        return random.choice(varible_neurons)



    def is_in_refract(self) -> bool:
        """
        Функция для проверки, не находится ли нейрон в рефрактном периоде.
        """

        if self.nn._REFRACT_PERIOD > len(self.last_activates):
            raise RuntimeError("Impossible to determine the refract period: the activation history too short")
        
        if True in self.last_activates:
            return True
        else:
            return False
    

    def random_inhibitory(self) -> None:
        """
        Проверяет, не перевозбуждается ли нейрон, и создает новые ингибирующие связи если да.
        Нейрон считается перевозбудившимся, если его потенциал больше чем коофицент активации (_ACTIVATION_DEVISOR).
        """

        if self.value > self.nn._ACTIVATION_DEVISOR:
            
            index = self._get_neuron()
            if index is None:
                return
           
            self.nn[index]._add_link(self.id, self.nn._BASE_OLD, random.uniform(-0.5,0))


    def get_activation_chain(self, neurons: list) -> list:
        """
        Узнаёт активировались ли нейроны последовательно и возвращает порядок активации или None
        """

        sorted_neurons = []
        for neuron in neurons: 
            if neuron.spike_iteration == -1:
                continue
            sorted_neurons.append(neuron)
        
        # neurons.times: [99, 22, 74, 12, -1, -1, 69] -> [12, 22, 69, 74, 99]
        # neurons.index: [00, 01, 02, 03, 04, 05, 06] -> [03, 01, 06, 02, 00]

        sorted_neurons.sort(key=lambda n: n.spike_iteration)
        return sorted_neurons if sorted_neurons else None


    def hebbs_rule(self) -> None:
        """
        Правило Хебба: нейроны активирующиеся вместе, связываются.
        Расчитывает цепочку активации нейронов в зависимости он номера итерации на котором они были активированны.
        Создаёт связи сразу между несколькими нейронами.
        """

        connected = self._get_radius()
        self_index = None

        for _ in connected:
            if self._is_need_correction():
                chain = self.get_activation_chain([self] + [self.nn[i] for i in self._get_radius()])
                if self in chain:
                    self_index = chain.index(self)

                if chain is None:
                    continue
                
                if not self_index is None:
                    chain = chain[self_index + 1:]
                else:
                    continue

                for n in range(len(chain) - 1): # от последнего нейрона связь не создаём
                    chain[n]._add_link(chain[n + 1].id, self.nn._BASE_OLD * max(0.01, self.nn.DOPHAMIN - 1), self.nn._BASE_STRENGTH)


    def destroy(self) -> None:
        """
        Обновляет состояние связей. Удаляет старые связи, если дофамин меньше 1.
        """

        rem = []
        age_step = self.nn.DOPHAMIN - 1
        age_step = max(-1, min(-0.01, age_step))  # защита от экстремумов
        for i in self.links:
            i[1] += age_step
            if i[1] < 0:
                rem.append(i)
        for i in rem:
            self.links.pop(self.links.index(i))


    def reLU(self, iteration: int) -> None:
        """
        Модифицированная функция активации ReLU: если значение больше порога, нейрон активируется.
        Учитывает уровень дофамина и серотонина для регуляции уровня шума и внимания.
        Принимает текущий уровень итерации для упрощённого расчёта цепочек активации.
        """

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
        effective_stair = max(0.1, effective_stair)

        self.value += noise
        if self.value > effective_stair and not self.is_in_refract():
            self.spike_iteration = iteration
            self._next_state(True)
            self.state = True
            self.value = effective_stair # или -= self.stair
            self.activates += 1
            if self.id not in self.nn.active_neurons:
                self.nn.active_neurons.append(self.id)
        else:
            self._next_state(False)
            self.state = False
            if self.id in self.nn.active_neurons:
                self.nn.active_neurons.remove(self.id)
    

    def fire(self) -> None:
        """
        Если нейрон активен, передаёт значение по связям.
        Функция ограничивает нижний порог активности нейросети.
        """

        if self.state:
            for i in self.links:
                self.nn[i[0]].value = max(i[2] + self.nn[i[0]].value, -1)
                i[1] += max(0.0, self.nn.DOPHAMIN - 1)