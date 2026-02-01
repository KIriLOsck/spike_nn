from nn import SpikeNeuralNetwork
import random
import struct

neurons = 10
radius = 3

nn = SpikeNeuralNetwork(neurons, 3, 2)

nn.SEROTONIN = 0.1
nn._ACTIVATION_RADIUS = radius
nn._BASE_STRENGTH = 0.05
nn._BASE_OLD = 10
nn.learning_rate = 0.8


train_data = [
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
]

data = random.choice(train_data)

def test_result(batch):
    for i in nn:
        i.spike_itertion = -1
        i.value = 0.0

    results = []
    for i in range(0, 100, 1):
        result = nn.iteration(i, batch[0], reLU=True)
        results.append(
            result
        )

    spikes = [0 for _ in range(len(result))]
    for out in results:
        for spike in range(len(out)):
            spikes[spike] += 1 if out[spike] else 0

    spikes_count = sum(spikes)
    correct = 0

    for i in batch[1]:
        if i:
            correct += 1

    maximals = [0 for i in range(correct)]

    maximum = 0
    for i in spikes:
        if maximum < i:
            maximum = i
            maximals.pop(0)
            maximals.append(i)

    correct_spikes = 0

    for result, value in zip(batch[1], spikes):
        if result:
            if value in maximals:
                correct_spikes += value

    return round((correct_spikes / (spikes_count or 1)) * 100, 1), spikes

l = 0
for _ in range(100):
    batch = random.choice(train_data)
    for i in range(10):
        nn.iteration(l, batch[0])
        if i % 5 == 0:
            if nn.get_outputs(reLU=True) == batch[1]:
                nn.DOPHAMIN += 0.5
            else:
                nn.DOPHAMIN -= 0.1
        l += 1

byte_model = b''
print(nn)

for neuron in nn:  
    byte_model += struct.pack("f", neuron.value)
    byte_model += (b'\xFF' if neuron.state else b'\x00')
    for linkIndex in range(radius * 2):
        if len(neuron.links) > linkIndex:
            index = (neuron.links[linkIndex][0] - neuron.id) + radius
            # print(index)
            if index > radius * 2 or index < 0 or neuron.id - (neuron.links[linkIndex][0] - neuron.id) < radius:
                byte_model += b'\xFF'
                byte_model += bytes(4)
                # print('skipping...')
                continue
            byte_model += struct.pack("b", index)
            byte_model += struct.pack("f", neuron.links[linkIndex][2])
        else:
            byte_model += b'\xFF'
            byte_model += bytes(4)

print(len(byte_model))
counter = 0
for i in byte_model:
    
    if counter % 25 == 0:
        print()
    print(hex(i), end=(" | " if i > 15 else "  | "))
    counter += 1

with open("model.bin", "bw") as file:
    file.write(byte_model)