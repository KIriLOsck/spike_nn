from nn import SpikeNeuralNetwork
import random, time
# import matplotlib.pyplot as plt

nn = SpikeNeuralNetwork(15, 3 , 2)
nn.SEROTONIN = 0.5
nn._ACTIVATION_RADIUS = 3
nn._BASE_STRENGTH = 0.02
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

    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 1],[1, 0]],
    [[1, 1, 0],[0, 1]],
    [[1, 0, 0],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 0],[0, 1]],
    [[0, 0, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
    [[0, 1, 1],[0, 1]],
    [[1, 1, 1],[1, 0]],
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
for _ in range(10):
    batch = random.choice(train_data)
    for i in range(10):
        nn.iteration(l, batch[0])
        if i % 5 == 0:
            if nn.get_outputs(reLU=True) == batch[1]:
                nn.DOPHAMIN += 0.5
            else:
                nn.DOPHAMIN -= 0.1
        print(nn, flush=True)
        time.sleep(0)
        l += 1
        print(nn.DOPHAMIN)

print(nn)

nn.SEROTONIN = 2.0
nn.DOPHAMIN = 1.0

print("Проверка на случайных данных...")
for _ in range(5):
    data = random.choice(train_data)
    print(data[0], end=" ")
    print(f"Точность: {test_result(data)[0]}%")

# plot_loss = [[], []]
# plot_dofamine = [[], []]
# plot_min_loss = [[], []]
# plot_mid_loss = [[], []]

# for batch, states in enumerate(history):
#     plot_loss[0].append(batch)
#     plot_loss[1].append(states[0])

#     plot_dofamine[0].append(batch)
#     plot_dofamine[1].append(states[3])

#     plot_min_loss[0].append(batch)
#     plot_min_loss[1].append(states[1])

#     plot_mid_loss[0].append(batch)
#     plot_mid_loss[1].append(states[2])

# plt.plot(*plot_loss, label="Loss", color='yellow')
# plt.plot(*plot_min_loss, label="Weights", color='red')
# plt.plot(*plot_dofamine, label="Dofamine", color='blue')
# plt.plot(*plot_mid_loss, label="Mid Loss", color='green')
# plt.grid(True)  
# plt.legend()
# plt.show()