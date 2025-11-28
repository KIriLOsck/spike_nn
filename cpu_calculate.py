from nn import SpikeNeuralNetwork
import random, time, os
# import matplotlib.pyplot as plt

nn = SpikeNeuralNetwork(15, 3 , 2)
nn.SEROTONIN = 0.1
nn._ACTIVATION_RADIUS = 3
nn._BASE_STRENGTH = 0.02
nn._BASE_OLD = 10
nn.learning_rate = 0.99

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
data2 = random.choice(train_data)
data3 = random.choice(train_data)

results1 = []
results2 = []
results3 = []

l = 0
for _ in range(10):
    batch = random.choice(train_data)
    for _ in range(10):
        nn.iteration(l, batch[0])
        if nn.get_outputs(reLU=True) == batch[1]:
            nn.DOPHAMIN += 0.5
        else:
            nn.DOPHAMIN -= 0.1
        print(nn, flush=True)
        time.sleep(0.1)
        l += 1
        print(nn.DOPHAMIN)

nn.SEROTONIN = 2.0
nn.DOPHAMIN = 1.0

for i in nn:
    i.spike_itertion = -1
    i.value = 0.0

for i in range(0, 100, 1):
    result1 = nn.iteration(i, data[0], reLU=True)
    results1.append(
        result1
    )

for i in range(100, 200, 1):
    result2 = nn.iteration(i, data2[0], reLU=True)
    results2.append(
        result2
    )


for i in range(200, 300, 1):
    result3 = nn.iteration(i, data3[0], reLU=True)
    results3.append(
        result3
    )


i = [0, 0]
for k in results1:
    if k[0]:
        i[0] += 1
    if k[1]:
        i[1] += 1

oi = [0, 0]
for k in results2:
    if k[0]:
        oi[0] += 1
    if k[1]:
        oi[1] += 1

ai = [0, 0]
for k in results3:
    if k[0]:
        ai[0] += 1
    if k[1]:
        ai[1] += 1

print(nn)
print(data, i)
print(data2, oi)
print(data3, ai)
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