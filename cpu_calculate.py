from nn import SpikeNeuralNetwork
import random, time, os
# import matplotlib.pyplot as plt

nn = SpikeNeuralNetwork(10, 2 , 1)

train_data = [
    [[0,1], [1]],
    [[1,1], [1]],
    [[0,0], [0]],
    [[1,0], [1]],
    [[0,1], [1]],
    [[1,1], [1]],
    [[0,0], [0]],
    [[1,0], [1]],
    [[0,1], [1]],
    [[1,1], [1]],
    [[0,0], [0]],
    [[1,0], [1]],
    [[0,1], [1]],
    [[1,1], [1]],
    [[0,0], [0]],
    [[1,0], [1]],
    [[0,1], [1]],
    [[1,1], [1]],
    [[0,0], [0]],
    [[1,0], [1]],
    [[0,1], [1]],
    [[1,1], [1]],
    [[0,0], [0]]    
]

data = random.choice(train_data)
results = []
for _ in range(100):
    os.system("cls")
    result = nn.iteration(data[0])[0]
    results.append(result)
    print(data, result)
    print(nn, flush=True)
    time.sleep(0.1)

i = 0
for j in results:
    if j:
        i += 1

print(f"True: {i}, False: {100-i}")

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