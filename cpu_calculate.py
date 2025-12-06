from nn import SpikeNeuralNetwork
import random, time, json
from datetime import datetime
from multiprocessing import Pool

# import matplotlib.pyplot as plt

# nn = SpikeNeuralNetwork(10, 3 , 2)
# nn.SEROTONIN = 0.1
# nn._ACTIVATION_RADIUS = 3
# nn._BASE_STRENGTH = 0.01
# nn._BASE_OLD = 25
# nn.learning_rate = 0.5

# train_data = [
#     [[1, 1, 1],[1, 0]],
#     [[1, 1, 0],[0, 1]],
#     [[1, 0, 0],[0, 1]],
#     [[1, 1, 1],[1, 0]],
#     [[0, 1, 0],[0, 1]],
#     [[0, 0, 1],[0, 1]],
#     [[1, 1, 1],[1, 0]],
#     [[0, 1, 1],[0, 1]],
#     [[1, 1, 1],[1, 0]],
#     [[1, 1, 1],[1, 0]]
# ]

# nn.train(train_data, 100, 10)
# print(nn)

# nn.SEROTONIN = 2.0
# nn.DOPHAMIN = 1.0

# print("Проверка на случайных данных...")
# accuratly = []
# for _ in range(5):
#     data = random.choice(train_data)
#     result = nn.test_result(data)[0]
#     print(data[0], end=" ")
#     print(f"Точность: {result}%")
#     accuratly.append(result)

# print(f"Итоговая точность: {sum(accuratly) / len(accuratly)}")

def dataWorker(i):
    start = datetime.now()

    neurons = random.randint(10, 250)
    radius = random.randint(2, neurons // 2)
    serotonin = random.uniform(0, 1)
    rate = random.uniform(0, 1)
    strenght = random.uniform(0, 1)
    old = random.uniform(0, 100)
    iterations = random.randint(1, 100)
    iter_step = random.randint(1, 100)

    nn = SpikeNeuralNetwork(neurons, 3, 2)
    nn.SEROTONIN = serotonin
    nn.learning_rate = rate

    nn._ACTIVATION_RADIUS = radius
    nn._BASE_STRENGTH = strenght
    nn._BASE_OLD = old

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
        [[1, 1, 1],[1, 0]]
    ]

    nn.train(train_data, iterations, iter_step)

    accuratly = []
    for _ in range(5):
        data = random.choice(train_data)
        result = nn.test_result(data)[0]
        accuratly.append(result)

    mid_accuratly = sum(accuratly) / len(accuratly)

    if sum(accuratly) % 50 == 0:
        mid_accuratly = 0

    work_time = str(datetime.now() - start)

    results = {
        "neurons": neurons,
        "radius": radius,
        "serotonin": serotonin,
        "rate": rate,
        "strenght": strenght,
        "old": old,
        "iterations": iterations,
        "iter_step": iter_step,
        "time": work_time,
        "mid_accuratly": mid_accuratly,
        "accuratly": accuratly
            }
    
    results = json.dumps(results) + ",\n\t"

    while 1:
        try:
            with open("log.json", "a") as file:
                file.write(results)

            print(i)
            return

        except:
            time.sleep(random.uniform(0, 10))
            continue

if __name__ == "__main__":
    with Pool(processes=8) as pool:
        results = pool.map(dataWorker, range(1000))

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