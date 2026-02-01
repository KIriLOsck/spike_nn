from nn import SpikeNeuralNetwork
import random, time, json

def collect_plot_data(
        iterations: int,
        step: int,
        neurons: int,
        radius: int,
        old: int = 9,
    ):

    data=[]

    for i in range(300):
        print(i)
        nn = SpikeNeuralNetwork(neurons, 2, 2)

        rate = random.uniform(0, 1)
        serotonin = random.uniform(0, 1)
        strenght = random.uniform(0, 1)

        nn.learning_rate = rate
        nn._ACTIVATION_RADIUS = radius
        nn.SEROTONIN = serotonin
        nn._BASE_OLD = old
        nn._BASE_STRENGTH = strenght

        train_data = [
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
            [[1, 1], [1, 1]]
        ]

        nn.train(train_data, iterations, step)

        accuratly = []
        for _ in range(5):
            batch = random.choice(train_data)
            result = nn.test_result(batch, step)[0]
            accuratly.append(result)

        mid_accuratly = sum(accuratly) / len(accuratly)

        if sum(accuratly) % 50 == 0:
            mid_accuratly = 0

        results = {
            "serotonin": serotonin,
            "rate": rate,
            "strenght": strenght,
            "old": old,
            "mid_accuratly": mid_accuratly,
                }
        
        data.append(results)

    return data



def dataWorker(i):
    from datetime import datetime

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

def generate_data(count: int):
    from multiprocessing import Pool

    with Pool(processes=8) as pool:
        pool.map(dataWorker, range(count))

def generate_plot(data, parametr: str, multiplier: int = 1):
    main_plot = [[], []]
    secondary_plot = [[], []]

    data.sort(key=lambda n: n[parametr])

    for i in range(0, len(data), len(data)//multiplier):
        main_plot[0].append(data[i][parametr])
        mid_accuratly = 0
        for j in data[i:i+len(data) // multiplier]:
            mid_accuratly += data[i]["mid_accuratly"]

        mid_accuratly /= len(data) // multiplier
        main_plot[1].append(mid_accuratly)

    mid_accuratly = 0
    for i in range(len(data)):
        if not i:
            mid_accuratly += data[i]["mid_accuratly"]
            secondary_plot[1].append(mid_accuratly)
            secondary_plot[0].append(data[i][parametr])
            continue

        mid_accuratly += data[i]["mid_accuratly"]

        secondary_plot[1].append(mid_accuratly / i)
        secondary_plot[0].append(data[i][parametr])

    return main_plot, secondary_plot

def show_plots(data):
    import matplotlib.pyplot as plt

    _, serotonin_mid = generate_plot(data, "serotonin")
    _, strength_mid = generate_plot(data, "strenght")
    _, radius_mid = generate_plot(data, "old")

    for i in range(len(radius_mid[0])):
        radius_mid[0][i] = radius_mid[0][i] / max(radius_mid[0])

    _, rate_mid = generate_plot(data, "rate")
        
    plt.plot(*radius_mid, label="Mid accuratly to radius", color='red')
    plt.plot(*serotonin_mid, label="Mid accuratly to serotonin", color='blue')
    plt.plot(*strength_mid, label="Mid accuratly to strenght", color='green')
    plt.plot(*rate_mid, label="Mid accuratly to rate", color='purple')


    plt.grid(True)  
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # data = []
    # with open("log.json", "r") as file:
    #     data = json.loads(file.read())
    # show_plots(data)

    data = collect_plot_data(100, 5, 10, 2)
    show_plots(data)

    nn = SpikeNeuralNetwork(50, 2 , 2)
    nn.SEROTONIN = 0.2
    nn._ACTIVATION_RADIUS = 3
    nn._BASE_STRENGTH = 0.21
    nn._BASE_OLD = 9
    nn.learning_rate = 0.5

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

    train_data = [
        [[0, 1], [1, 0]],
        [[1, 0], [0, 1]],
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]]
    ]

    nn.train(train_data, 100, 10)
    print(nn)

    nn.SEROTONIN = 2.0
    nn.DOPHAMIN = 1.0

    print("Проверка на случайных данных...")
    accuratly = []
    for _ in range(5):
        data = random.choice(train_data)
        result = nn.test_result(data, 10)[0]
        print(data[0], end=" ")
        print(f"Точность: {result}%")
        accuratly.append(result)

    print(f"Итоговая точность: {sum(accuratly) / len(accuratly)}")