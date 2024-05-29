#ProjAY

import numpy as np
import matplotlib.pyplot as plt
import numpy as np



class Person:
    def __init__(self, fitness, dim, minx, maxx):
        self.position = [minx + np.random.rand() * (maxx - minx) for _ in range(dim)]
        self.fitness = fitness(self.position)

def f1(position):  # rastrigin
    return np.sum([(xi * xi) - (10 * np.cos(2 * np.pi * xi)) + 10 for xi in position])
def f2(position):  # sphere
    fitness_value = 0.0
    for i in range(len(position)):
        xi = position[i]
        fitness_value += (xi * xi)
    return fitness_value;


def f3(position):  # rosenbrock
    fitness_value = 0.0
    for i in range(len(position) - 1):
        xi = position[i]
        xi1 = position[i + 1]
        fitness_value += 100 * (xi1 - xi ** 2) ** 2 + (1 - xi) ** 2
    return fitness_value


def f4(position):  # griewank
    fitness_value = 0.0
    sum_term = 0.0
    prod_term = 1.0
    for i in range(len(position)):
        xi = position[i]
        sum_term += xi ** 2 / 4000
        prod_term *= np.cos(xi / np.sqrt(i + 1))
    fitness_value = 1 + sum_term - prod_term
    return fitness_value


def f5(position):  # ackley
    dim = len(position)
    sum_term = np.sum(np.square(position))
    cos_term = np.sum(np.cos(2 * np.pi * np.array(position)))
    fitness_value = -20 * np.exp(-0.2 * np.sqrt(sum_term / dim)) - np.exp(cos_term / dim) + 20 + np.exp(1)
    return fitness_value
def sgo(fitness, max_iter, n, dim, minx, maxx):
    Fitness_Curve = np.zeros(max_iter)
    society = [Person(fitness, dim, minx, maxx) for _ in range(n)]
    Xbest = [0.0 for _ in range(dim)]
    c = 0.2
    Fbest = float('inf')

    for i in range(n):
        if society[i].fitness < Fbest:
            Fbest = society[i].fitness
            Xbest = np.copy(society[i].position)

    Iter = 0
    while Iter < max_iter:
        for i in range(n):
            Xnew = [c * society[i].position[j] + np.random.rand() * (Xbest[j] - society[i].position[j])
                    for j in range(dim)]
            Xnew = [max(min(x, maxx), minx) for x in Xnew]

            fnew = fitness(Xnew)
            if fnew < society[i].fitness:
                society[i].position = Xnew
                society[i].fitness = fnew
            if fnew < Fbest:
                Fbest = fnew
                Xbest = Xnew

        Fitness_Curve[Iter] = Fbest
        Iter += 1

    return Xbest, Fitness_Curve

def mc(fitness, max_iter, n, dim, minx, maxx, theta, beta, gamma):
    Fitness_Curve = np.zeros(max_iter)
    society = [Person(fitness, dim, minx, maxx) for _ in range(n)]
    Xbest = [0.0 for _ in range(dim)]
    Fbest = float('inf')

    for i in range(n):
        if society[i].fitness < Fbest:
            Fbest = society[i].fitness
            Xbest = np.copy(society[i].position)

    Iter = 0
    while Iter < max_iter:
        for i in range(n):
            Xnew = [theta * society[i].position[j] + beta * np.random.rand() * (Xbest[j] - society[i].position[j])
                    + gamma * np.random.rand() * (maxx - minx)
                    for j in range(dim)]
            Xnew = [max(min(x, maxx), minx) for x in Xnew]

            fnew = fitness(Xnew)
            if fnew < society[i].fitness:
                society[i].position = Xnew
                society[i].fitness = fnew
            if fnew < Fbest:
                Fbest = fnew
                Xbest = Xnew

        Fitness_Curve[Iter] = Fbest
        Iter += 1

    return Fitness_Curve
def qmc(fitness, max_iter, n, dim, minx, maxx, theta, beta, gamma):
    Fitness_Curve = np.zeros(max_iter)
    society = [Person(fitness, dim, minx, maxx) for _ in range(n)]
    Xbest = [0.0 for _ in range(dim)]
    Fbest = float('inf')

    Iter = 0
    while Iter < max_iter:
        for i in range(n):
            # Use numpy for generating quasi-random numbers
            quasi_random = np.random.rand(2 * dim)
            Xnew = [theta * society[i].position[j] +
                    beta * quasi_random[j] * (Xbest[j] - society[i].position[j]) +
                    gamma * quasi_random[dim + j] * (maxx - minx)
                    for j in range(dim)]
            Xnew = [max(min(x, maxx), minx) for x in Xnew]

            fnew = fitness(Xnew)
            if fnew < society[i].fitness:
                society[i].position = Xnew
                society[i].fitness = fnew
            if fnew < Fbest:
                Fbest = fnew
                Xbest = Xnew

        Fitness_Curve[Iter] = Fbest
        Iter += 1

    return Fitness_Curve

def plot_fitness_curve(algorithm, fitness_curve, title):
    plt.plot(range(len(fitness_curve)), fitness_curve, label=algorithm)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.show()

# User input
selected_function = int(input("Select a function : "))
max_iter = 1000
n = 20
dim = 30
minx = -5.12
maxx = 5.12

fitness_function = None
function_name = ""

if selected_function == 1:
    fitness_function = f1
    function_name = "Rastrigin"
if selected_function == 2:
    fitness_function = f2
    function_name = "Sphere"
if selected_function == 3:
    fitness_function = f3
    function_name = "Rosenbrock"
if selected_function == 4:
    fitness_function = f4
    function_name = "Griewank"
if selected_function == 5:
    fitness_function = f5
    function_name = "Ackley"


def plot_fitness_curves(algorithms, fitness_curves, titles, function_name):
    for algorithm, curve in zip(algorithms, fitness_curves):
        plt.plot(range(len(curve)), curve, label=f"{algorithm} - {function_name}")

    plt.title("Fitness Curves")
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Value')
    plt.legend()
    plt.show()

# Run SGO
Xbest_sgo, Fitness_Curve_sgo = sgo(fitness_function, max_iter, n, dim, minx, maxx)

# Run MC
theta_mc = 0.9
beta_mc = 0.5
gamma_mc = 1.5
Fitness_Curve_mc = mc(fitness_function, max_iter, n, dim, minx, maxx, theta_mc, beta_mc, gamma_mc)

# Run QMC
theta_qmc = 0.9
beta_qmc = 0.5
gamma_qmc = 1.5
Fitness_Curve_qmc = qmc(fitness_function, max_iter, n, dim, minx, maxx, theta_qmc, beta_qmc, gamma_qmc)

# Plot all fitness curves on a single graph
algorithms = ["SGO", "MC", "QMC"]
fitness_curves = [Fitness_Curve_sgo, Fitness_Curve_mc, Fitness_Curve_qmc]
titles = ["Fitness Curve - SGO", "Fitness Curve - MC", "Fitness Curve - QMC"]

plot_fitness_curves(algorithms, fitness_curves, titles, function_name)
