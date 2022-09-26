import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


class TMGWO:
    def __init__(
        self,
        fitness,
        D=30,
        P=20,
        G=500,
        x_max=1,
        x_min=0,
        a_max=2,
        a_min=0,
    ):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.x_max = x_max
        self.x_min = x_min
        self.a_max = a_max
        self.a_min = a_min
        self.Mp = 0.5

        self._iter = 0
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.G)
        self.score_alpha = np.inf
        self.score_beta = np.inf
        self.score_delta = np.inf
        self.X_alpha = None
        self.X_beta = None
        self.X_delta = None

        self.X = np.random.choice(2, size=[self.P, self.D])

        self.update_score()

        self._itter = self._iter + 1

    def optimize(self):
        while self._iter < self.G:
            a = self.a_max - (self.a_max - self.a_min) * (self._iter / self.G)

            for i in range(self.P):
                r1 = np.random.uniform(size=self.D)
                r2 = np.random.uniform(size=self.D)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.X_alpha - self.X[i, :])
                X1 = self.X_alpha - A * D

                r1 = np.random.uniform(size=self.D)
                r2 = np.random.uniform(size=self.D)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.X_beta - self.X[i, :])
                X2 = self.X_beta - A * D

                r1 = np.random.uniform(size=self.D)
                r2 = np.random.uniform(size=self.D)
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * self.X_delta - self.X[i, :])
                X3 = self.X_delta - A * D

                self.X[i, :] = np.mean([X1, X2, X3])

                self.X[i, :] = np.random.uniform() >= 1 / (
                    1 + np.exp(-1 * self.X[i, :])
                )

            self.update_score()

            fitness = self.fitness(self.X_alpha)
            Xmutated1 = self.X_alpha.copy()
            for i in range(len(self.X_alpha)):
                r = np.random.uniform()

                if r < self.Mp and self.X_alpha[i] == 1:
                    Xmutated1[i] = 0
                    fitness_mutated = self.fitness(Xmutated1)
                    if fitness_mutated < fitness:
                        fitness = fitness_mutated.copy()
                        self.X_alpha = Xmutated1.copy()

            fitness = self.fitness(self.X_alpha)
            Xmutated2 = self.X_alpha.copy()
            for i in range(len(self.X_alpha)):
                r = np.random.uniform()

                if r < self.Mp and self.X_alpha[i] == 0:
                    Xmutated2[i] = 1
                    fitness_mutated = self.fitness(Xmutated2)
                    if fitness_mutated < fitness:
                        fitness = fitness_mutated.copy()
                        self.X_alpha = Xmutated2.copy()

            self.score_alpha = self.fitness(self.X_alpha)
            self._iter = self._iter + 1

    def plot_curve(self):
        plt.figure()
        plt.title("loss curve [" + str(round(self.gBest_curve[-1], 3)) + "]")
        plt.plot(self.gBest_curve, label="loss")
        plt.grid()
        plt.legend()
        plt.show()

    def update_score(self):
        score_all = self.fitness(self.X)
        for idx, score in enumerate(score_all):
            if score < self.score_alpha:
                self.score_alpha = score.copy()
                self.X_alpha = self.X[idx, :].copy()

            if score > self.score_alpha and score < self.score_beta:
                self.score_beta = score.copy()
                self.X_beta = self.X[idx, :].copy()

            if (
                score > self.score_alpha
                and score > self.score_beta
                and score < self.score_delta
            ):
                self.score_delta = score.copy()
                self.X_delta = self.X[idx, :].copy()

        self.gBest_X = self.X_alpha.copy()
        self.gBest_score = self.score_alpha.copy()
        self.gBest_curve[self._iter] = self.score_alpha.copy()
