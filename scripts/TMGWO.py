import numpy as np

np.random.seed(42)


class TMGWO:
    def __init__(
        self,
        fitness,
        D=0,  # dimensi/jumlah fitur
        P=0,  # jumlah wolves
        G=0,  # jumlah iterasi
        Mp=0,  # probabilitas mutasi
    ):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.Mp = Mp

        self.x_max = 1
        self.x_min = 0
        self.a_max = 2
        self.a_min = 0
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
            a = self.a_max - (self.a_max - self.a_min) * (self._iter / self.G)  # eq 3.4

            for i in range(self.P):
                # buat X1
                r1 = np.random.uniform(size=self.D)
                r2 = np.random.uniform(size=self.D)
                A = 2 * a * r1 - a  # eq 3.5
                C = 2 * r2  # eq 3.6
                D = np.abs(C * self.X_alpha - self.X[i, :])  # eq 3.11
                X1 = self.X_alpha - A * D  # eq 3.8
                # buat X2
                r1 = np.random.uniform(size=self.D)
                r2 = np.random.uniform(size=self.D)
                A = 2 * a * r1 - a  # eq 3.5
                C = 2 * r2  # eq 3.6
                D = np.abs(C * self.X_beta - self.X[i, :])  # eq 3.12
                X2 = self.X_beta - A * D  # eq 3.9
                # buat X3
                r1 = np.random.uniform(size=self.D)
                r2 = np.random.uniform(size=self.D)
                A = 2 * a * r1 - a  # eq 3.5
                C = 2 * r2  # eq 3.6
                D = np.abs(C * self.X_delta - self.X[i, :])  # eq 3.13
                X3 = self.X_delta - A * D  # eq 3.10
                X_pos_cont = (
                    X1 + X2 + X3
                ) / 3  # eq 3.7 (continuous position, need to be converted to binary)
                self.X[i, :] = np.random.uniform(size=self.D) >= 1 / (
                    1 + np.exp(-X_pos_cont)
                )  # eq 3.14 convert continuous position to binary using sigmoid function

            self.update_score()
            fitness = self.fitness(self.X_alpha)
            Xmutated1 = self.X_alpha.copy()
            for i in range(len(self.X_alpha)):  # first mutation
                r = np.random.uniform()

                if r < self.Mp and self.X_alpha[i] == 1:
                    Xmutated1[i] = 0
                    fitness_mutated = self.fitness(Xmutated1)
                    if fitness_mutated < fitness:
                        fitness = fitness_mutated.copy()
                        self.X_alpha = Xmutated1.copy()

            fitness = self.fitness(self.X_alpha)
            Xmutated2 = self.X_alpha.copy()
            for i in range(len(self.X_alpha)):  # second mutation
                r = np.random.uniform()

                if r < self.Mp and self.X_alpha[i] == 0:
                    Xmutated2[i] = 1
                    fitness_mutated = self.fitness(Xmutated2)
                    if fitness_mutated < fitness:
                        fitness = fitness_mutated.copy()
                        self.X_alpha = Xmutated2.copy()

            self.score_alpha = self.fitness(self.X_alpha)
            self._iter = self._iter + 1

    def update_score(self):
        score_all = self.fitness(self.X)
        for score in sorted(score_all):  # iterate through all scores
            if score < self.score_alpha:  # if current score is better than alpha
                self.score_alpha = score.copy()  # update alpha
                pos = list(score_all).index(score)  # get position of alpha
                self.X_alpha = self.X[pos, :].copy()  # update alpha position
            elif (
                score > self.score_alpha  # if current score is not better than alpha
                and score < self.score_beta  # but better than beta
            ):
                self.score_beta = score.copy()  # update beta
                pos = list(score_all).index(score)  # get position of beta
                self.X_beta = self.X[pos, :].copy()  # update beta position
            elif (
                score > self.score_alpha  # if current score is not better than alpha
                and score > self.score_beta  # and not better than beta
                and score < self.score_delta  # but better than delta
            ):
                self.score_delta = score.copy()  # update delta
                pos = list(score_all).index(score)  # get position of delta
                self.X_delta = self.X[pos, :].copy()  # update delta position

        self.gBest_X = self.X_alpha.copy()
        self.gBest_score = self.score_alpha.copy()
        self.gBest_curve[self._iter] = self.score_alpha.copy()
