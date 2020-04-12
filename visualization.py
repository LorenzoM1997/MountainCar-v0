import numpy as np
import matplotlib
import matplotlib.pyplot as plt

returnsArray = np.load("returns.npy")
returns = returnsArray.tolist()

n_iter = len(returns)

iterations = range(0, n_iter)

plt.figure(figsize=(10,10), dpi=80)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.savefig('returns')
