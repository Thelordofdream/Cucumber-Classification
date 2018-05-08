import numpy as np
import matplotlib.pyplot as plt

gen_error_1 = np.load('generation_v1.npy')
gen_error_2 = np.load('generation_v2.npy')
gen_error_3 = np.load('generation_v3.npy')
x = [i * 10 for i in range(1, 501)]
plt.plot(x, gen_error_1, 'b-', label="ResNet-47-P1")
plt.plot(x, gen_error_2, 'g-', label="ResNet-47-P2-all")
plt.plot(x, gen_error_3, 'r-', label="ResNet-47-P2-sub")
plt.legend(loc='best')
plt.show()