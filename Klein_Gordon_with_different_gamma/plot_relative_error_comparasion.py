'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：plot_relative_error_comparasion.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/18 下午3:56 
'''
import matplotlib.pyplot as plt
import math

# Data
lam_threshold = [1, 10, 100, 1000, 10000, 100000, 1000000]
relative_l2_error = [0.03777153276675437, 0.034358915340732966, 0.00972611708929684,
                     0.005988868888697431, 0.005994103986349339, 0.005911623278269709,
                     0.005427416513077558]

plt.figure(figsize=(10, 8))
plt.semilogx(lam_threshold, relative_l2_error, marker='o', linestyle='-', color='b')

plt.xlabel('$\gamma$', fontsize=14)
plt.ylabel('Relative Error', fontsize=14)
plt.yscale('log')

ax = plt.gca()
ax.set_xticks(lam_threshold)
ax.set_xticklabels([f'$10^{int(math.log10(lam))}$' for lam in lam_threshold])

plt.legend(['Relative Error'], loc='best', fontsize=12)
plt.grid(True)
plt.savefig('relative_l2_error_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()