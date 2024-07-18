'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：plot.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/17 下午7:26 
'''
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = {
    "model_type": ["IAW_PINN"] * 6,
    "network_structure": ["3x50", "3x70", "5x50", "5x70", "7x50", "7x70"],
    "relative_error_u_nw": [0.083709621, 0.016373056, 0.009395324, 0.007045447, 0.032403815, 0.008975276],
    "relative_error_f_nw": [0.025520518, 0.013280908, 0.013691637, 0.00918487, 0.027506691, 0.008572734],
    "relative_error_u_1000": [0.00957207, 0.00499067, 0.00591017, 0.00528927, 0.00715489, 0.00610221],
    "relative_error_f_1000": [0.00960518, 0.00469967, 0.00543392, 0.00367533, 0.00811686, 0.00403999]
}

df = pd.DataFrame(data)

# Plotting the data
fig1, ax = plt.subplots(figsize=(10, 6))

# Plot for f_nw and u_nw
ax.plot(df['network_structure'], df['relative_error_u_nw'], marker='o', label='$\gamma =\infty$')
ax.plot(df['network_structure'], df['relative_error_u_1000'], marker='s', label='$\gamma =10^3$')
ax.set_xlabel('Network Structure')
ax.set_ylabel('Relative Error u')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("relative_errors_u.pdf")
plt.show()

fig2, ax = plt.subplots(figsize=(10, 6))

# Plot for f_1000 and u_1000
ax.plot(df['network_structure'], df['relative_error_f_nw'],  marker='o', label='$\gamma =\infty$')
ax.plot(df['network_structure'], df['relative_error_f_1000'], marker='s', label='$\gamma =10^3$')
ax.set_xlabel('Network Structure')
ax.set_ylabel('Relative Error f')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("relative_errors_f.pdf")
plt.show()

# Calculate the average of u and f for both sets
df['average_error_nw'] = (df['relative_error_u_nw'] + df['relative_error_f_nw']) / 2
df['average_error_1000'] = (df['relative_error_u_1000'] + df['relative_error_f_1000']) / 2

# Plotting the data
fig3, ax = plt.subplots(figsize=(10, 6))

# Plot for average errors
ax.plot(df['network_structure'], df['average_error_nw'], marker='o', label='$\gamma =\infty$')
ax.plot(df['network_structure'], df['average_error_1000'], marker='s', label='$\gamma =10^3$')
ax.set_xlabel('Network Structure')
ax.set_ylabel('Average Relative Error')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig("average_relative_errors.pdf")
plt.show()