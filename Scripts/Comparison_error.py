import numpy as np
import matplotlib.pyplot as plt

# Read error lists from text files
error_list1 = np.loadtxt('error_list_pytorch.txt')
error_list2 = np.loadtxt('error_list_numpy_PCA.txt')

# Create x-axis values for the error lists
iterations = np.arange(len(error_list1))
print(len(error_list1))
print(len(error_list2))

# Plotting the error lists
plt.plot(iterations*10, error_list1, label='Pytorch - step 100000000 - time 12.00948429107666 seconds')
plt.plot(iterations*10, error_list2, label='Numpy - step 1000 - time 132.37873601913452 seconds.')

# Add labels and title to the plot
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Error List Comparison')

# Add legend
plt.legend()

# Display the plot
plt.show()
