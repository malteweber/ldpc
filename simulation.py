#%%
import numpy as np

from utils.graph_utils import girth_average, girth_median, girth_min_node_fraction
from utils.ldpc_utils import create_parity_check_matrix
from utils.graph_utils import create_tanner_graph as tanner_graph
from utils.simulation_utils import run_binary_channel_simulation, std_from_signal_to_noise_ration, run_gaussian_channel_simulation

from ldpc import LDPC
#%%
n = 400
w_r = 3
w_c = 2
p_values = [i/100 for i in range(1, 10)]
snr_values = [*range(1, 10)]
num_samples = 100
num_iterations = 50
nr_of_matrices = 500
#%%

ga_values = []
ga_best_values = 0
ga_best_matrix = None


gmedian_values = []
gmedian_best_value = 0
gmedian_best_matrix = None


gmin_values = []
gmin_best_value = 0
gmin_best_matrix = None


for i in range(nr_of_matrices):
    if i % 50 == 0:
        print(f'Iteration {i}')
    matrix = create_parity_check_matrix(n, w_r, w_c)
    
    if matrix is not None: 
        ga = girth_average(tanner_graph(matrix))
        gm = girth_median(tanner_graph(matrix))
        gmin = girth_min_node_fraction(tanner_graph(matrix))
        if ga != np.inf: ga_values.append(ga)
        if gm != np.inf: gmedian_values.append(gm)
        if gmin != np.inf: gmin_values.append(gmin)
    
        if ga > ga_best_values:
            ga_best_values = ga
            ga_best_matrix = matrix
    
        if gm > gmedian_best_value:
            gmedian_best_value = gm
            gmedian_best_matrix = matrix
            
        if gmin > gmin_best_value:
            gmin_best_value = gmin
            gmin_best_matrix = matrix
            
    else:
        print('Matrix not valid')

random_matrix = None
while random_matrix is None:
    random_matrix = create_parity_check_matrix(n, w_r, w_c)
#%%

import matplotlib.pyplot as plt
plt.hist(ga_values, bins=50)
plt.axvline(ga_best_values, color='r', linestyle='dashed', linewidth=1)
plt.axvline(girth_average(tanner_graph(random_matrix)), color='g', linestyle='dashed', linewidth=1)

plt.xlabel('Girth Average')
plt.ylabel('Häufigkeit')
plt.title('Girth Average Histogramm')

plt.legend(['Best', 'Random'])

#plt.show()
plt.savefig('girth_average_histogram.png')
#%%

plt.hist(gmedian_values, bins=50)
plt.axvline(gmedian_best_value, color='r', linestyle='dashed', linewidth=1)
plt.axvline(girth_median(tanner_graph(random_matrix)), color='g', linestyle='dashed', linewidth=1)

plt.xlabel('Girth Median')
plt.ylabel('Häufigkeit')
plt.title('Girth Median Histogram')

plt.legend(['Best', 'Random'])

#plt.show()
plt.savefig('girth_median_histogram.png')
#%%

plt.hist(gmin_values, bins=50)
plt.axvline(gmin_best_value, color='r', linestyle='dashed', linewidth=1)
plt.axvline(girth_min_node_fraction(tanner_graph(random_matrix)), color='g', linestyle='dashed', linewidth=1)

plt.xlabel('Knotenanteil mit niedrigstem Girth')
plt.ylabel('Häufigkeit')
plt.title('Knotenanteil mit niedrigstem Girth Histogramn')

plt.legend(['Best', 'Random'])

#plt.show()
plt.savefig('girth_min_histogram.png')
#%%
code_ga = LDPC(ga_best_matrix)
convergence_rates_ga = []
bit_errors_ga = []
for p in p_values:
    print(f'Running simulation for p={p} with GA code')
    convergence_rate, bit_error_rate = run_binary_channel_simulation(code_ga, p, num_samples, num_iterations)
    convergence_rates_ga.append(convergence_rate)
    bit_errors_ga.append(bit_error_rate)
    
code_gm = LDPC(gmedian_best_matrix)
convergence_rates_gal = []
bit_errors_gal = []
for p in p_values:
    print(f'Running simulation for p={p} with GM code')
    convergence_rate, bit_error_rate = run_binary_channel_simulation(code_gm, p, num_samples, num_iterations)
    convergence_rates_gal.append(convergence_rate)
    bit_errors_gal.append(bit_error_rate)
    
code_random = LDPC(random_matrix)
convergence_rates_random = []
bit_errors_random = []
for p in p_values:
    print(f'Running simulation for p={p} with random code')
    convergence_rate, bit_error_rate = run_binary_channel_simulation(code_random, p, num_samples, num_iterations)
    convergence_rates_random.append(convergence_rate)
    bit_errors_random.append(bit_error_rate)
    
code_gmin = LDPC(gmin_best_matrix)
convergence_rates_gmin = []
bit_errors_gmin = []
for p in p_values:
    print(f'Running simulation for p={p} with GMIN code')
    convergence_rate, bit_error_rate = run_binary_channel_simulation(code_gmin, p, num_samples, num_iterations)
    convergence_rates_gmin.append(convergence_rate)
    bit_errors_gmin.append(bit_error_rate)    

#%%
# plot the convergence rates
# use something other than colors to distinguish the lines
import matplotlib.pyplot as plt
line_styles = ['-', '--', '-.', ':']

plt.plot(p_values, convergence_rates_ga, label='GA', linestyle=line_styles[0])
plt.plot(p_values, convergence_rates_gal, label='GM', linestyle=line_styles[1])
plt.plot(p_values, convergence_rates_gmin, label='GMIN', linestyle=line_styles[2])
plt.plot(p_values, convergence_rates_random, label='Random', linestyle=line_styles[3])
plt.xlabel('p')
plt.ylabel('Rate der erfolgreichen Dekodierungen')
plt.title('Rate der erfolgreichen Dekodierungen auf dem binären symmetrischen Kanal mit Flip-Wahrscheinlichkeit p')
plt.legend()
#plt.show()
plt.savefig('convergence_rate_binary.png')
#%%
# plot the bit error rates
line_styles = ['-', '--', '-.', ':']

plt.plot(p_values, bit_errors_ga, label='GA', linestyle=line_styles[0])
plt.plot(p_values, bit_errors_gal, label='GM',  linestyle=line_styles[1])
plt.plot(p_values, bit_errors_gmin, label='GMIN', linestyle=line_styles[2])
plt.plot(p_values, bit_errors_random, label='Random', linestyle=line_styles[3])
plt.xlabel('p')
plt.ylabel('Biterror-Rate')
plt.title('Biterro-Rate auf dem binären symmetrischen Kanal mit Flip-Wahrscheinlichkeit p')
plt.legend()
#plt.show()
plt.savefig('biterror_rate_binary.png')
#%%

convergence_rates_ga_gaussian = []
bit_errors_ga_gaussian = []
for snr in snr_values:
    print(f'Running gaussian simulation for p={snr} with GA code')
    code_ga = LDPC(ga_best_matrix)
    std = std_from_signal_to_noise_ration(snr)
    convergence_rate, bit_error_rate = run_gaussian_channel_simulation(code_ga, std, num_samples, num_iterations)
    convergence_rates_ga_gaussian.append(convergence_rate)
    bit_errors_ga_gaussian.append(bit_error_rate)

convergence_rates_gm_gaussian = []
bit_errors_gm_gaussian = []
for snr in snr_values:
    print(f'Running gaussian simulation for p={snr} with GM code')
    code_gm = LDPC(gmedian_best_matrix)
    std = std_from_signal_to_noise_ration(snr)
    convergence_rate, bit_error_rate = run_gaussian_channel_simulation(code_gm, std, num_samples, num_iterations)
    convergence_rates_gm_gaussian.append(convergence_rate)
    bit_errors_gm_gaussian.append(bit_error_rate)
    
convergence_rates_random_gaussian = []
bit_errors_random_gaussian = []
for snr in snr_values:
    print(f'Running gaussian simulation for p={snr} with random code')
    code_random = LDPC(random_matrix)
    std = std_from_signal_to_noise_ration(snr)
    convergence_rate, bit_error_rate = run_gaussian_channel_simulation(code_random, std, num_samples, num_iterations)
    convergence_rates_random_gaussian.append(convergence_rate)
    bit_errors_random_gaussian.append(bit_error_rate)

convergence_rates_gmin_gaussian = []
bit_errors_gmin_gaussian = []
for snr in snr_values:
    print(f'Running gaussian simulation for p={snr} with GMIN code')
    code_gmin = LDPC(gmin_best_matrix)
    std = std_from_signal_to_noise_ration(snr)
    convergence_rate, bit_error_rate = run_gaussian_channel_simulation(code_gmin, std, num_samples, num_iterations)
    convergence_rates_gmin_gaussian.append(convergence_rate)
    bit_errors_gmin_gaussian.append(bit_error_rate)

    
#%%
line_styles = ['-', '--', '-.', ':']

plt.plot(snr_values, bit_errors_ga_gaussian, label='GA', linestyle=line_styles[0])
plt.plot(snr_values, bit_errors_gm_gaussian, label='GM', linestyle=line_styles[1])
plt.plot(snr_values, bit_errors_gmin_gaussian, label='GMIN', linestyle=line_styles[2])
plt.plot(snr_values, bit_errors_random_gaussian, label='Random', linestyle=line_styles[3])
plt.xlabel('snr')
plt.ylabel('Biterror-Rate')
plt.title('Biterror-Rate auf dem Gaußschen Kanal mit SNR')
plt.legend()
#plt.show()

plt.savefig('biterror_gaussian.png')
#%%

line_styles = ['-', '--', '-.', ':']

plt.plot(snr_values, convergence_rates_ga_gaussian, label='GA', linestyle=line_styles[0])
plt.plot(snr_values, convergence_rates_gm_gaussian, label='GM',  linestyle=line_styles[1])
plt.plot(snr_values, convergence_rates_gmin_gaussian, label='GMIN', linestyle=line_styles[2])
plt.plot(snr_values, convergence_rates_random_gaussian, label='Random', linestyle=line_styles[3])
plt.xlabel('snr')
plt.ylabel('Rate der erfolgreichen Dekodierungen')
plt.title('Rate der erfolgreichen Dekodierungen auf dem Gaußschen Kanal mit SNR')
plt.legend()
#plt.show()
plt.savefig('convergence_rate_gaussian.png')
#%%
