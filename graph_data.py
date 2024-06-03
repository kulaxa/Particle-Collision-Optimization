import matplotlib.pyplot as plt
import os

# Function to parse the data from a line
def parse_line(line):
    values = line.strip().split(';')
    execution_time = float(values[0].split(':')[1])
    particle_counter = float(values[1].split(':')[1])
    using_gpu = int(values[2].split(':')[1])
    using_cpu_optimized = int(values[3].split(':')[1])
    thread_count = int(values[4].split(':')[1])
    grid_size = int(values[5].split(':')[1])
    return execution_time, particle_counter, using_gpu, using_cpu_optimized, thread_count, grid_size

# Read data from the file
# file_path = 'results_1.txt'
# file_paths = ['results_unoptimized_1_thread.txt', 'results_unoptimized_6_thread.txt',
#               'results_optimized_1_thread_100.txt', 'results_optimized_6_thread_100.txt',
#               'results_optimized_1_thread_200.txt', 'results_optimized_6_thread_200.txt']

file_paths = [
    'results_pc_better/results_optimized_1_300.txt', 'results_pc_better/results_optimized_2_300.txt',
    'results_pc_better/results_optimized_4_300.txt', 'results_pc_better/results_optimized_6_300.txt',
    'results_pc_better/results_optimized_8_300.txt', 'results_pc_better/results_optimized_10_300.txt',
    'results_pc_better/results_optimized_12_300.txt', 'results_pc_better/results_optimized_14_300.txt'
]

os.makedirs("graphs", exist_ok=True)
counter = 0
for file_path in file_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse data and create lists for FPS and PARTICLE_COUNTER
    execution_times = []
    particle_counter_values = []

    for line in lines:
        execution_time, particle_counter, using_gpu, using_cpu_optimized, thread_count, grid_size = parse_line(line)
        if particle_counter != 0:
            execution_times.append(execution_time)
            particle_counter_values.append(particle_counter)
    label = ""
    if using_gpu:
        label += "GPU"
    elif using_cpu_optimized:
        label += f"CPU Optimized ({thread_count} threads, grid size {grid_size})"
    else:
        label += f"CPU Unoptimized ({thread_count} threads)"
    # Create and save the plot
    plt.plot(particle_counter_values, execution_times, marker='.', markersize='1', label=label)
    counter += 1
title = "Execution Times vs Particle Count"
plt.title(title)
plt.xlabel('Number of Particles')
plt.ylabel('Execution Time (ms)')
plt.ylim(0, 70)  # Set y-axis limits
plt.xlim(0, 350000)  # Set x-axis limits
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=2, fontsize="8")

plt.subplots_adjust(bottom=0.18)


plt.grid(True)
plt.savefig(f'graphs/fps_vs_particle_counter_plot_gpu{using_gpu}_optimized_cpu{using_cpu_optimized}_{thread_count}_grid_size{grid_size}.png')  # Save as PNG file
