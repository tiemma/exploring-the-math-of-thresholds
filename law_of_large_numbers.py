from itertools import product
import concurrent.futures
import time

from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy

percentiles = [90, 95, 99, 99.9, 99.99, 99.999]
headers = ["w", "x", "y", "z", "n", "Availability"]
std_dev_headers = ["percentile", "std_dev"]
percentile_headers = ["n"] + list(map(lambda x: f"p{x}", percentiles))
avg_percentiles_sum_data = [0 for _ in range(len(percentiles))]
std_dev_percentiles_data = [list() for _ in range(len(percentiles))]
all_avg_percentiles_data = []
show_table_data = False

number_of_jobs = 250
x_datapoints = range(1, number_of_jobs + 1)
pool = concurrent.futures.ProcessPoolExecutor(max_workers=16)
perms = [list() for _ in range(len(x_datapoints) + 1)]


def alignment(size):
    return tuple(["left"] * size)


def time_run(func, label, i, use_num=False):
    start = time.time()
    if use_num:
        resp = func(i)
    else:
        resp = func()
    end = time.time()
    print(f"Iteration {i} - {label} took {end - start} seconds to run")

    return resp


def gen_perms(x):
    results = []
    for perm in product(list(range(x + 1)), repeat=4):
        if sum(perm) == x:
            results.append(perm)
    return x, results


def future_run(x):
    return time_run(gen_perms, "gen_perms", x, True)


with pool as executor:
    for x, result in executor.map(future_run, x_datapoints):
        perms[x - 1] = result

for i in x_datapoints:
    table_data = []
    availability_data = []


    def generate_series():
        for perm in perms[i - 1]:
            w, x, y, z = perm
            availability = 100 * (x / 3 + 2 * y / 3 + z) / i
            if show_table_data:
                table_data.append([w, x, y, z, i, availability])
            availability_data.append(availability)


    time_run(generate_series, "generate_series", i)


    def calculate_percentiles():
        for idx, percentile in enumerate(percentiles):
            percentile_dp = round(numpy.percentile(availability_data, percentile), 4)
            std_dev_percentiles_data[idx].append(percentile_dp)
            avg_percentiles_sum_data[idx] = avg_percentiles_sum_data[idx] + percentile_dp


    time_run(calculate_percentiles, "calculate_percentiles", i)

    if show_table_data:
        print(tabulate(table_data[::-1], headers=headers, tablefmt='github', colalign=alignment(len(headers))))

    all_avg_percentiles_data = all_avg_percentiles_data + [[f"avg-{i}"] + list(
        map(lambda x: x / i, avg_percentiles_sum_data))]

std_percentiles = []
for idx, percentile in enumerate(percentiles):
    std_percentiles.append([percentile, numpy.std(std_dev_percentiles_data[idx])])
print(tabulate(std_percentiles, headers=std_dev_headers,
               tablefmt='github', colalign=alignment(len(std_dev_headers))))

fig, ax = plt.subplots()

for idx, percentile in enumerate(percentiles):
    percentile_label = f"p{percentile}"
    ax.plot(x_datapoints, list(map(lambda x: x[idx + 1], all_avg_percentiles_data)), label=percentile_label)
    last_value = all_avg_percentiles_data[-1][idx + 1]
    ax.axhline(y=last_value, label=f"{percentile_label} threshold at {last_value}", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Availability")
    ax.set_title('Law of Large Numbers')
    ax.legend(loc='lower left', prop={'size': 7}, borderpad=0.5, labelspacing=0)

print(tabulate(all_avg_percentiles_data, headers=percentile_headers,
               tablefmt='github', colalign=alignment(len(percentile_headers))))

plt.grid(True)
plt.autoscale()
plt.tight_layout()
fig.savefig(f"./law_of_large_numbers_{number_of_jobs}.png", bbox_inches="tight")
plt.show()
