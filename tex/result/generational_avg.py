import argparse
import math

import numpy as np


def load_data(filepath):
    data = []
    rows = 0
    count = 0
    with open(filepath, "r") as infile:
        for row in infile:
            if row.strip() == "== generation":
                rows = count
                count = 0
                continue
            count += 1
            vals = row.strip().split(",")
            data.append(list(map(lambda x: float(x), vals[:3])))
    print("rows: %d" % rows)
    return data[len(data) - rows:]

def build_generations(data, n_generations=20, population_size=10):
    gens = {}
    for row in data:
        if row[0] not in gens.keys():
            gens[row[0]] = []
        gens[row[0]].append(row[1:])

    cur_pop = gens[0]
    final_pop = [[0] + x for x in gens[0]]
    for i in range(1, n_generations):
        new_pop = gens[i]
        combined = sorted(cur_pop, key=lambda x: x[1],  reverse=True)[:population_size] + new_pop
        final_pop.extend([[i] + x for x in combined])
        cur_pop = combined
    return final_pop

def generational_stats(data, n_generations):

    averages = {}
    std_devs = {}
    pop_size = len(data[0])
    pops = {i:[] for i in range(n_generations)}
    for r in data:
        pops[r[0]].append(r[2])

    for (k, individuals) in pops.items():
        averages[k] = np.average(individuals[:pop_size])
        std_devs[k] = np.std(individuals[:pop_size], ddof=1)

    return averages, std_devs

def write_to_files(filepath, averages, std_devs, n_generations):
    with open(filepath, "w") as ofile:
        for i in range(n_generations):
            avg = averages[i]
            low = avg - std_devs[i]
            high = avg + std_devs[i]
            ofile.write("{},{:.4f},{:.4f},{:.4f}\n".format(i, avg, low, high))


def gen_samples(avgs, stds, n_samples, filepath):
    samples = []
    with open(filepath, "w") as ofile:
        for i in range(n_samples):
            for (gen, avg) in avgs.items():
                std = stds[gen]
                ofile.write("{}\t{:.4f}\n".format(gen, np.random.normal(avg, std)))
            ofile.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="Path to data.")
    parser.add_argument("--outpath", help="Path to save data.")
    parser.add_argument("--n_generations", type=int, help="Number of generations in data.")
    args = parser.parse_args()
    data = load_data(args.datapath)
    gens = build_generations(data)
    
    avgs, stds = generational_stats(gens, args.n_generations)
    for (gen, avg) in avgs.items():
        print("gen: %d avg: %f +/- %f" % (gen, avg, stds[gen]))

    # gen_samples(avgs, stds, 100, args.outpath)
    write_to_files(args.outpath, avgs, stds, args.n_generations)
    
if __name__ == "__main__":
    main()

