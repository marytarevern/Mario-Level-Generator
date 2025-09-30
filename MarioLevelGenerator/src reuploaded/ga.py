import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    #"|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.8,
            pathPercentage=0.4,
            emptyPercentage=0.6,
            linearity= 1.0,
            leniency= 0.0,
            solvability=2.0,
            meaningfulJumps=3,
            jumps=4,
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        left = 1
        right = width - 2
                #  -     X    ?    M    B   o     T   E
        weights = [0.8,0.0,0.015,0.01,0.03,0.04,0.005,0.05]
        pipetop = False
        pipemid = False
        fix_pipe = False
        for y in range(height):
            if y < 3:
                continue
            for x in range(left, right):
                if y >= 13 and x <= 5:
                    continue
                if y < height - 1: 
                    if genome[y][x] == "T" and genome[y+1][x] == "|":
                        genome[y+1][x] = "T"
                if x > 1:
                    if genome[y][x-1] == "T" or genome[y][x-1] == "|":
                        continue
                if random.random() <= 0.05:
                    #bottom row is only solid blocks or occasional pits
                    if x <= 5 and y != (height - 1):
                        weights = [0.8,0.0,0.015,0.01,0.3,0.04,0.0,0.05]
                    elif y != (height - 1):
                        weights = [0.8,0.0,0.015,0.01,0.3,0.04,0.005,0.05]
                    else:
                        weights = [0.05,0.95,0.0,0.0,0.0,0.0,0.0,0.0]
                    genome[y][x] = random.choices(options, weights=weights)[0]

                    #don't build pipes right next to each other
                    if genome[y][x] == "T":
                        skip = False
                        for i in range(height):
                            if genome[i][x-1] == "T" or genome[i][x-1] == "|":
                                skip = True
                                break
                            if genome[i][x+1] == "T" or genome[i][x+1] == "|":
                                skip = True
                                break
                        if skip:
                            genome[y][x] = "-"
                            continue

                #fix pipes if place "T"
                if genome[y][x] == "T":
                    temp_y = y + 1
                    counter = 0
                    while (temp_y < height - 1) and (genome[y][x] != "-") and counter < 3:
                        genome[temp_y][x] = "|"
                        temp_y += 1
                        counter += 1
                    if temp_y < height:
                        if genome[temp_y][x] != "X" or genome[temp_y][x+1] != "X":
                            genome[temp_y][x] = "X"
                            genome[temp_y][x+1] = "X"
                #dont put enemy if not on ground
                if genome[y][x] == "E":
                    if genome[y+1][x] == "o" or genome[y+1][x] == "-" or genome[y+1][x] == "E":
                        genome[y][x] = "-"
                if genome[y][x] == "o" or genome[y][x] == "-":
                    if genome[y-1][x] == "E":
                        genome[y][x] = "B"
                #fix pipes
                if genome[y][x] == "-" or genome[y][x] == "E" or genome[y][x] == "o":
                    if genome[y-1][x] == "|" or genome[y-1][x] == "T":
                        genome[y-1][x] = "?"
                        genome[y-1][x+1] = "?"
                    if y < height - 1:
                        if genome[y+1][x] == "|":
                            genome[y+1][x] = "T"
                if genome[y][x] != "|" and genome[y][x] != "T":
                    if genome[y-1][x] == "|" or genome[y-1][x-1] == "|":
                        genome[y][x] = "X"
                    
                
        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 0
        right = width - 1

        #multiple point crossover
        quarter = math.floor(width/4)
        half = math.floor(width/2)

        #parent a (self)
        a1 = [row[left:quarter] for row in self.genome]
        a2 = [row[left + quarter:half] for row in self.genome]
        a3 = [row[half:quarter*3] for row in self.genome]
        a4 = [row[left + quarter*3:width] for row in self.genome]

        #parent b (other)
        b1 = [row[left:quarter] for row in other.genome]
        b2 = [row[left + quarter:half] for row in other.genome]
        b3 = [row[half:quarter*3] for row in other.genome]
        b4 = [row[left + quarter*3:width] for row in other.genome]

        
        c1 = [a+b+c+d for a, b, c, d, in zip(a1, b2, a3, b4)]
        c2 = [a+b+c+d for a, b, c, d, in zip(b1, a2, b3, a4)]
        c3 = [a+b+c+d for a, b, c, d, in zip(a1, a2, b3, b4)]
        c4 = [a+b+c+d for a, b, c, d, in zip(b1, b2, a3, a4)]

        # do mutation; note we're returning a one-element tuple here
        return (Individual_Grid(self.mutate(c1))), (Individual_Grid(self.mutate(c2))), (Individual_Grid(self.mutate(c3))), (Individual_Grid(self.mutate(c4)))

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.7,
            linearity=0.8,
            leniency= 1.0,
            solvability=2.0,
            meaningfulJumps=4,
            jumps=8,
        )
        '''meaningfulJumpVariance=0.5,
        negativeSpace=0.6,
        pathPercentage=0.5,
        emptyPercentage=0.6,
        linearity=-0.5,
        solvability=2.0'''
            
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 2
        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        #print("in mutate children DE")
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.1 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    h = offset_by_upto(h, 8, min=1, max=height - 4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):
        #print("in generate children DE")
        # STUDENT How does this work?  Explain it in your writeup.
        #print(self.genome, flush=True)
        #print(len(self.genome), flush=True)
        
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []
        ga = a_part + b_part
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        # do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))

    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = [(100, '3_coin', 4)]
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_Grid
#Individual = Individual_DE


def generate_successors(population):
    results = []
    #print("in generate successors")

    def truncation_selection():
        sorted_population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)


        truncation_ratio = 0.2
        num_parents = int(len(sorted_population) * truncation_ratio)
        num_parents = max(2, num_parents)
        parent_pool = sorted_population[:num_parents]


        while len(results) < len(population):

            parent1 = random.choice(parent_pool)
            parent2 = random.choice(parent_pool)
            children = Individual.generate_children(parent1, parent2)

            for child in children:
                results.append(child)
                if len(results) >= len(population):
                    break
            #print(len(results))

        return results

    #set up tournament selection
    def tournament_selection(population, tournaments, best_indiv_chance):
        if tournaments > len(population):
            tournaments = len(population)
        tournament_size = max(math.floor(len(population)/tournaments), 1)
        tournament_players = []
        parents = []
        start = 0
        end = 0
        for i in range(tournaments):
            end += tournament_size
            tournament_players.append(population[start:end])
            start += tournament_size
        #population = sorted(population, key=lambda x: x._fitness, reverse=True)

        for i in range(tournaments):
            tournament_players[i] = sorted(tournament_players[i], key=lambda x: x._fitness, reverse=True)
            for j in range(tournament_size):
                chance = best_indiv_chance * (1 - best_indiv_chance)**i
                rand_num = random.random()
                if (chance < rand_num) or (j >= len(tournament_players[i]) - 1):
                    #pick this one
                    parents.append(tournament_players[i][j])
                    break
        return parents

    max_tournaments = min(len(population), 30)
    #use tournament selection
    parents = tournament_selection(population, max_tournaments, 0.3)

    #select 2 random parents at a time
    #print("finished tournament")
    while len(parents) >= 2:
        couple = random.sample(parents, k=2)
        #print(couple)
        for child in Individual.generate_children(couple[0], couple[1]):
            results.append(child)
        parents.remove(couple[0])
        parents.remove(couple[1])

    #comment out one or the other to change which selection function
    return results                  #tournament
    #return truncation_selection()   #truncation


def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    max_generations = 20
    fitness_threshold = 700
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("-" + "".join(row) + "--"+ "\n")
                generation += 1
                # STUDENT Determine stopping condition
                best = max(population, key=Individual.fitness)
                stop_condition = (generation >= max_generations) or (best.fitness() >= fitness_threshold)
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pool.terminate()
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("-" + "".join(row) + "--"+ "\n")
