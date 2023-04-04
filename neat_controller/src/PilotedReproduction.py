from __future__ import division

import math
import random
from itertools import count

import neat.nn
from neat.config import ConfigParameter, DefaultClassConfig
from numpy import mean

from PilotedFeedForward import FeedForwardNetwork as PilotedFeedForwardNetwork;


class PilotedReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)
        self.ancestors = {}

    @staticmethod
    def parse_config(param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            network = PilotedFeedForwardNetwork.create(g, genome_config)
            # Custom for all species names to be in Old English
            genome = GenomeInfo(key, g, g, network, species=0, fitness=0)
            new_genomes[key] = genome
            self.ancestors[key] = tuple()
        return new_genomes

    def reproduceCrossover(self, config, parent1, parent2):
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents. Genomes with the lowest fitness scores
        are eliminated. Remaining individuals create the next generation of genomes.
        """
        # Note that if the parents are not distinct, crossover will produce a
        # genetically identical clone of the parent (but with a different ID).
        gid = next(self.genome_indexer)
        if parent1.behaviorGenome:
            b_genome = config.genome_type(gid)
            b_genome.configure_crossover(parent1.behaviorGenome, parent2.behaviorGenome, config.genome_config)
            b_genome.mutate(config.genome_config)

        if parent1.structure_genome:
            s_genome = config.genome_type(gid)
            s_genome.configure_crossover(parent1.bodyGenome, parent2.bodyGenome, config.genome_config)
            s_genome.mutate(config.genome_config)

        gdata = GenomeInfo(gid, b_genome, s_genome, "")
        self.ancestors[gid] = (parent1, parent2)
        return gdata

    @staticmethod
    def compute_spawn(adjusted_fitness, previous_sizes, pop_size, min_species_size):
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum = sum(adjusted_fitness)

        spawn_amounts = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d = (s - ps) * 0.5
            c = int(round(d))
            spawn = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn = sum(spawn_amounts)
        norm = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    """Given a population of genomes of multiple different species, return a new population of their offspring. 
        
        species = {speciesKey: {genomeKey, fitness, adjustedfitness} }
     """

    def reproducePopulation(self, config, species, pop_size, generation, generationNumber):
        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.
        """
        :param config:
        :param species:  All species at start of this current generation
        :param pop_size: Final number of genomes after reproduction
        :param generation: Genomes to be reproduces
        :param generationNumber: Counter for number of generations that have existed
        :return:
        """

        """
        1. Make list of all fitnesses in the population sso we can find min and max fitness
        2. Update species class for all species to have new fitnesses for each of its members
        3. Rest of function should work as normal
        
        1. 
        """
        print("DEBUG Reproduce Population =====")
        all_fitnesses = [g.fitness for g in generation.values() if g.fitness > 0]
        print("Number of genome fitness > 0: "+str(len(all_fitnesses)))
        if len(all_fitnesses) == 0:
            all_fitnesses = [0]
        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        print("Highest fitness : "+str(max_fitness))
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        fitness_range = max(1.0, max_fitness - min_fitness)
        """species:(
                m:( key: genome{fitness:, key, connections, nodes})
                ) 
            """
        for afs in species.species.values():
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in (afs.members.values())])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in species.species.values()]
        avg_adjusted_fitness = mean(adjusted_fitnesses)  # type: float
        print("avg adjusted fitness: "+str(avg_adjusted_fitness))
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes = [len(s.members) for s in species.species.values()]
        min_species_size = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                           pop_size, min_species_size)
        print("spawn amounts: "+str(spawn_amounts))
        new_population = {}
        old_species = species.species
        species.species = {}
        for spawn, s in zip(spawn_amounts, old_species.values()):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            repro_cutoff = int(math.ceil(self.reproduction_config.survival_threshold *
                                         len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff = max(repro_cutoff, 2)
            old_members = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1.behaviorGenome, parent2.behaviorGenome, config.genome_config)
                child.mutate(config.genome_config)
                # TO DO
                # Add create a GenomeInfo wrapper to hold genome in
                network = PilotedFeedForwardNetwork.create(child, config.genome_config)
                genomeInfo = GenomeInfo(key=gid, behaviorGenome=child, bodyGenome=child, network=network, species=None)
                new_population[gid] = genomeInfo
                self.ancestors[gid] = (parent1_id, parent2_id)
        print("New population size after reproduction function ends: "+str(len(new_population.items())))
        print(str(new_population.keys()))
        #New Genomes have their species identified after creation
        species.speciate(config, new_population, generationNumber)
        return new_population


class GenomeInfo:

    def __init__(self, key, behaviorGenome, bodyGenome, network, species, fitness=0):
        self.key = key
        self.behaviorGenome = behaviorGenome
        self.bodyGenome = bodyGenome
        self.species = species
        self.network = network
        self.fitness = fitness

    def to_dict(self):
        d = {"key": self.key,
             "behaviorGenome": self.behaviorGenome,
             "bodyGenome": self.bodyGenome,
             "species": self.species}
        return d

    def feedforward(self, inputs):
        output = self.network.activate(inputs)
        return output

    def distance(self, other, config):
        return self.behaviorGenome.distance(other.behaviorGenome, config)

    def setFitness(self, fitness):
        self.fitness = fitness

    def getFitness(self):
        return self.fitness
