from __future__ import print_function


from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
import uuid


# This population only manipulates population when requested to by an outside program.
# Hence, the name "Piloted" as i thas to be piloted by another program.
class PilotedPopulation(object):

    def __init__(self, config,  initial_state=None):
        self.config = config
        self.reporters = ReporterSet()
        self.reproduction = config.reproduction_type(config.reproduction_config, self.reporters)

        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.species = config.species_set_type(config.species_set_config, self.population, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
        else:
            self.population, self.species, self.generation = initial_state

        self.best_genome = None

    def getAllGenome(self):
        return self.population

    def getSingleGenome(self, gid):
        return self.population[gid]

    def eliminateGenome(self, gid):
        del(self.population[gid])

    """
        Creates a new population using genomes given. Delete all genomes from species and population
        that were not passed as argument as they were not fit enough to reproduce.
        """
    def reproduceFittestRemoveRest(self, fittestGenomes):
        # Update fitness of all the genomes in population and in species
        # As genomes not in fittestGenomes will have fiobject_hook=jsonKeys2intobject_hook=jsonKeys2inttness of 0, they will likely not breed.
        for k, v in fittestGenomes.items():
            self.population[k].fitness = v
            self.population[k].behaviorGenome.fitness = v
            #check that the above update also affects the referenced gneome in species
            s = self.species.get_species(k)

        newPop = self.reproduction.reproducePopulation(self.config, self.species, self.config.pop_size, self.population, self.generation)
        self.population = newPop
        self.generation += 1
        return self.population




    # return a read only container class holding a copy of all data for that genome including id, resources and
    # structure
    def crossoverReproduction(self, genomeA, genomeB):
        newGenome = self.reproduction.singleSexualReproduction(genomeA, genomeB)
        return newGenome

    # return a read only container class holding a copy of all data for that genome including id, resources and
    # structure
    def cloneReproduction(self, genome):
        newGenome = self.reproduction.cloneReprodcution(genome)
        return newGenome

    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)
