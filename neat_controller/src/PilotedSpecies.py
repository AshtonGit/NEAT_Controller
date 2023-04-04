from itertools import count

from neat.math_util import mean, stdev
from neat.six_util import iteritems, iterkeys, itervalues
from neat.config import ConfigParameter, DefaultClassConfig
from neat.species import GenomeDistanceCache, Species


class PilotedSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config, generation, reporters):
        # pylint: disable=super-init-not-called
        self.members = None
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}

        self.created = generation
        self.last_improved = generation
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, members):
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in itervalues(self.members)]

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    " TO DO: Add body genome speciation to the algo alongside behavior genome when speciating."
    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """

        "assert isinstance(population, dict)"
        genomes = {}

        for key, genomewrap in population.items():
            genomes[key] = genomewrap

        compatibility_threshold = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        unspeciated = set(iterkeys(genomes))
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in iteritems(self.species):
            candidates = []
            for gid in unspeciated:
                g = genomes[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition behavior_genomes into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = genomes[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in iteritems(new_representatives):
                rep = genomes[rid]
                d = distances(rep.behaviorGenome, g.behaviorGenome)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
                genomes[gid].species = sid
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]
                genomes[gid].species = sid

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in iteritems(new_representatives):
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, genomes[gid]) for gid in members)
            s.update(genomes[rid], member_dict)

        gdmean = mean(itervalues(distances.distances))
        gdstdev = stdev(itervalues(distances.distances))
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
