import gzip
import pickle
import random
import socket
import json
import time
from _thread import *

import neat
from PilotedSpecies import PilotedSpeciesSet

from PilotedPopulation import PilotedPopulation
from PilotedReproduction import PilotedReproduction
from neat import config as conf
from _thread import *
import threading

""" 
# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"]) 

"""


class NeatManager:

    def __init__(self):
        self.config = conf.Config(neat.DefaultGenome, PilotedReproduction, PilotedSpeciesSet,
                                  neat.DefaultStagnation,
                                  "./venv/config")
        self.pop = PilotedPopulation(self.config)
        self.reproduction = self.pop.reproduction

    def SetConfig(self, newConfigFilePath):
        self.config = conf.Config(neat.DefaultGenome, PilotedReproduction, PilotedSpeciesSet, neat.DefaultStagnation,
                                  newConfigFilePath)

    def create_population(self):
        self.pop = PilotedPopulation(self.config)
        self.reproduction = self.pop.reproduction
        return self.get_population()

    def get_genomeInfo(self, key):
        return self.pop.population[key]

    def get_generation(self):
        return self.pop.generation;

    def get_population(self):
        # Return id's and species for all genomes in the current population
        # TODO: make genomes json serializable so that they may be sent over network.
        population = {}
        for key, item in self.pop.population.items():

            if not item.species:
                population[key] = -1
            else:
                population[key] = item.species

        print("debug get current population " + str(population))
        return population

    def crossover(self, key1, key2):
        genome1, genome2 = self.get_genomeInfo(key1), self.get_genomeInfo(key2)
        child = self.reproduction.crossover(self.config, genome1, genome2)
        return {child.key: child.species}

    """
    Creates a new population using genomes given. Delete all genomes from species and population
    that were not passed as argument as they were not fit enough to reproduce.
    """

    def reproduceFittestRemoveRest(self, fittestGenomes):
        self.pop.population = self.pop.reproduceFittestRemoveRest(fittestGenomes)
        msg = self.get_population()
        return msg

    def feedforward(self, genomeKey, inputs):
        return self.get_genomeInfo(genomeKey).feedforward(inputs)

    def threaded_connection_handler(self, ServerSocket):
        Client, address = ServerSocket.accept()
        print(f'Connected to: {address[0]}:{str(address[1])}')
        start_new_thread(self.client_handler, (Client,))
        while True:
            # establish connection with client

            # lock acquired by client
            print_lock.acquire()

            # Start a new thread and return its identifier
            start_new_thread(self.client_handler, (Client,))



    def save_population(self, filename):
        print("Saving population to file " + filename)
        try:
            with gzip.open(filename, 'w', compresslevel=5) as f:
                data = (self.pop.generation, self.config, self.pop.population, self.pop.species, random.getstate())
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except error:
            print(error)
            return False

        return True

    def load_population(self, filename):
        print("Loading population from file " + filename)
        try:
            with gzip.open(filename) as f:
                generation, config, population, species_set, rndstate = pickle.load(f)
                random.setstate(rndstate)
                self.pop = PilotedPopulation(config, (population, species_set, generation))

        except error:
            print(error)
            return None

        return self.get_population()

    def client_handler(self, connection):
        while True:
            try:
                data = connection.recv(2048)
                """ 
                Split message into header and json to understand request
                a. Feedforward - data is the genomeid and network inputs. Return network outputs
                 after putting inputs through network

                b. crossover - data is genomeid of two genomes. 
                        i. create new genome using combining genomes of both parents
                        ii.  return genome and genome id of new genome if was breeding successfull.
                             else return -1

                """
                if data:
                    message = data.decode('utf-8')
                    print(message)
                    decoded = message.split('$')
                    command = decoded[0]
                    print("Command Received:", command)
                    if command == "create population":
                        reply = self.create_population()
                    elif command == "get population":
                        reply = self.get_population()
                    elif command == "crossover":
                        key1 = json.loads(decoded[1])
                        key2 = json.loads(decoded[2])
                        reply = self.crossover(key1, key2)
                    elif command == "reproduce population":
                        fittestGenomes = json.loads(decoded[1], object_hook=jsonKeys2int)
                        reply = self.reproduceFittestRemoveRest(fittestGenomes)
                    elif command == "feedforward":
                        key = json.loads(decoded[1])
                        networkInputs = json.loads(decoded[2])
                        reply = self.feedforward(key, networkInputs)
                        print("Feedforward outputs: " + str(reply))
                    elif command == "save population":
                        filename = decoded[1]
                        reply = self.save_population(filename)
                    elif command == "load population":
                        filename = decoded[1]
                        reply = self.load_population(filename)
                    elif command == "get generation":
                        reply = self.get_generation()
                    else:
                        reply = "No valid command given"

                    json_reply = json.dumps(reply)
                    encoded_msg = json_reply.encode()
                    print("encoded message: " + str(encoded_msg))
                    connection.sendall(encoded_msg)
                    print("message sent")

            except Exception as err:
                print(err)
                print(err.args)
                connection.close()
                return
            connection.close()
            return

    def accept_connections(self, ServerSocket):
        Client, address = ServerSocket.accept()
        print(f'Connected to: {address[0]}:{str(address[1])}')
        start_new_thread(self.client_handler, (Client,))



print_lock = threading.Lock()
