import logging
import os
import signal
import time
import traceback
from multiprocessing import Event

import requests
import torch
import torch.utils.data
from torch.autograd import Variable

from distribution.node_client import NodeClient
from helpers.configuration_container import ConfigurationContainer
from helpers.db_logger import DbLogger
from helpers.heartbeat import Heartbeat
from helpers.math_helpers import is_square
from helpers.network_helpers import get_network_devices
from helpers.reproducible_helpers import set_random_seed
from training.mixture.mixed_generator_dataset import MixedGeneratorDataset
from training.mixture.score_factory import ScoreCalculatorFactory

GENERATOR_PREFIX = 'generator-'
DISCRIMINATOR_PREFIX = 'discriminator-'


class LipizzanerMaster:
    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.cc = ConfigurationContainer.instance()
        self.heartbeat_event = None
        self.heartbeat_thread = None
        self.experiment_id = None

    def run(self):
        if os.environ.get('DOCKER', False) == 'True':
            self._logger.info('Detected Docker environment, enforcing auto-discover.')
            self.cc.settings['general']['distribution']['auto_discover'] = True

        if self.cc.settings['general']['distribution']['auto_discover']:
            self._logger.info('Running in auto-discover mode. Detecting clients...')
            clients = self._load_available_clients()
            self.cc.settings['general']['distribution']['client_nodes'] = clients
            self._logger.info('Detected {} clients ({})'.format(len(clients), clients))
        else:
            # Expand port ranges to multiple client entries
            self.expand_clients()
            clients = self.cc.settings['general']['distribution']['client_nodes']
        accessible_clients = self._accessible_clients(clients)

        if len(accessible_clients) == 0 or not is_square(len(accessible_clients)):
            self._logger.critical('{} clients found, but Lipizzaner currently only supports square grids.'
                                  .format(len(accessible_clients)))
            self._terminate(stop_clients=False)

        ### THIS WAS NOT COMMENTED BEFORE
        # if len(accessible_clients) != len(clients):
        #     non_accessible = set([c['address'] for c in accessible_clients]) & \
        #                      set([c['address'] for c in clients])
        #     self._logger.critical('Client with address {} is either busy or not accessible.'.format(non_accessible))
        #     self._terminate(stop_clients=False)

        # It is not possible to obtain reproducible result for large grid due to nature of asynchronous training
        # But still set seed here to minimize variance
        set_random_seed(self.cc.settings['general']['seed'],
                        self.cc.settings['trainer']['params']['score']['cuda'])
        self._logger.info("Seed used in master: {}".format(self.cc.settings['general']['seed']))

        self.heartbeat_event = Event()
        self.heartbeat_thread = Heartbeat(self.heartbeat_event,
                                          self.cc.settings['general']['distribution']['master_node'][
                                              'exit_clients_on_disconnect'])

        signal.signal(signal.SIGINT, self._sigint)
        self._start_experiments()
        self.heartbeat_thread.start()

        self.heartbeat_thread.join()

        # When this is reached, the heartbeat thread has stopped.
        # This either happens when the experiments are done, or if they were terminated
        if self.heartbeat_thread.success:
            self._gather_results()
            self._terminate(stop_clients=False, return_code=0)
        else:
            self._terminate(stop_clients=False, return_code=-1)

    def _sigint(self, signal, frame):
        self._terminate()

    def _accessible_clients(self, clients):
        accessible_clients = []
        for client in clients:
            assert client['address'] is not None
            address = 'http://{}:{}/status'.format(client['address'], client['port'])
            try:
                resp = requests.get(address)
                assert resp.status_code == 200
                assert not resp.json()['busy']
                accessible_clients.append(client)
            except Exception:
                pass

        return accessible_clients

    def _load_available_clients(self):
        ip_addresses = get_network_devices()
        possible_clients = []
        for ip in ip_addresses:
            possible_clients.append({'address': ip, 'port': 5000})

        accessible_clients = sorted(self._accessible_clients(possible_clients), key=lambda x: x['address'])
        # Docker swarm specific: lowest address is overlay network address, remove it
        if os.environ.get('SWARM', False) == 'True' and len(accessible_clients) != 0:
            del accessible_clients[0]

        return accessible_clients

    def _start_experiments(self):
        self.cc.settings['general']['distribution']['start_time'] = time.strftime('%Y-%m-%d_%H-%M-%S')

        # If DB logging is enabled, create a new experiment and attach its ID to settings for clients
        db_logger = DbLogger()
        if db_logger.is_enabled:
            self.experiment_id = db_logger.create_experiment(self.cc.settings)
            self.cc.settings['general']['logging']['experiment_id'] = self.experiment_id


        for client_id, client in enumerate(self.cc.settings['general']['distribution']['client_nodes']):
            address = 'http://{}:{}/experiments'.format(client['address'], client['port'])
            self.cc.settings['general']['distribution']['client_id'] = client_id
            try:
                resp = requests.post(address, json=self.cc.settings)
                assert resp.status_code == 200, resp.text
                self._logger.info('Successfully started experiment on {}'.format(address))
            except AssertionError as err:
                self._logger.critical('Could not start experiment on {}: {}'.format(address, err))
                self._terminate()

    def _terminate(self, stop_clients=True, return_code=-1):
        try:
            if self.heartbeat_thread:
                self._logger.info('Stopping heartbeat...')
                self.heartbeat_thread.stopped.set()
                self.heartbeat_thread.join()

            if stop_clients:
                self._logger.info('Stopping clients...')
                node_client = NodeClient(None)
                node_client.stop_running_experiments()
        finally:
            db_logger = DbLogger()
            if db_logger.is_enabled and self.experiment_id is not None:
                db_logger.finish_experiment(self.experiment_id)

            exit(return_code)

    def _gather_results(self):
        self._logger.info("Collecting results from clients...")

        # Initialize node client
        dataloader = self.cc.create_instance(self.cc.settings["dataloader"]["dataset_name"])
        network_factory = self.cc.create_instance(self.cc.settings["network"]["name"], dataloader.n_input_neurons)
        node_client = NodeClient(network_factory)
        db_logger = DbLogger()

        results = node_client.gather_results(self.cc.settings["general"]["distribution"]["client_nodes"], 120)

        scores = []
        best_score = None
        generators_to_be_optimized = None
        weights_generator_to_be_optimized = None
        best_node = None
        for (node, generator_pop, discriminator_pop, weights_generator, weights_discriminator) in results:
            node_name = "{}:{}".format(node["address"], node["port"])
            try:
                output_dir = self.get_and_create_output_dir(node)

                for generator in generator_pop.individuals:
                    source = generator.source.replace(":", "-")
                    filename = "{}{}.pkl".format(GENERATOR_PREFIX, source)
                    torch.save(
                        generator.genome.net.state_dict(), os.path.join(output_dir, "generator-{}.pkl".format(source))
                    )

                    with open(os.path.join(output_dir, "mixture.yml"), "a") as file:
                        file.write("{}: {}\n".format(filename, weights_generator[generator.source]))

                for discriminator in discriminator_pop.individuals:
                    source = discriminator.source.replace(":", "-")
                    filename = "{}{}.pkl".format(DISCRIMINATOR_PREFIX, source)
                    torch.save(discriminator.genome.net.state_dict(), os.path.join(output_dir, filename))

                # Save images
                dataset = MixedGeneratorDataset(
                    generator_pop,
                    weights_generator,
                    self.cc.settings["master"]["score_sample_size"],
                    self.cc.settings["trainer"]["mixture_generator_samples_mode"],
                )
                image_paths = self.save_samples(dataset, output_dir, dataloader)
                self._logger.info(
                    "Saved mixture result images of client {} to target directory {}.".format(node_name, output_dir)
                )

                # Calculate inception or FID score
                score = float("-inf")
                if self.cc.settings["master"]["calculate_score"]:
                    calc = ScoreCalculatorFactory.create()
                    self._logger.info("Score calculator: {}".format(type(calc).__name__))
                    self._logger.info(
                        "Calculating score score of {}. Depending on the type, this may take very long.".format(
                            node_name
                        )
                    )

                    score = calc.calculate(dataset)
                    self._logger.info(
                        "Node {} with weights {} yielded a score of {}".format(node_name, weights_generator, score)
                    )
                    scores.append((node, score))

                    if (
                        (best_score is None)
                        or (best_score > score[0] and calc.is_reversed)
                        or (best_score < score[0] and (not calc.is_reversed))
                    ):
                        # if best_score is None or (best_score is not None and score[0] < best_score):
                        if best_score is not None:
                            self._logger.info(
                                "Node {} with is better than {} ({} vs {}). Now, it is the best'".format(
                                    node_name, best_node, score[0], best_score
                                )
                            )
                        else:
                            self._logger.info("Node {} with an score of {} is the best'".format(node_name, score[0]))
                        best_node = node_name
                        best_score = score[0]
                        generators_to_be_optimized = generator_pop
                        weights_generator_to_be_optimized = weights_generator

                if db_logger.is_enabled and self.experiment_id is not None:
                    db_logger.add_experiment_results(self.experiment_id, node_name, image_paths, score)
            except Exception as ex:
                self._logger.error("An error occured while trying to gather results from {}: {}".format(node_name, ex))
                traceback.print_exc()

        if self.cc.settings["master"]["calculate_score"] and scores:
            best_node = sorted(scores, key=lambda x: x[1], reverse=ScoreCalculatorFactory.create().is_reversed)[-1]
            self._logger.info(
                "Best result: {}:{} = {}".format(best_node[0]["address"], best_node[0]["port"], best_node[1])
            )

        if weights_generator_to_be_optimized is None:
            self._logger.info("Impossible to create samples. Problems gathering results.")

        # Save images
        dataset = MixedGeneratorDataset(
            generators_to_be_optimized,
            weights_generator_to_be_optimized,
            self.cc.settings["master"]["score_sample_size"],
            self.cc.settings["trainer"]["mixture_generator_samples_mode"],
        )

        self._logger.info("-------------------------------------------------------------------")
        self._logger.info("-------------------------------------------------------------------")
        generators = generators_to_be_optimized
        weights_generators = weights_generator_to_be_optimized
        population_size = len(weights_generator_to_be_optimized)
        generations = 200
        score_sample_size = self.cc.settings["master"]["score_sample_size"]
        mixture_generator_samples_mode = self.cc.settings["trainer"]["mixture_generator_samples_mode"]
        mixture_sigma = self.cc.settings["trainer"]["params"]["mixture_sigma"]
        es_random_init = False
        self.optimize_generator_mixture_weights(
            generators,
            weights_generators,
            population_size,
            generations,
            score_sample_size,
            mixture_generator_samples_mode,
            mixture_sigma,
            es_random_init,
        )


    def _gather_results_old(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._logger.info("Collecting results from clients...")

        # Initialize node client
        dataloader = self.cc.create_instance(self.cc.settings["dataloader"]["dataset_name"])
        network_factory = self.cc.create_instance(self.cc.settings["network"]["name"], dataloader.n_input_neurons)
        node_client = NodeClient(network_factory)
        db_logger = DbLogger()

        results = node_client.gather_results(self.cc.settings["general"]["distribution"]["client_nodes"], 120)

        scores = []
        for (node, generator_pop, discriminator_pop, weights_generator, weights_discriminator) in results:
            node_name = "{}:{}".format(node["address"], node["port"])
            try:
                output_dir = self.get_and_create_output_dir(node)

                for generator in generator_pop.individuals:
                    source = generator.source.replace(":", "-")
                    filename = "{}{}.pkl".format(GENERATOR_PREFIX, source)
                    torch.save(
                        generator.genome.net.state_dict(), os.path.join(output_dir, "generator-{}.pkl".format(source))
                    )

                    with open(os.path.join(output_dir, "mixture.yml"), "a") as file:
                        file.write("{}: {}\n".format(filename, weights_generator[generator.source]))

                for discriminator in discriminator_pop.individuals:
                    source = discriminator.source.replace(":", "-")
                    filename = "{}{}.pkl".format(DISCRIMINATOR_PREFIX, source)
                    torch.save(discriminator.genome.net.state_dict(), os.path.join(output_dir, filename))

                # Save images
                dataset = MixedGeneratorDataset(
                    generator_pop,
                    weights_generator,
                    self.cc.settings["master"]["score_sample_size"],
                    self.cc.settings["trainer"]["mixture_generator_samples_mode"],
                )
                image_paths = self.save_samples(dataset, output_dir, dataloader)
                self._logger.info(
                    "Saved mixture result images of client {} to target directory {}.".format(node_name, output_dir)
                )

                # Calculate inception or FID score
                score = float("-inf")
                generators_to_be_optimized = None
                best_score = None
                if self.cc.settings["master"]["calculate_score"]:
                    calc = ScoreCalculatorFactory.create()
                    self._logger.info("Score calculator: {}".format(type(calc).__name__))
                    self._logger.info(
                        "Calculating score score of {}. Depending on the type, this may take very long.".format(
                            node_name
                        )
                    )

                    score = calc.calculate(dataset)
                    self._logger.info(
                        "Node {} with weights {} yielded a score of {}".format(node_name, weights_generator, score)
                    )
                    scores.append((node, score))

                    if best_score is None or (best_score is not None and score[0] < best_score):
                        best_score = score[0]
                        generators_to_be_optimized = generator_pop
                        weights_generator_to_be_optimized = weights_generator

                if db_logger.is_enabled and self.experiment_id is not None:
                    db_logger.add_experiment_results(self.experiment_id, node_name, image_paths, score)
            except Exception as ex:
                self._logger.error("An error occured while trying to gather results from {}: {}".format(node_name, ex))
                traceback.print_exc()

        if self.cc.settings["master"]["calculate_score"] and scores:
            best_node = sorted(scores, key=lambda x: x[1], reverse=ScoreCalculatorFactory.create().is_reversed)[-1]
            self._logger.info(
                "Best result: {}:{} = {}".format(best_node[0]["address"], best_node[0]["port"], best_node[1])
            )

        # Save images
        dataset = MixedGeneratorDataset(
            generators_to_be_optimized,
            weights_generator_to_be_optimized,
            self.cc.settings["master"]["score_sample_size"],
            self.cc.settings["trainer"]["mixture_generator_samples_mode"],
        )

        self._logger.info("-------------------------------------------------------------------")
        self._logger.info("-------------------------------------------------------------------")
        generators = generators_to_be_optimized
        weights_generators = weights_generator_to_be_optimized
        population_size = len(weights_generator_to_be_optimized)
        generations = 200
        score_sample_size = self.cc.settings["master"]["score_sample_size"]
        mixture_generator_samples_mode = self.cc.settings["trainer"]["mixture_generator_samples_mode"]
        mixture_sigma = self.cc.settings["trainer"]["params"]["mixture_sigma"]
        es_random_init = False
        self.optimize_generator_mixture_weights(
            generators,
            weights_generators,
            population_size,
            generations,
            score_sample_size,
            mixture_generator_samples_mode,
            mixture_sigma,
            es_random_init,
        )

    def optimize_generator_mixture_weights(
        self,
        generators,
        weights_generators,
        population_size,
        generations,
        score_sample_size,
        mixture_generator_samples_mode,
        mixture_sigma,
        es_random_init=False,
    ):

        from helpers.pytorch_helpers import noise
        import numpy as np
        from collections import OrderedDict

        # Not necessary for single-cell grids, as mixture must always be [1]
        if population_size == 1:
            return

        # Create random vector from latent space
        z_noise = noise(score_sample_size, generators.individuals[0].genome.data_size)

        # Include option to start from random weights
        if es_random_init:
            aux_weights = np.random.rand(len(weights_generators))
            aux_weights /= np.sum(aux_weights)
            weights_generators = OrderedDict(zip(weights_generators.keys(), aux_weights))
            weights_generators = weights_generators

        dataset = MixedGeneratorDataset(
            generators,
            weights_generators,
            score_sample_size,
            mixture_generator_samples_mode,
            z_noise,
        )

        score_calc = ScoreCalculatorFactory.create()
        self._logger.info("Score calculator: {}".format(type(score_calc).__name__))
        self._logger.info(
            "Calculating score score of {}. Depending on the type, this may take very long.".format(weights_generators)
        )

        score = score_calc.calculate(dataset)[0]
        init_score = score

        self._logger.info("Mixture weight mutation - Starting mixture weights optimization ...")
        self._logger.info("Init score: {}\tInit weights: {}.".format(init_score, weights_generators))

        for g in range(generations):

            # Mutate mixture weights
            z = np.random.normal(loc=0, scale=mixture_sigma, size=len(weights_generators))
            transformed = np.asarray([value for _, value in weights_generators.items()])
            transformed += z

            # Don't allow negative values, normalize to sum of 1.0
            transformed = np.clip(transformed, 0, None)
            transformed /= np.sum(transformed)
            new_mixture_weights = OrderedDict(zip(weights_generators.keys(), transformed))

            # TODO: Testing the idea of not generating the images again
            dataset = MixedGeneratorDataset(
                generators,
                new_mixture_weights,
                score_sample_size,
                mixture_generator_samples_mode,
                z_noise,
            )

            if score_calc is not None:
                score_after_mutation, tvd_after_mutation = score_calc.calculate(dataset)
                self._logger.info(
                    "Mixture weight mutation - Generation: {} \tScore of new weights: {} - TVD after mutation: {}\tNew weights: {}.".format(
                        g, score_after_mutation, tvd_after_mutation, new_mixture_weights
                    )
                )

                # For fid the lower the better, for inception_score, the higher the better
                if (score_after_mutation < score and score_calc.is_reversed) or (
                    score_after_mutation > score and (not score_calc.is_reversed)
                ):
                    weights_generators = new_mixture_weights
                    score = score_after_mutation
                    tvd = tvd_after_mutation
                    self._logger.info(
                        "NEW MIXTURE - Generation: {} \tNew score: {} - tNew TVD: \tWeights changed to: {}.".format(
                            g, score, tvd, weights_generators
                        )
                    )
        mixture_weights_generators = weights_generators

        self._logger.info(
            "Mixture weight mutation - Score before mixture weight optimzation: {}\tScore after mixture weight optimzation: {} - TVD: {}".format(
                init_score, score, tvd
            )
        )
        return mixture_weights_generators

    def _gather_results_old(self):
        self._logger.info('Collecting results from clients...')

        # Initialize node client
        dataloader = self.cc.create_instance(self.cc.settings['dataloader']['dataset_name'])
        network_factory = self.cc.create_instance(self.cc.settings['network']['name'], dataloader.n_input_neurons)
        node_client = NodeClient(network_factory)
        db_logger = DbLogger()

        results = node_client.gather_results(self.cc.settings['general']['distribution']['client_nodes'], 120)

        scores = []
        for (node, generator_pop, discriminator_pop, weights_generator, weights_discriminator) in results:
            node_name = '{}:{}'.format(node['address'], node['port'])
            try:
                output_dir = self.get_and_create_output_dir(node)

                for generator in generator_pop.individuals:
                    source = generator.source.replace(':', '-')
                    filename = '{}{}.pkl'.format(GENERATOR_PREFIX, source)
                    torch.save(generator.genome.net.state_dict(),
                               os.path.join(output_dir, 'generator-{}.pkl'.format(source)))

                    with open(os.path.join(output_dir, 'mixture.yml'), "a") as file:
                        file.write('{}: {}\n'.format(filename, weights_generator[generator.source]))

                for discriminator in discriminator_pop.individuals:
                    source = discriminator.source.replace(':', '-')
                    filename = '{}{}.pkl'.format(DISCRIMINATOR_PREFIX, source)
                    torch.save(discriminator.genome.net.state_dict(),
                               os.path.join(output_dir, filename))

                # Save images
                dataset = MixedGeneratorDataset(generator_pop,
                                                weights_generator,
                                                self.cc.settings['master']['score_sample_size'],
                                                self.cc.settings['trainer']['mixture_generator_samples_mode'])

                if self.cc.settings['dataloader']['dataset_name'] == 'circular':
                    image_paths = self.save_samples(dataset, output_dir, dataloader, 5000, 1000)
                else:
                    image_paths = self.save_samples(dataset, output_dir, dataloader)

                self._logger.info('Saved mixture result images of client {} to target directory {}.'
                                  .format(node_name, output_dir))

                # Calculate inception or FID score
                score = float('-inf')
                if self.cc.settings['master']['calculate_score']:
                    calc = ScoreCalculatorFactory.create()
                    self._logger.info('Score calculator: {}'.format(type(calc).__name__))
                    self._logger.info('Calculating score score of {}. Depending on the type, this may take very long.'
                                      .format(node_name))

                    score = calc.calculate(dataset)
                    self._logger.info('Node {} with weights {} yielded a score of {}'
                                      .format(node_name, weights_generator, score))
                    scores.append((node, score))

                if db_logger.is_enabled and self.experiment_id is not None:
                    db_logger.add_experiment_results(self.experiment_id, node_name, image_paths, score)
            except Exception as ex:
                self._logger.error('An error occured while trying to gather results from {}: {}'.format(node_name, ex))
                traceback.print_exc()

        if self.cc.settings['master']['calculate_score'] and scores:
            best_node = sorted(scores, key=lambda x: x[1], reverse=ScoreCalculatorFactory.create().is_reversed)[-1]
            self._logger.info('Best result: {}:{} = {}'.format(best_node[0]['address'],
                                                               best_node[0]['port'], best_node[1]))

    def get_and_create_output_dir(self, node):
        directory = os.path.join(self.cc.output_dir, 'master', self.cc.settings['general']['distribution']['start_time'],
                                 '{}-{}'.format(node['address'], node['port']))
        os.makedirs(directory, exist_ok=True)
        return directory

    def save_samples(self, dataset, output_dir, image_specific_loader, n_images=10, batch_size=100):
        image_format = self.cc.settings['general']['logging']['image_format']
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        loaded = image_specific_loader.load()
        paths = []

        for i, data in enumerate(dataloader):
            shape = loaded.dataset.train_data.shape if hasattr(loaded.dataset, 'train_data') else None
            path = os.path.join(output_dir, 'mixture-{}.{}'.format(i + 1, image_format))
            image_specific_loader.save_images(Variable(data), shape, path)
            paths.append(path)

            if i + 1 == n_images:
                break

        return paths

    def expand_clients(self):
        clients = self.cc.settings['general']['distribution']['client_nodes']
        clients_to_expand = [c for c in clients if isinstance(c['port'], str) and '-' in c['port']]

        if clients_to_expand:
            clients = [x for x in clients if x not in clients_to_expand]
            for client in clients_to_expand:
                rng = client['port'].split('-')
                if len(rng) != 2 or not rng[0].isdigit() or not rng[1].isdigit():
                    raise Exception('Configuration for client {} has incorrect format.'.format(client))

                for port in range(int(rng[0]), int(rng[1]) + 1):
                    clients.append({'address': client['address'], 'port': port})
            self.cc.settings['general']['distribution']['client_nodes'] = clients
