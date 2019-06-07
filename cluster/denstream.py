"""
DenStream.

https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/clusterers/denstream/WithDBSCAN.java
"""
import logging
from typing import Iterable, Any

from utils import setup_logger, setup_java_gateway

_LOGGER = setup_logger('dstream-with-dbscan', logging.DEBUG)

_IMPORTS = [
    'com.yahoo.labs.samoa.instances.SparseInstance',
    'com.yahoo.labs.samoa.instances.DenseInstance',

    'com.yahoo.labs.samoa.instances.Instance',
    'com.yahoo.labs.samoa.instances.Instances',

    'com.yahoo.labs.samoa.instances.InstanceStream',
    'com.yahoo.labs.samoa.instances.InstancesHeader',
    'com.yahoo.labs.samoa.instances.Attribute',

    'moa.core.FastVector',
    'moa.clusterers.denstream.WithDBSCAN'
]


class DenStreamWithDBSCAN:
    """DenStream using DBSCAN."""

    def __init__(self,
                 dimensions: int,
                 window_range: int,
                 epsilon: float,
                 beta: float,
                 mu: float,
                 number_intialization_points: int,
                 offline_multiplier: float,
                 lambda_: float,
                 processing_speed: int) -> None:
        """
        Initialize clusterer.

        :param dimensions: Data dimensionality.
        :param window_range: Horizon window range.
        :param epsilon: Defines the epsilon neighbourhood.
        :param beta: Beta.
        :param mu: Mu.
        :param number_intialization_points: Number of points to use for initialization.
        :param offline_multiplier: Offline multiplier for epsilion.
        :param lambda_: Lambda.
        :param processing_speed: Number of incoming points per time unit.
        """
        self._dimensions = dimensions

        self._window_range = window_range
        self._epsilon = epsilon
        self._beta = beta
        self._mu = mu
        self._number_intialization_points = number_intialization_points
        self._offline_multiplier = offline_multiplier

        self._lambda_ = lambda_
        self._processing_speed = processing_speed

        self._gateway = setup_java_gateway(imports=_IMPORTS)

        self._header = self._generate_header()
        self._clusterer = self._initialize_clusterer()

    @property
    def window_range(self) -> float:
        """Get horizon window range."""
        return self._window_range

    @property
    def epsilon(self) -> float:
        """Get Epsilon neighbourhood."""
        return self._epsilon

    @property
    def beta(self) -> float:
        """Get DBSCAN beta parameter."""
        return self._beta

    @property
    def mu(self) -> float:
        """Get DBSCAN mu parameter."""
        return self._mu

    @property
    def number_intialization_points(self) -> float:
        """Get Number of points to used for initialization."""
        return self._number_intialization_points

    @property
    def offline_multiplier(self) -> float:
        """Get offline multiplier for epsilion."""
        return self._offline_multiplier

    @property
    def lambda_(self) -> float:
        """Get DBSCAN lambda parameter."""
        return self._lambda_

    @property
    def processing_speed(self) -> float:
        """Get processing speed per time unit."""
        return self._processing_speed

    def _generate_header(self):
        """
        Generate header.

        Follows the same steps as:
        https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/streams/generators/RandomRBFGenerator.java#L154
        """
        gateway = self._gateway

        FastVector = gateway.jvm.FastVector
        Attribute = gateway.jvm.Attribute
        InstancesHeader = gateway.jvm.InstancesHeader
        Instances = gateway.jvm.Instances

        attributes = FastVector()

        for i in range(self._dimensions):
            attributes.addElement(
                Attribute(f'att {(i + 1)}')
            )

        header = InstancesHeader(
            Instances('', attributes, 0)
        )
        header.setClassIndex(header.numAttributes() - 1)

        return header

    def _initialize_clusterer(self) -> Any:
        """Initialize clusterer."""

        clusterer = self._gateway.jvm.WithDBSCAN()

        clusterer.horizonOption.setValue(self.window_range)
        clusterer.epsilonOption.setValue(self.epsilon)
        clusterer.betaOption.setValue(self.beta)
        clusterer.muOption.setValue(self.mu)
        clusterer.initPointsOption.setValue(self.number_intialization_points)
        clusterer.offlineOption.setValue(self.offline_multiplier)
        clusterer.lambdaOption.setValue(self.lambda_)
        clusterer.speedOption.setValue(self.processing_speed)

        clusterer.resetLearning()

        return clusterer

    def transform(self) -> None:
        """Transform."""
        raise NotImplementedError

    def partial_fit(self, batch: Iterable[Iterable[float]]) -> None:
        """
        Partial fit model.

        :param batch: An iterable of vectors.
        """
        for vector in batch:
            instance = self._gateway.jvm.DenseInstance(self._dimensions)

            for index, number in enumerate(vector):
                instance.setValue(index, number)

            instance.setDataset(self._header)

            self._clusterer.trainOnInstanceImpl(instance)

    def fit(self, batch: Iterable[Iterable[float]]) -> None:
        """
        Partial fit model.

        :param batch: An iterable of vectors.
        """
        self.partial_fit(batch)
