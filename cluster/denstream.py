"""
DenStream.

https://github.com/Waikato/moa/blob/master/moa/src/main/java/moa/clusterers/denstream/WithDBSCAN.java
"""
import logging
from typing import Dict, List, Iterable, Any

from py4j.java_gateway import JavaGateway
from sklearn.base import BaseEstimator, ClusterMixin

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


class DenStreamWithDBSCAN(BaseEstimator, ClusterMixin):
    """DenStream using DBSCAN."""

    def __init__(self,
                 dimensions: int,
                 window_range: int = 1000,
                 epsilon: float = 0.02,
                 beta: float = 0.2,
                 mu: float = 1.0,
                 number_intialization_points: int = 1000,
                 offline_multiplier: float = 2.0,
                 lambda_: float = 0.25,
                 processing_speed: int = 100) -> None:
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

        self._gateway: JavaGateway = setup_java_gateway(imports=_IMPORTS)

        self._header: Any = self._generate_header()
        self._clusterer: Any = self._initialize_clusterer()
        self._instances: List[Any] = []

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

    @property
    def labels_(self) -> List[int]:
        """Get labels."""
        return [*self.get_clustering_result().values()]

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

    def _create_instance(self, vector) -> Any:
        """Create instance."""
        instance = self._gateway.jvm.DenseInstance(
            float(self._dimensions)
        )

        for index, number in enumerate(vector):
            instance.setValue(index, number)

        instance.setDataset(self._header)

        return instance

    def transform(self) -> None:
        """Transform."""
        raise NotImplementedError

    def partial_fit(self, X: Iterable[Iterable[float]]) -> None:
        """
        Partial fit model.

        :param X: An iterable of vectors.
        """
        for vector in X:
            instance = self._create_instance(vector)

            self._clusterer.trainOnInstanceImpl(instance)
            self._instances.append(instance)

    def fit(self, X: Iterable[Iterable[float]]) -> None:
        """
        Partial fit model.

        :param X: An iterable of vectors.
        """
        self.partial_fit(X)

    def fit_predict(self, X: Iterable[Iterable[float]], y=None) -> List[int]:
        """
        Partial fit model.

        :param X: An iterable of vectors.
        """
        self.fit(X)

        return self.labels_

    def get_clustering_result(self) -> Dict[int, int]:
        """
        Get clustering result.

        Result is in the form {
            'point_index': cluster_index
        }
        """
        return self._clusterer.getClusteringResult().classValues(self._instances)
