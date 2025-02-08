from src.novel_swarms.config.ResultsConfig import ResultsConfig
from src.novel_swarms.results.Cluster import Cluster
from src.novel_swarms.results.Trends import Trends


def foo():
    pass


# define a main function
def main(config: ResultsConfig, world_metadata=None, heterogeneous=False, dim_reduction=False):
    if config.show_trends:
        Trends().graphArchiveComparisons(config.archive)
        Trends().plotMetricHistograms(config.archive)

    clustering = Cluster(config=config, world_metadata=world_metadata, heterogeneous=heterogeneous, dim_reduction=False)
    clustering.displayGUI()
