# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from .comparisons import Comparisons
from .methods import Log2FC, TTest, TTestPaired, EdgeRUnpaired, DESeq2Unpaired


__all__ = [Comparisons, Log2FC, TTest, TTestPaired, EdgeRUnpaired, DESeq2Unpaired]
