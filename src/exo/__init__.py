from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("exo")
except PackageNotFoundError:
    # Direct source deployments intentionally have no installed Exo metadata.
    __version__ = "0.3.70"
