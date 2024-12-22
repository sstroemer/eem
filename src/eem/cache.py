import logging
import pathlib as pl
import hashlib


class Cache:
    """
    Cache class for managing cached entries in the `.eem` directory.
    """

    def __init__(self):
        if not pl.Path(".eem").exists():
            pl.Path(".eem").mkdir()

        self._dir = pl.Path(".eem")
        self._logger = logging.getLogger("eem")

    def has(self, folder: str, key: str):
        """
        Check if an entry exists in the specified cache folder.

        Args:
            folder (str): The name of the folder to check.
            key (str): The name of the entry to check for.

        Returns:
            bool: True if the entry exists, False otherwise.
        """
        file = self._get_folder(folder) / key
        return file.is_file()

    def get(self, folder: str, key: str, *, strict: bool = True):
        """
        Retrieve an entry path from a specified folder and entry key.

        Args:
            folder (str): The folder in which to look for the entry.
            key (str): The key to retrieve within the folder.
            strict (bool, optional): If True, raises an assertion error if the entry does not exist. Defaults to True.

        Returns:
            Path: The path to the requested entry.

        Raises:
            AssertionError: If strict is True and the file does not exist.
        """
        file = self._get_folder(folder) / key
        if strict:
            assert file.is_file(), f"Cached file {file} does not exist."
        return file

    def has_get(self, folder: str, key: str):
        """
        Check if en entry exists in the cache and retrieve its access key to be used.

        Args:
            folder (str): The folder to check.
            key (str): The key to check.

        Returns:
            tuple: A tuple containing a boolean indicating if the entry exists and the access string of the entry.
        """
        return self.has(folder, key), self.get(folder, key, strict=False)

    def mkid(self, *args, **kwargs):
        """
        Generate a unique identifier based on the provided arguments and keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: A 16-character hexadecimal string representing the start of the SHA-256 hash of the input arguments.
        """
        return hashlib.sha256(repr(args + tuple(sorted(kwargs.items()))).encode("utf-8")).hexdigest()[0:16]

    def _get_folder(self, folder: str):
        path = self._dir / folder
        if not path.exists():
            path.mkdir()
        return path
