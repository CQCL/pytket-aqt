# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import sys


class ShortNameFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        num_parts_desired = 2
        parts = record.name.split(".")
        record.shortname = (
            ".".join(parts[-num_parts_desired:])
            if len(parts) >= num_parts_desired
            else record.name
        )
        return super().format(record)


def configure_logging(level: int = logging.WARNING, to_stdout=True, to_file=None):
    """
    Configures logging for pytket_aqt

    By default, logs are sent to stdout with a formatter that includes timestamps,
    log level, and the last two parts of the module name. Optionally, logs can also
    be written to a file.

    The default log level is logging.Warning.

    Log Levels:
        1. DEBUG: Detailed information, typically useful for diagnosing problems.
        2. INFO: Confirmation that things are working as expected.
        3. WARNING: An indication that something unexpected happened.
        4. ERROR: A more serious problem that prevented some functionality.
        5. CRITICAL: A very serious error that may prevent the program from continuing.


    :param level: Log level. Use constants from the `logging` module (e.g., logging.DEBUG, logging.INFO).
    :param to_stdout: Whether to log to stdout
    :param to_file: Path to a log file. If provided, logs will be written to this file.

    It is possible to log both to stdout and a provided log file simultaneously


    :returns:
        The configured package level logger for pytket_aqt.


    """
    logger = logging.getLogger(__package__)
    logger.setLevel(level)
    logger.handlers.clear()  # Avoid duplicate handlers

    formatter = ShortNameFormatter(
        "%(asctime)s [%(levelname)s] %(shortname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if to_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)

    if to_file:
        file_handler = logging.FileHandler(to_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
