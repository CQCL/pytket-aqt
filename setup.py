# Copyright 2020-2023 Cambridge Quantum Computing
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

import shutil
import os
from setuptools import setup, find_namespace_packages  # type: ignore

metadata: dict = {}
with open("_metadata.py") as fp:
    exec(fp.read(), metadata)
shutil.copy(
    "_metadata.py",
    os.path.join("pytket", "extensions", "aqt", "_metadata.py"),
)

setup(
    name="pytket-aqt",
    version=metadata["__extension_version__"],
    author="TKET development team",
    author_email="tket-support@cambridgequantum.com",
    python_requires=">=3.8",
    project_urls={
        "Documentation": "https://cqcl.github.io/pytket-aqt/api/index.html",
        "Source": "https://github.com/CQCL/pytket-aqt",
        "Tracker": "https://github.com/CQCL/pytket-aqt/issues",
    },
    description="Extension for pytket, providing access to AQT backends",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_namespace_packages(include=["pytket.*"]),
    include_package_data=True,
    install_requires=["pytket ~= 1.9", "requests ~= 2.22", "types-requests"],
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
)
