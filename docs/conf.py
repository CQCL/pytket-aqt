import re
from typing import Any
from urllib.parse import urljoin

from docutils import nodes
from docutils.nodes import Element, TextElement
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

author = "Quantinuum"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

pygments_style = "borland"

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/CQCL/pytket-aqt",
    "use_repository_button": True,
    "use_issues_button": True,
    "logo": {
        "image_light": "Quantinuum_logo_black.png",
        "image_dark": "Quantinuum_logo_white.png",
    },
}

html_static_path = ["_static"]

html_css_files = ["custom.css"]

# -- Extension configuration -------------------------------------------------

pytketdoc_base = "https://tket.quantinuum.com/api-docs/"

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    pytketdoc_base: None,
    "https://qiskit.org/documentation/": None,
    "http://docs.qulacs.org/en/latest/": None,
}

autodoc_member_order = "groupwise"

# The following code is for resolving broken hyperlinks in the doc.


# Mappings for broken hyperlinks that intersphinx cannot resolve
external_url_mapping = {
    "BasePass": urljoin(pytketdoc_base, "passes.html#pytket.passes.BasePass"),
    "Predicate": urljoin(pytketdoc_base, "predicates.html#pytket.predicates.Predicate"),
    "ResultHandle": urljoin(
        pytketdoc_base,
        "backends.html#pytket.backends.resulthandle.ResultHandle",
    ),
    "BackendResult": urljoin(
        pytketdoc_base,
        "backends.html#pytket.backends.backendresult.BackendResult",
    ),
    "Circuit": urljoin(pytketdoc_base, "circuit_class.html#pytket.circuit.Circuit"),
    "BasisOrder": urljoin(pytketdoc_base, "circuit.html#pytket.circuit.BasisOrder"),
    "QubitPauliOperator": urljoin(
        pytketdoc_base, "utils.html#pytket.utils.QubitPauliOperator"
    ),
    "QubitPauliString": urljoin(
        pytketdoc_base, "pauli.html#pytket.pauli.QubitPauliString"
    ),
}

# Correct mappings for intersphinx to resolve
custom_internal_mapping = {
    "pytket.utils.outcomearray.OutcomeArray": "pytket.utils.OutcomeArray",
    "pytket.utils.operators.QubitPauliOperator": "pytket.utils.QubitPauliOperator",
    "pytket.backends.backend.Backend": "pytket.backends.Backend",
    "qiskit.dagcircuit.dagcircuit.DAGCircuit": "qiskit.dagcircuit.DAGCircuit",
    "qiskit.providers.basebackend.BaseBackend": "qiskit.providers.BaseBackend",
    "qiskit.qobj.qasm_qobj.QasmQobj": "qiskit.qobj.QasmQobj",
    "qiskit.result.result.Result": "qiskit.result.Result",
}


def add_reference(
    app: Sphinx, env: BuildEnvironment, node: Element, contnode: TextElement
) -> nodes.reference | None:
    # Fix references in docstrings that are inherited from the
    # base pytket.backends.Backend class.
    mapping = app.config.external_url_mapping
    if node.astext() in mapping:
        newnode = nodes.reference(
            "",
            "",
            internal=False,
            refuri=mapping[node.astext()],
            reftitle=node.get("reftitle", node.astext()),
        )
        newnode.append(contnode)
        return newnode
    return None


def correct_signature(  # noqa: PLR0913
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: dict,
    signature: str,
    return_annotation: str,
) -> tuple[str, str]:
    new_signature = signature
    new_return_annotation = return_annotation
    for k, v in app.config.custom_internal_mapping.items():
        if signature is not None:
            new_signature = new_signature.replace(k, v)
        if return_annotation is not None:
            new_return_annotation = new_return_annotation.replace(k, v)
    # e.g. Replace <CXConfigType.Snake: 0> by
    # CXConfigType.Snake to avoid silent failure in later stages.
    if new_signature is not None:
        enums_signature = re.findall(r"<.+?\: \d+>", new_signature)
        for e in enums_signature:
            new_signature = new_signature.replace(e, e[1 : e.find(":")])

    if new_return_annotation is not None:
        enums_return = re.findall(r"<.+?\: \d+>", new_return_annotation)
        for e in enums_return:
            new_return_annotation = new_return_annotation.replace(e, e[1 : e.find(":")])

    return new_signature, new_return_annotation


def setup(app: Sphinx) -> None:
    app.add_config_value("custom_internal_mapping", {}, "env")
    app.add_config_value("external_url_mapping", {}, "env")
    app.connect("missing-reference", add_reference)
    app.connect("autodoc-process-signature", correct_signature)
