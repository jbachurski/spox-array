import logging

import pytest

import spox._future


@pytest.fixture(scope="session", autouse=True)
def config_value_prop():
    logging.getLogger().setLevel(logging.DEBUG)
    spox._future.set_value_prop_backend(spox._future.ValuePropBackend.ONNXRUNTIME)
