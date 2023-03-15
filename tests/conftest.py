import logging

import pytest

import spox._future


@pytest.fixture(scope="session", autouse=True)
def execute_before_any_test():
    logging.getLogger().setLevel(logging.DEBUG)
    spox._future.set_value_prop_backend(spox._future.ValuePropBackend.ONNXRUNTIME)
