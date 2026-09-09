from layer_profiler.privacy import safe_model_identifier


def test_registry_model_identifiers_are_preserved():
    assert safe_model_identifier("Qwen/Qwen3.5-0.8B") == "Qwen/Qwen3.5-0.8B"


def test_local_model_paths_are_not_serialized():
    assert safe_model_identifier("/Users/private/models/demo") == "local-model"
    assert safe_model_identifier("./models/demo") == "local-model"
    assert safe_model_identifier(r"C:\\Users\\private\\model") == "local-model"
