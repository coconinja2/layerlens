from layer_profiler import LayerProfiler, ProfileConfig


class Handle:
    def remove(self):
        pass


class FakeLayer:
    def __init__(self):
        self.pre_hooks = []
        self.post_hooks = []

    def children(self):
        return iter(())

    def register_forward_pre_hook(self, hook, **kwargs):
        self.pre_hooks.append(hook)
        return Handle()

    def register_forward_hook(self, hook, **kwargs):
        self.post_hooks.append(hook)
        return Handle()

    def __call__(self, value):
        for hook in self.pre_hooks:
            hook(self, (value,))
        output = value + 1
        for hook in self.post_hooks:
            hook(self, (value,), output)
        return output


class FakeModel(FakeLayer):
    def __init__(self):
        super().__init__()
        self.layer = FakeLayer()

    def named_modules(self):
        return iter((("", self), ("model.layers.0", self.layer)))

    def __call__(self, value):
        for hook in self.pre_hooks:
            hook(self, (value,), {})
        output = self.layer(value)
        for hook in self.post_hooks:
            hook(self, (value,), {}, output)
        return output


def test_explicit_prefill_and_decode_are_attributed():
    model = FakeModel()
    config = ProfileConfig(synchronize_device=False, record_shapes=False, record_memory=False)
    with LayerProfiler(model, config) as profiler:
        assert profiler.profile_forward(model, 1, step=0) == 2
        assert profiler.profile_forward(model, 2, step=1) == 3

    assert [event.phase for event in profiler.events] == ["prefill", "decode"]
    assert [event.step for event in profiler.events] == [0, 1]
    assert len(profiler.steps) == 2
    assert profiler.trace().metadata["selected_modules"] == 1


def test_generated_token_ids_can_be_excluded_for_privacy():
    model = FakeModel()
    config = ProfileConfig(synchronize_device=False, record_shapes=False, record_memory=False)
    with LayerProfiler(model, config) as profiler:
        profiler.profile_forward(model, 1, step=0)
    profiler.attach_generated_tokens([42], include_ids=False)

    token = profiler.trace().tokens[0]
    assert token.token_id is None
    assert token.text is None
