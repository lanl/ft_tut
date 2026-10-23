import unittest

from spus_ft.inference import autoregressive_rollout, autoregressive_tensor_rollout


class IdentityStepper:
    def predict_next(self, state, timestep: int):
        return state


class TestInference(unittest.TestCase):
    def test_autoregressive_rollout_feeds_predictions_forward(self) -> None:
        initial = [[1.0, 2.0]]
        targets = [[[0.0, 0.0] for _ in range(3)]]

        result = autoregressive_rollout(IdentityStepper(), initial, targets)

        self.assertEqual(len(result.predictions), 1)
        self.assertEqual(len(result.predictions[0]), 3)
        self.assertEqual(result.predictions[0][0], [1.0, 2.0])
        self.assertEqual(result.predictions[0][2], [1.0, 2.0])

    def test_autoregressive_tensor_rollout_feeds_predictions_forward(self) -> None:
        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def dim(self):
                if isinstance(self.value[0], list) and isinstance(self.value[0][0], list) and isinstance(self.value[0][0][0], list):
                    return 4
                return 3

            def unsqueeze(self, axis):
                if axis == 0:
                    return FakeTensor([self.value])
                if axis == 1:
                    return FakeTensor([[item] for item in self.value])
                raise NotImplementedError

            def to(self, device):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

        class FakeTorch:
            def from_numpy(self, array):
                return FakeTensor(array.tolist())

            def stack(self, tensors, dim=0):
                if dim != 1:
                    raise NotImplementedError
                return [tensor.value for tensor in tensors]

            def inference_mode(self):
                def decorator(fn):
                    return fn
                return decorator

        class FakeModel:
            def __call__(self, state):
                data = state.value
                return FakeTensor([sample[0] for sample in data])

        initial = FakeTensor([[[1.0, 2.0], [3.0, 4.0]]])
        result = autoregressive_tensor_rollout(
            model=FakeModel(),
            initial_condition=initial,
            rollout_steps=2,
            device='cpu',
            torch_module=FakeTorch(),
        )

        self.assertEqual(len(result), 2)

    def test_autoregressive_tensor_rollout_uses_four_dimensional_fns_kf_model_input(self) -> None:
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape

            def dim(self):
                return len(self.shape)

            def unsqueeze(self, axis):
                new_shape = list(self.shape)
                new_shape.insert(axis, 1)
                return FakeTensor(tuple(new_shape))

            def to(self, device):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

        class FakeTorch:
            def stack(self, tensors, dim=0):
                return list(tensors)

            def inference_mode(self):
                def decorator(fn):
                    return fn

                return decorator

        class FakeFNSKFModel:
            in_channels = 2

            def __init__(self):
                self.received_shapes = []

            def __call__(self, state):
                self.received_shapes.append(state.shape)
                return state

        model = FakeFNSKFModel()
        initial = FakeTensor((1, 2, 128, 128))

        result = autoregressive_tensor_rollout(
            model=model,
            initial_condition=initial,
            rollout_steps=2,
            device='cpu',
            torch_module=FakeTorch(),
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(model.received_shapes, [(1, 2, 128, 128), (1, 2, 128, 128)])
