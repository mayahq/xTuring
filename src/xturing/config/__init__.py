import torch

from xturing.utils.interactive import is_interactive_execution

# check if cuda is available, if not use cpu and throw warning
if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

DEFAULT_DTYPE = torch.float16 if DEFAULT_DEVICE.type == "cuda" else torch.float32
IS_INTERACTIVE = is_interactive_execution()

if DEFAULT_DEVICE.type == "cpu":
    print("WARNING: CUDA is not available, using CPU instead, can be very slow")


def assert_not_cpu_int8():
    assert DEFAULT_DEVICE.type != "cpu", "Int8 models are not supported on CPU"
