import torch
import torch.nn.functional as F

from megatron.core.transformer.torch_norm import SeeDNorm


def _manual_seednorm(x: torch.Tensor, module: SeeDNorm, activation: str) -> torch.Tensor:
    if activation == "tanh":
        act_fn = torch.tanh
    elif activation == "softsign":
        act_fn = F.softsign
    else:
        raise ValueError(f"Unsupported activation in test: {activation}")

    beta = module.beta
    alpha = module.alpha
    gamma = module.gamma
    if getattr(module, "zero_centered_gamma", False):
        gamma = gamma + torch.ones_like(gamma)
    eps = module.eps

    rescale = act_fn(torch.matmul(x, beta))
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    dynamic_scale = rescale.unsqueeze(-1) * alpha
    return (dynamic_scale + gamma) * x_norm


def test_seednorm_matches_reference_tanh():
    torch.manual_seed(0)
    hidden_size = 7
    module = SeeDNorm(hidden_size=hidden_size, activation="tanh").double()
    x = torch.randn(4, hidden_size, dtype=torch.float64, requires_grad=True)

    out = module(x)
    ref = _manual_seednorm(x, module, activation="tanh")

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_seednorm_supports_softsign_activation():
    torch.manual_seed(1)
    hidden_size = 5
    module = SeeDNorm(hidden_size=hidden_size, activation="softsign").double()
    x = torch.randn(3, hidden_size, dtype=torch.float64, requires_grad=True)

    out = module(x)
    ref = _manual_seednorm(x, module, activation="softsign")

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_seednorm_weight_decay_flags():
    module = SeeDNorm(hidden_size=3, activation="tanh")
    assert getattr(module.alpha, "apply_weight_decay")
    assert getattr(module.beta, "apply_weight_decay")
    assert not hasattr(module.gamma, "apply_weight_decay")

    module = SeeDNorm(hidden_size=3, activation="softsign")
    assert getattr(module.alpha, "apply_weight_decay")
    assert hasattr(module.beta, "skip_weight_decay")
    assert not hasattr(module.beta, "apply_weight_decay")
    assert not hasattr(module.gamma, "apply_weight_decay")

def test_seednorm_supports_zero_centered_gamma():
    torch.manual_seed(2)
    hidden_size = 6
    module = SeeDNorm(hidden_size=hidden_size, activation="tanh", zero_centered_gamma=True).double()
    x = torch.randn(2, hidden_size, dtype=torch.float64, requires_grad=True)

    out = module(x)
    ref = _manual_seednorm(x, module, activation="tanh")

    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)

    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert torch.allclose(module.gamma.detach(), torch.zeros_like(module.gamma.detach()))