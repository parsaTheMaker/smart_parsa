"""Small, self-contained port of Point Transformer V3 serialization.

The Morton and Hilbert encoders follow the official PointTransformerV3
serialization package.  Keeping them local avoids making the training entry
point depend on a second repository checkout.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _morton_encode(grid_coord: torch.Tensor, depth: int) -> torch.Tensor:
    x = grid_coord[:, 0].long()
    y = grid_coord[:, 1].long()
    z = grid_coord[:, 2].long()
    code = torch.zeros_like(x)
    for bit in range(int(depth)):
        code = code | (((x >> bit) & 1) << (3 * bit + 2))
        code = code | (((y >> bit) & 1) << (3 * bit + 1))
        code = code | (((z >> bit) & 1) << (3 * bit + 0))
    return code


def _right_shift(binary: torch.Tensor, k: int = 1) -> torch.Tensor:
    if binary.shape[-1] <= int(k):
        return torch.zeros_like(binary)
    return F.pad(binary[..., :-int(k)], (int(k), 0), mode="constant", value=0)


def _gray2binary(gray: torch.Tensor) -> torch.Tensor:
    shift = 2 ** (int(torch.ceil(torch.log2(torch.tensor(gray.shape[-1], device=gray.device))).item()) - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift //= 2
    return gray


def _hilbert_encode(locs: torch.Tensor, num_dims: int = 3, num_bits: int = 16) -> torch.Tensor:
    """Vectorized 3D Hilbert encoder used by the official PTv3 code."""
    if locs.shape[-1] != int(num_dims):
        raise ValueError(f"Expected {num_dims} coordinate dimensions, got {locs.shape[-1]}.")
    if int(num_dims) * int(num_bits) > 63:
        raise ValueError("Hilbert serialization requires at most 63 encoded bits.")

    bitpack_mask = 1 << torch.arange(0, 8, device=locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)
    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -int(num_bits) :]
    )

    for bit in range(int(num_bits)):
        for dim in range(int(num_dims)):
            mask = gray[:, dim, bit]
            if bit + 1 >= gray.shape[-1]:
                continue
            tail = gray.shape[-1] - bit - 1
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, tail),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, int(num_bits) * int(num_dims)))
    binary = _gray2binary(gray)
    padded = F.pad(binary, (64 - int(num_bits) * int(num_dims), 0), "constant", 0)
    packed = (
        padded.flip(-1).reshape((-1, 8, 8))
        * (1 << torch.arange(0, 8, device=locs.device))
    ).sum(2).squeeze().to(torch.uint8)
    return packed.view(torch.int64).squeeze()


@torch.inference_mode()
def encode(grid_coord: torch.Tensor, batch: torch.Tensor | None, depth: int, order: str) -> torch.Tensor:
    order = str(order)
    if order not in {"z", "z-trans", "hilbert", "hilbert-trans"}:
        raise ValueError(f"Unsupported PTv3 serialization order: {order}")
    coords = grid_coord
    if order.endswith("-trans"):
        coords = coords[:, [1, 0, 2]]
    if order.startswith("z"):
        code = _morton_encode(coords, int(depth))
    else:
        code = _hilbert_encode(coords, num_dims=3, num_bits=int(depth))
    if batch is not None:
        code = (batch.long() << (int(depth) * 3)) | code
    return code
