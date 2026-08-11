"""Small CPU smoke test for the isolated SHIFT-Crash SMART conditioning path."""

from __future__ import annotations

import torch

from models.shift_crash_smart import ShiftCrashSMART
from shift_crash_training import _relative_prediction_consistency


def main():
    torch.manual_seed(42)
    model = ShiftCrashSMART(
        spatial_dim=3,
        surface_channels=3,
        volume_channels=1,
        parameter_channels=6,
        conditioning_input_channels=6,
        conditioning_parameter_indices=(0, 1, 2, 3, 4, 5),
        latent_dim=32,
        latent_geometry_points=8,
        subsampled_geometry_points=16,
        num_encoder_decoder_blocks=2,
        num_heads=4,
        pos_scale_factor=10,
        dropout=0.0,
        conditioning_hidden_dim=16,
        residual_update_scale=0.5,
        normalize_residuals=True,
        geometry_feature_channels=8,
        query_feature_channels=8,
        part_embedding_size=16,
        part_embedding_dim=4,
    )
    model.configure_conditioning("token_only")
    model.initialize_shift_crash_weights()
    modulators = [module for module in model.modules() if module.__class__.__name__ == "Modulator"]
    if not modulators or not all(module.conditioning_residual_scale == 0.0 for module in modulators):
        raise AssertionError("SHIFT-Crash token-only conditioning did not disable repeated FiLM.")
    model.eval()
    geometry = torch.rand(2, 32, 3)
    query = torch.rand(2, 12, 3)
    geometry_features = torch.rand(2, 32, 8)
    geometry_features[:, :4, -1] = 1.0
    query_features = torch.rand(2, 12, 8)
    geometry_part_ids = torch.randint(0, 16, (2, 32))
    query_part_ids = torch.randint(0, 16, (2, 12))
    empty_volume = query.new_empty((2, 0, 3))
    params = torch.tensor(
        [[-1.0, -0.5, 0.0, 0.5, 1.0, -0.25], [0.75, 0.25, -0.75, -0.25, 0.5, 1.0]],
        dtype=torch.float32,
    )
    if model.conditioning_channels != 6 or model.conditioning_parameter_indices != (0, 1, 2, 3, 4, 5):
        raise AssertionError("SHIFT-Crash did not select all six documented design parameters.")
    prepared = model.prepare_conditioning(params)
    if prepared.shape != (2, 1, 6) or not torch.equal(prepared[:, 0], params):
        raise AssertionError("All-design-variable conditioning selection is incorrect.")

    with torch.no_grad():
        seeds = torch.tensor([101, 202], dtype=torch.long)
        output_2d = model(geometry, query, empty_volume, params, geometry_features, query_features, geometry_part_ids, query_part_ids, seeds)[0]
        output_3d = model(geometry, query, empty_volume, params.unsqueeze(1), geometry_features, query_features, geometry_part_ids, query_part_ids, seeds)[0]
        swapped = model(geometry, query, empty_volume, params.flip(0), geometry_features, query_features, geometry_part_ids, query_part_ids, seeds)[0]
        repeated = model(geometry, query, empty_volume, params, geometry_features, query_features, geometry_part_ids, query_part_ids, seeds)[0]

    if not torch.allclose(output_2d, output_3d, rtol=1.0e-5, atol=1.0e-6):
        raise AssertionError("[B,6] and [B,1,6] raw conditioning paths disagree.")
    if not torch.equal(output_2d, repeated):
        raise AssertionError("Explicit evaluation sampling seeds did not make SMART deterministic.")
    sensitivity = (output_2d - swapped).abs().max().item()
    if not sensitivity > 1.0e-7:
        raise AssertionError(f"Conditioning did not change predictions; max difference={sensitivity:.3e}.")

    model.train()
    prediction = model(geometry, query, empty_volume, params, geometry_features, query_features, geometry_part_ids, query_part_ids)[0]
    prediction.square().mean().backward()
    conditioning_grad_sq = sum(
        float(parameter.grad.float().square().sum().item())
        for parameter in model.material_token.parameters()
        if parameter.grad is not None
    )
    conditioning_grad_norm = conditioning_grad_sq ** 0.5
    if not conditioning_grad_norm > 0.0:
        raise AssertionError("Conditioning MLPs received no gradient.")
    global_head_grad_sq = sum(
        float(parameter.grad.float().square().sum().item())
        for parameter in model.global_response_head.parameters()
        if parameter.grad is not None
    )
    if not global_head_grad_sq > 0.0:
        raise AssertionError("Global response head received no gradient.")

    # Prediction agreement must stay in the same units as relative L2.  A
    # common response scaling therefore cannot silently change its effective
    # fixed SATLoss7 weight.
    response = torch.randn(2, 12, 3)
    match = _relative_prediction_consistency(response * 1.1, response, response)
    scaled_match = _relative_prediction_consistency(response * 11.0, response * 10.0, response * 10.0)
    if not torch.allclose(match, scaled_match, rtol=1.0e-5, atol=1.0e-6):
        raise AssertionError("Relative prediction consistency is not scale invariant.")

    with torch.no_grad():
        _, _, latent_features = model.encode(
            geometry,
            model.prepare_conditioning(params),
            geometry_features=geometry_features,
            geometry_part_ids=geometry_part_ids,
            return_final=True,
            sampling_seeds=seeds,
        )
    latent_std = float(latent_features.float().std().item())
    if not 0.8 < latent_std < 1.2:
        raise AssertionError(f"Residual normalization produced unexpected latent std={latent_std:.4f}.")

    print("SHIFT-Crash conditioning smoke test passed.")
    print(f"FiLM modulators disabled: {len(modulators)}")
    print(f"Selected design-token indices: {model.conditioning_parameter_indices}")
    print(f"Final latent feature std: {latent_std:.6e}")
    print(f"Prediction sensitivity to swapped conditions: {sensitivity:.6e}")
    print(f"Material-token gradient norm: {conditioning_grad_norm:.6e}")
    print(f"Global-response-head gradient norm: {global_head_grad_sq ** 0.5:.6e}")
    print(f"Relative prediction-match value: {float(match):.6e}")


if __name__ == "__main__":
    main()
