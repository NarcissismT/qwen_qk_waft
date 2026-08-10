from __future__ import annotations

import tempfile
from pathlib import Path

from tests.test_data import (
    test_dataset_converts_displacement_to_work_canvas,
    test_dataset_loads_line_mask_and_instance_annotations,
    test_dataset_rejects_unknown_flow_format,
)
from tests.test_geometry import (
    test_absolute_map_resize_scales_source_coordinates,
    test_identity_map_samples_native_pixels,
    test_translation_and_map_displacement_round_trip,
    test_waft_feature_warp_uses_official_zero_padding,
    test_bfloat16_autocast_keeps_512_identity_coordinates_in_float32,
)
from tests.test_formal_quality import test_ocr_character_retention_and_stage_a_gain
from tests.test_infer import test_native_inference_resamples_rgb_exactly_once
from tests.test_model import (
    test_curriculum_iteration_count_does_not_rescale_sequence_loss,
    test_complete_forward_uses_absolute_backward_map,
    test_evaluation_reports_geometry_topology_and_calibration_metrics,
    test_masked_mean_excludes_nonfinite_values_outside_valid_region,
    test_nonfinite_loss_term_detection_reports_the_source,
    test_official_checkpoint_strictly_loads_vit_dpt_and_update_heads,
    test_official_waft_architecture_contract,
    test_one_cycle_scheduler_starts_below_phase_peak_learning_rate,
    test_phase_transitions_preserve_the_previous_function,
    test_training_losses_are_finite_and_backpropagate,
    test_zero_flow_waft_core_matches_the_official_a2_update_equations,
    test_every_gate_iteration_receives_supervision,
    test_bending_uses_continuous_flat_region_weights,
    test_annotated_text_line_metric_fits_a_real_baseline,
)
from tests.test_probe import (
    test_similarity_margin_excludes_the_true_match_and_cycle_is_consistent,
)
from tests.test_qwen_layout import (
    test_lora_loader_matches_original_diffsynth_fused_ba_rule,
    test_runtime_img_shapes_split_target_then_source,
    test_unit_rotary_frequency_preserves_qk,
)


def main() -> None:
    tests = (
        test_identity_map_samples_native_pixels,
        test_translation_and_map_displacement_round_trip,
        test_absolute_map_resize_scales_source_coordinates,
        test_waft_feature_warp_uses_official_zero_padding,
        test_bfloat16_autocast_keeps_512_identity_coordinates_in_float32,
        test_runtime_img_shapes_split_target_then_source,
        test_unit_rotary_frequency_preserves_qk,
        test_ocr_character_retention_and_stage_a_gain,
        test_complete_forward_uses_absolute_backward_map,
        test_evaluation_reports_geometry_topology_and_calibration_metrics,
        test_training_losses_are_finite_and_backpropagate,
        test_masked_mean_excludes_nonfinite_values_outside_valid_region,
        test_curriculum_iteration_count_does_not_rescale_sequence_loss,
        test_one_cycle_scheduler_starts_below_phase_peak_learning_rate,
        test_nonfinite_loss_term_detection_reports_the_source,
        test_official_checkpoint_strictly_loads_vit_dpt_and_update_heads,
        test_official_waft_architecture_contract,
        test_phase_transitions_preserve_the_previous_function,
        test_similarity_margin_excludes_the_true_match_and_cycle_is_consistent,
        test_zero_flow_waft_core_matches_the_official_a2_update_equations,
        test_every_gate_iteration_receives_supervision,
        test_bending_uses_continuous_flat_region_weights,
        test_annotated_text_line_metric_fits_a_real_baseline,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    with tempfile.TemporaryDirectory(prefix="qwen_qk_waft_test_") as directory:
        test_lora_loader_matches_original_diffsynth_fused_ba_rule(Path(directory))
        print("PASS test_lora_loader_matches_original_diffsynth_fused_ba_rule")
        test_dataset_converts_displacement_to_work_canvas(Path(directory))
        print("PASS test_dataset_converts_displacement_to_work_canvas")
        test_dataset_rejects_unknown_flow_format(Path(directory))
        print("PASS test_dataset_rejects_unknown_flow_format")
        test_dataset_loads_line_mask_and_instance_annotations(Path(directory))
        print("PASS test_dataset_loads_line_mask_and_instance_annotations")
        test_native_inference_resamples_rgb_exactly_once(Path(directory))
        print("PASS test_native_inference_resamples_rgb_exactly_once")


if __name__ == "__main__":
    main()
