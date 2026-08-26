// Entry point for tests
// See the GitHub Action for a demo of how to build and run tests.

#include <iostream>
#include "test/test_activations.cpp"
#include "test/test_conv1d.cpp"
#include "test/test_conv_1x1.cpp"
#include "test/test_convnet.cpp"
#include "test/test_dsp.cpp"
#include "test/test_film.cpp"
#include "test/test_film_realtime_safe.cpp"
#include "test/test_lora.cpp"
#include "test/test_lora_realtime_safe.cpp"
#include "test/test_fast_lut.cpp"
#include "test/test_get_dsp.cpp"
#include "test/test_ring_buffer.cpp"
#include "test/test_wavenet/test_layer.cpp"
#include "test/test_wavenet/test_layer_array.cpp"
#include "test/test_wavenet/test_full.cpp"
#include "test/test_wavenet/test_real_time_safe.cpp"
#include "test/test_wavenet/test_condition_processing.cpp"
#include "test/test_wavenet/test_head1x1.cpp"
#include "test/test_wavenet/test_output_head.cpp"
#include "test/test_wavenet/test_layer_head_config.cpp"
#include "test/test_wavenet/test_layer1x1.cpp"
#include "test/test_wavenet/test_factory.cpp"
#include "test/test_gating_activations.cpp"
#include "test/test_wavenet_gating_compatibility.cpp"
#include "test/test_blending_detailed.cpp"
#include "test/test_input_buffer_verification.cpp"
#include "test/test_linear.cpp"
#include "test/test_lstm.cpp"
#include "test/test_lstm_realtime_safe.cpp"
#include "test/test_wavenet_configurable_gating.cpp"
#include "test/test_noncontiguous_blocks.cpp"
#include "test/test_extensible.cpp"
#include "test/test_container.cpp"
#include "test/test_render_slim.cpp"
#include "test/test_slimmable_wavenet.cpp"
#include "test/test_a2_fast.cpp"
#include "test/test_parametric_wavenet.cpp"
#include "test/test_render_parametric.cpp"

int main()
{
  std::cout << "Running tests..." << std::endl;
  // TODO Automatically loop, catch exceptions, log results

  test_activations::TestFastTanh::test_core_function();
  test_activations::TestFastTanh::test_get_by_init();
  test_activations::TestFastTanh::test_get_by_str();

  test_activations::TestLeakyReLU::test_core_function();
  test_activations::TestLeakyReLU::test_get_by_init();
  test_activations::TestLeakyReLU::test_get_by_str();

  test_activations::TestSoftsign::test_core_function();
  test_activations::TestSoftsign::test_get_by_init();
  test_activations::TestSoftsign::test_get_by_str();

  test_lut::TestFastLUT::test_sigmoid();
  test_lut::TestFastLUT::test_tanh();

  test_activations::TestPReLU::test_core_function();
  test_activations::TestPReLU::test_per_channel_behavior();
  test_activations::TestPReLU::test_wrong_number_of_channels_matrix();
  test_activations::TestPReLU::test_wrong_size_array();
  test_activations::TestPReLU::test_valid_array_size();

  // Typed ActivationConfig tests
  test_activations::TestTypedActivationConfig::test_simple_config();
  test_activations::TestTypedActivationConfig::test_all_simple_types();
  test_activations::TestTypedActivationConfig::test_leaky_relu_config();
  test_activations::TestTypedActivationConfig::test_prelu_single_slope_config();
  test_activations::TestTypedActivationConfig::test_prelu_multi_slope_config();
  test_activations::TestTypedActivationConfig::test_leaky_hardtanh_config();
  test_activations::TestTypedActivationConfig::test_softsign_config();
  test_activations::TestTypedActivationConfig::test_from_json_string();
  test_activations::TestTypedActivationConfig::test_from_json_object();
  test_activations::TestTypedActivationConfig::test_from_json_prelu_multi();
  test_activations::TestTypedActivationConfig::test_from_json_softsign_string();
  test_activations::TestTypedActivationConfig::test_from_json_softsign_object();
  test_activations::TestTypedActivationConfig::test_unknown_activation_throws();

  test_dsp::test_construct();
  test_dsp::test_get_input_level();
  test_dsp::test_get_output_level();
  test_dsp::test_has_input_level();
  test_dsp::test_has_output_level();
  test_dsp::test_set_input_level();
  test_dsp::test_set_output_level();
  test_dsp::test_reset_prewarm_on_reset_default();
  test_dsp::test_set_prewarm_on_reset();
  test_dsp::test_reset_and_prewarm_forces_prewarm();
  test_dsp::test_reset_and_prewarm_restores_prewarm_on_throw();
  test_dsp::test_scoped_prewarm_on_reset_default();

  test_linear::test_direct_known_values();
  test_linear::test_fft_matches_direct_irregular_chunks();
  test_linear::test_auto_selection();
  test_linear::test_parse_implementation();
  test_linear::test_direct_process_realtime_safe();
  test_linear::test_fft_process_realtime_safe();
  test_linear::test_auto_direct_process_realtime_safe();
  test_linear::test_auto_fft_process_realtime_safe();

  test_ring_buffer::test_construct();
  test_ring_buffer::test_reset();
  test_ring_buffer::test_reset_with_receptive_field();
  test_ring_buffer::test_write();
  test_ring_buffer::test_read_with_lookback();
  test_ring_buffer::test_advance();
  test_ring_buffer::test_rewind();
  test_ring_buffer::test_multiple_writes_reads();
  test_ring_buffer::test_reset_zeros_history_area();
  test_ring_buffer::test_rewind_preserves_history();

  test_conv1d::test_construct();
  test_conv1d::test_set_size();
  test_conv1d::test_reset();
  test_conv1d::test_process_basic();
  test_conv1d::test_process_with_bias();
  test_conv1d::test_process_multichannel();
  test_conv1d::test_process_dilation();
  test_conv1d::test_process_multiple_calls();
  test_conv1d::test_get_output_different_sizes();
  test_conv1d::test_set_size_and_weights();
  test_conv1d::test_get_num_weights();
  test_conv1d::test_reset_multiple();
  test_conv1d::test_process_grouped_basic();
  test_conv1d::test_process_grouped_with_bias();
  test_conv1d::test_process_grouped_multiple_groups();
  test_conv1d::test_process_grouped_kernel_size();
  test_conv1d::test_process_grouped_dilation();
  test_conv1d::test_process_grouped_channel_isolation();
  test_conv1d::test_get_num_weights_grouped();
  test_conv1d::test_process_8x4_kernel6_matches_reference();
  test_conv1d::test_process_1x4_kernel16_matches_reference();

  test_conv_1x1::test_construct();
  test_conv_1x1::test_construct_with_groups();
  test_conv_1x1::test_construct_validation_in_channels();
  test_conv_1x1::test_construct_validation_out_channels();
  test_conv_1x1::test_process_basic();
  test_conv_1x1::test_process_with_bias();
  test_conv_1x1::test_process_underscore();
  test_conv_1x1::test_process_grouped_basic();
  test_conv_1x1::test_process_grouped_with_bias();
  test_conv_1x1::test_process_grouped_multiple_groups();
  test_conv_1x1::test_process_grouped_channel_isolation();
  test_conv_1x1::test_process_underscore_grouped();
  test_conv_1x1::test_set_max_buffer_size();
  test_conv_1x1::test_process_multiple_calls();
  test_conv_1x1::test_process_underscore_4x6_matches_reference();
  test_conv_1x1::test_process_underscore_8x6_matches_reference();
  test_conv_1x1::test_process_underscore_4x4_with_bias_matches_reference();

  test_film::test_set_max_buffer_size();
  test_film::test_process_bias_only();
  test_film::test_process_scale_only();
  test_film::test_process_inplace_with_shift();
  test_film::test_process_inplace_scale_only();
  test_film::test_process_inplace_partial_frames();
  test_film::test_process_with_groups();
  test_film::test_process_with_groups_scale_only();

  test_film_realtime_safe::test_allocation_tracking_pass();
  test_film_realtime_safe::test_allocation_tracking_fail();
  test_film_realtime_safe::test_film_process_with_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_without_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_inplace_with_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_inplace_without_shift_realtime_safe();
  test_film_realtime_safe::test_film_process_large_dimensions_realtime_safe();
  test_film_realtime_safe::test_film_process_partial_frames_realtime_safe();
  test_film_realtime_safe::test_film_process_varying_dimensions_realtime_safe();
  test_film_realtime_safe::test_film_process_consecutive_calls_realtime_safe();

  test_lora::test_set_max_buffer_size();
  test_lora::test_process_matches_hand_computed_matmul();
  test_lora::test_process_inplace_adds_to_input();
  test_lora::test_process_partial_frames();
  test_lora::test_zero_weights_produce_zero_output();
  test_lora::test_process_varying_condition_matches_per_frame_recompute();

  test_lora_realtime_safe::test_lora_process_constant_condition_realtime_safe();
  test_lora_realtime_safe::test_lora_process_varying_condition_realtime_safe();
  test_lora_realtime_safe::test_lora_process_inplace_realtime_safe();
  test_lora_realtime_safe::test_lora_process_partial_frames_realtime_safe();
  test_lora_realtime_safe::test_lora_process_consecutive_calls_realtime_safe();

  test_wavenet::test_layer::test_gated();
  test_wavenet::test_layer::test_layer_getters();
  test_wavenet::test_layer::test_non_gated_layer();
  test_wavenet::test_layer::test_layer_activations();
  test_wavenet::test_layer::test_layer_multichannel();
  test_wavenet::test_layer::test_layer_bottleneck();
  test_wavenet::test_layer::test_layer_bottleneck_gated();
  test_wavenet::test_layer_array::test_layer_array_basic();
  test_wavenet::test_layer_array::test_layer_array_receptive_field();
  test_wavenet::test_layer_array::test_layer_array_with_head_input();
  test_wavenet::test_layer_array::test_layer_array_different_activations();
  test_wavenet::test_full::test_wavenet_model();
  test_wavenet::test_output_head::test_post_stack_head_receptive_field();
  test_wavenet::test_output_head::test_wavenet_with_post_stack_head_processes();
  test_wavenet::test_output_head::test_wavenet_with_two_layer_post_stack_head_applies_activation_per_layer_input();
  test_wavenet::test_layer_head_config::test_legacy_head_size_and_head_bias_implies_kernel_one();
  test_wavenet::test_layer_head_config::test_nested_head_with_kernel_size_three();
  test_wavenet::test_layer_head_config::test_nested_head_with_dilation_three();
  test_wavenet::test_full::test_wavenet_multiple_arrays();
  test_wavenet::test_full::test_wavenet_zero_input();
  test_wavenet::test_full::test_wavenet_different_buffer_sizes();
  test_wavenet::test_full::test_wavenet_prewarm();
  test_wavenet::test_head1x1::test_head1x1_inactive();
  test_wavenet::test_head1x1::test_head1x1_active();
  test_wavenet::test_head1x1::test_head1x1_gated();
  test_wavenet::test_head1x1::test_head1x1_groups();
  test_wavenet::test_head1x1::test_head1x1_different_out_channels();
  test_wavenet::test_layer1x1::test_layer1x1_active();
  test_wavenet::test_layer1x1::test_layer1x1_inactive();
  test_wavenet::test_layer1x1::test_layer1x1_inactive_bottleneck_mismatch();
  test_wavenet::test_layer1x1::test_layer1x1_post_film_active();
  test_wavenet::test_layer1x1::test_layer1x1_post_film_is_applied_for_every_gating_mode();
  test_wavenet::test_layer1x1::test_layer1x1_post_film_inactive_with_layer1x1_inactive();
  test_wavenet::test_layer1x1::test_layer1x1_gated();
  test_wavenet::test_layer1x1::test_layer1x1_groups();
  test_wavenet::test_factory::test_factory_without_head_key();
  test_wavenet::test_allocation_tracking_pass();
  test_wavenet::test_allocation_tracking_fail();
  test_wavenet::test_conv1d_process_realtime_safe();
  test_wavenet::test_conv1d_grouped_process_realtime_safe();
  test_wavenet::test_conv1d_grouped_dilated_process_realtime_safe();
  test_wavenet::test_layer_process_realtime_safe();
  test_wavenet::test_layer_bottleneck_process_realtime_safe();
  test_wavenet::test_layer_grouped_process_realtime_safe();
  test_wavenet::test_layer_all_films_with_shift_realtime_safe();
  test_wavenet::test_layer_all_films_without_shift_realtime_safe();
  test_wavenet::test_layer_post_activation_film_gated_realtime_safe();
  test_wavenet::test_layer_post_activation_film_blended_realtime_safe();
  test_wavenet::test_layer_array_process_realtime_safe();
  test_wavenet::test_process_realtime_safe();
  test_wavenet::test_process_with_post_stack_head_realtime_safe();
  test_wavenet::test_process_3in_2out_realtime_safe();
  test_wavenet::test_condition_processing::test_with_condition_dsp();
  test_wavenet::test_condition_processing::test_with_condition_dsp_multichannel();

  test_convnet::test_convnet_basic();
  test_convnet::test_convnet_batchnorm();
  test_convnet::test_convnet_multiple_blocks();
  test_convnet::test_convnet_zero_input();
  test_convnet::test_convnet_different_buffer_sizes();
  test_convnet::test_convnet_prewarm();
  test_convnet::test_convnet_multiple_calls();

  // LSTM tests
  test_lstm::test_lstm_basic();
  test_lstm::test_lstm_multiple_layers();
  test_lstm::test_lstm_zero_input();
  test_lstm::test_lstm_different_buffer_sizes();
  test_lstm::test_lstm_prewarm();
  test_lstm::test_lstm_multiple_calls();
  test_lstm::test_lstm_multichannel();
  test_lstm::test_lstm_large_hidden_size();
  test_lstm::test_lstm_different_input_size();
  test_lstm::test_lstm_state_evolution();
  test_lstm::test_lstm_no_layers();

  // LSTM real-time safety tests (issue #218)
  test_lstm_realtime_safe::test_lstm_process_single_layer_realtime_safe();
  test_lstm_realtime_safe::test_lstm_process_multi_layer_realtime_safe();
  test_lstm_realtime_safe::test_lstm_process_multichannel_realtime_safe();
  test_lstm_realtime_safe::test_lstm_process_large_hidden_realtime_safe();
  test_lstm_realtime_safe::test_lstm_process_consecutive_calls_realtime_safe();

  // Gating activations tests
  test_gating_activations::TestGatingActivation::test_basic_functionality();
  test_gating_activations::TestGatingActivation::test_with_custom_activations();
  //  test_gating_activations::TestGatingActivation::test_error_handling();

  // Wavenet gating compatibility tests
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_wavenet_style_gating();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_column_by_column_processing();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_memory_contiguity();
  test_wavenet_gating_compatibility::TestWavenetGatingCompatibility::test_multiple_channels();

  test_gating_activations::TestBlendingActivation::test_basic_functionality();
  test_gating_activations::TestBlendingActivation::test_blending_behavior();
  test_gating_activations::TestBlendingActivation::test_with_custom_activations();
  //  test_gating_activations::TestBlendingActivation::test_error_handling();
  test_gating_activations::TestBlendingActivation::test_edge_cases();

  // Detailed blending tests
  test_blending_detailed::TestBlendingDetailed::test_blending_with_different_activations();
  test_blending_detailed::TestBlendingDetailed::test_input_buffer_usage();

  // Input buffer verification tests
  test_input_buffer_verification::TestInputBufferVerification::test_buffer_stores_pre_activation_values();
  test_input_buffer_verification::TestInputBufferVerification::test_buffer_with_different_activations();

  // Configurable gating/blending tests
  run_configurable_gating_tests();

  // Non-contiguous block correctness tests (outerStride != rows)
  test_noncontiguous_blocks::test_conv1x1_process_toprows();
  test_noncontiguous_blocks::test_conv1x1_process_toprows_with_bias();
  test_noncontiguous_blocks::test_conv1x1_process_toprows_2x2();
  test_noncontiguous_blocks::test_conv1x1_process_toprows_4x4();
  test_noncontiguous_blocks::test_conv1x1_toprows_matches_contiguous();
  test_noncontiguous_blocks::test_film_process_toprows_with_shift();
  test_noncontiguous_blocks::test_film_process_toprows_scale_only();
  test_noncontiguous_blocks::test_film_toprows_matches_contiguous();
  test_noncontiguous_blocks::test_film_process_inplace_toprows();
  test_noncontiguous_blocks::test_gating_output_toprows();
  test_noncontiguous_blocks::test_gating_toprows_matches_contiguous();
  test_noncontiguous_blocks::test_blending_output_toprows();

  test_get_dsp::test_gets_input_level();
  test_get_dsp::test_gets_output_level();
  test_get_dsp::test_null_input_level();
  test_get_dsp::test_null_output_level();
  test_get_dsp::test_version_patch_one_beyond_supported();
  test_get_dsp::test_version_minor_one_beyond_supported();
  test_get_dsp::test_version_too_early();
  test_get_dsp::test_is_version_supported_core_behavior();
  test_get_dsp::test_register_custom_version_support_checker();
  test_get_dsp::test_get_dsp_default_allows_constructor_reset_prewarm();
  test_get_dsp::test_get_dsp_default_inherits_scoped_prewarm_default();
  test_get_dsp::test_get_dsp_prewarm_option_suppresses_constructor_reset_prewarm();
  test_get_dsp::test_get_dsp_prewarm_option_forces_constructor_reset_prewarm();
  test_get_dsp::test_get_dsp_with_returned_config_constructs_once();

  // Finally, some end-to-end tests.
  test_get_dsp::test_load_and_process_nam_files();

  // Extensibility: external architecture registration and get_dsp (issue #230)
  test_extensible::run_extensibility_tests();

  // Container / SlimmableContainer tests
  test_container::test_container_loads_from_json();
  test_container::test_container_processes_audio();
  test_container::test_container_slimmable_selects_submodel();
  test_container::test_container_boundary_values();
  test_container::test_container_slimmable_breakpoints();
  test_container::test_container_empty_submodels_throws();
  test_container::test_container_last_max_value_must_cover_one();
  test_container::test_container_unsorted_submodels_throws();
  test_container::test_container_sample_rate_mismatch_throws();
  test_container::test_container_load_from_file();
  test_container::test_container_default_is_max_size();
  test_container::test_container_reset_only_resets_active_submodel();
  test_container::test_container_switch_resets_before_activation();
  test_container::test_container_get_allowed_sizes_reads_array_layers_shape();
  test_container::test_container_prewarm_for_ratio_resets_without_activating();
  test_container::test_container_set_slimmable_size_after_prewarm_skips_redundant_reset();
  test_container::test_container_set_slimmable_size_without_prewarm_still_resets();
  test_container::test_container_prewarm_for_ratio_already_active_is_noop();
  test_container::test_container_prewarm_for_ratio_stale_stage_falls_back_to_reset();

  // [Fork] ContainerModel knob-API delegation (Gap 2)
  test_container::test_container_knob_accessors_delegate_to_first_submodel();
  test_container::test_container_set_knob_values_reaches_inactive_submodel();

  // Render --slim tests
  test_render_slim::test_slim_changes_output();
  test_render_slim::test_slim_rejects_non_slimmable();
  test_render_slim::test_slim_boundary_values();
  test_render_slim::test_slim_applied_before_processing();

  // SlimmableWavenet tests
  test_slimmable_wavenet::test_loads_from_file();
  test_slimmable_wavenet::test_implements_slimmable();
  test_slimmable_wavenet::test_processes_audio();
  test_slimmable_wavenet::test_slimming_changes_output();
  test_slimmable_wavenet::test_boundary_values();
  test_slimmable_wavenet::test_default_is_max_size();
  test_slimmable_wavenet::test_ratio_mapping();
  test_slimmable_wavenet::test_slimmable_breakpoints();
  test_slimmable_wavenet::test_from_json();
  test_slimmable_wavenet::test_wavenet_without_slimmable_loads_as_regular();
  test_slimmable_wavenet::test_unsupported_method_throws();
  test_slimmable_wavenet::test_slimmed_matches_small_model();

#if defined(NAM_ENABLE_A2_FAST)
  // A2 fast-path WaveNet: detector coverage + numerical match against generic.
  test_a2_fast::test_detector_matches_nano();
  test_a2_fast::test_detector_matches_medium();
  test_a2_fast::test_detector_matches_width5();
  test_a2_fast::test_detector_matches_width6();
  test_a2_fast::test_detector_matches_width7();
  test_a2_fast::test_detector_matches_standard();
  test_a2_fast::test_detector_matches_width9();
  test_a2_fast::test_parametric_detector_matches_supported_widths();
  test_a2_fast::test_parametric_detector_rejects_lora_without_schema_version();
  test_a2_fast::test_parametric_detector_rejects_lora_without_rank();
  test_a2_fast::test_parametric_detector_accepts_lora();
  test_a2_fast::test_parametric_detector_accepts_explicit_film_type();
  test_a2_fast::test_detector_accepts_nonstandard_head_scale();
  test_a2_fast::test_detector_rejects_wrong_channels();
  test_a2_fast::test_detector_rejects_wrong_kernel_sizes();
  test_a2_fast::test_detector_rejects_wrong_activation();
  test_a2_fast::test_detector_rejects_gating();
  test_a2_fast::test_detector_rejects_condition_dsp();
  test_a2_fast::test_detector_rejects_legacy_gated();
  test_a2_fast::test_matches_generic_nano();
  test_a2_fast::test_matches_generic_medium();
  test_a2_fast::test_matches_generic_width5();
  test_a2_fast::test_matches_generic_width6();
  test_a2_fast::test_matches_generic_width7();
  test_a2_fast::test_matches_generic_standard();
  test_a2_fast::test_matches_generic_width9();
  test_a2_fast::test_parametric_skip_matches_generic_nano();
  test_a2_fast::test_parametric_skip_matches_generic_medium();
  test_a2_fast::test_parametric_skip_matches_generic_width5();
  test_a2_fast::test_parametric_skip_matches_generic_width6();
  test_a2_fast::test_parametric_skip_matches_generic_width7();
  test_a2_fast::test_parametric_skip_matches_generic_standard();
  test_a2_fast::test_parametric_skip_matches_generic_width9();
  test_a2_fast::test_parametric_matches_generic_nano();
  test_a2_fast::test_parametric_matches_generic_medium();
  test_a2_fast::test_parametric_matches_generic_width5();
  test_a2_fast::test_parametric_matches_generic_width6();
  test_a2_fast::test_parametric_matches_generic_width7();
  test_a2_fast::test_parametric_matches_generic_standard();
  test_a2_fast::test_parametric_matches_generic_width9();
  test_a2_fast::test_parametric_lora_matches_generic_nano();
  test_a2_fast::test_parametric_lora_matches_generic_medium();
  test_a2_fast::test_parametric_lora_matches_generic_width5();
  test_a2_fast::test_parametric_lora_matches_generic_width6();
  test_a2_fast::test_parametric_lora_matches_generic_width7();
  test_a2_fast::test_parametric_lora_matches_generic_standard();
  test_a2_fast::test_parametric_lora_matches_generic_width9();
  test_a2_fast::test_parametric_lora_matches_generic_rank_equals_channels();
  test_a2_fast::test_parametric_no_lora_still_matches_generic();
  test_a2_fast::test_prewarm_matches_generic_nano();
  test_a2_fast::test_prewarm_matches_generic_standard();
  test_a2_fast::test_process_realtime_safe_nano();
  test_a2_fast::test_process_realtime_safe_medium();
  test_a2_fast::test_process_realtime_safe_width5();
  test_a2_fast::test_process_realtime_safe_width6();
  test_a2_fast::test_process_realtime_safe_width7();
  test_a2_fast::test_process_realtime_safe_standard();
  test_a2_fast::test_process_realtime_safe_width9();

  // [Fork] Knob smoothing convergence (Gap 1): proves the one-pole smoother
  // actually ramps rather than snapping instantly, and that it converges rather than
  // stalling partway.
  test_a2_fast::test_parametric_knob_smoothing_converges();
  // [Fork] Knob smoothing must recalibrate for a larger buffer size reached via process()'s
  // implicit SetMaxBufferSize() fallback, not just via an explicit Reset().
  test_a2_fast::test_parametric_knob_smoothing_recalibrates_after_implicit_resize();
#endif

  // ParametricWaveNet: config parsing, construction, processing, SetKnobValues, registry, factory
  test_get_dsp::test_dual_payload_prefers_embedded_model();
  test_get_dsp::test_dual_payload_without_embed_reads_top_level();
  test_get_dsp::test_bogus_top_level_without_embed_throws();
  test_parametric_wavenet::test_untagged_legacy_model_is_rejected();
  test_parametric_wavenet::test_residual_model_is_rejected();
  test_parametric_wavenet::test_unknown_head_mode_is_rejected();
  test_parametric_wavenet::test_newer_schema_version_is_rejected();
  test_parametric_wavenet::test_skip_model_loads_and_takes_fast_path();
  test_parametric_wavenet::test_complete_lora_tagged_model_takes_fast_path();
  test_parametric_wavenet::test_incomplete_lora_tagged_model_takes_generic_path();
  test_parametric_wavenet::test_explicit_film_type_still_takes_fast_path();
  test_parametric_wavenet::test_config_parse_valid_json();
  test_parametric_wavenet::test_config_parse_missing_layers_throws();
  test_parametric_wavenet::test_config_parse_invalid_channel_count_throws();
  test_parametric_wavenet::test_config_parse_intermediate_channel_count_does_not_throw();
  test_parametric_wavenet::test_config_parse_missing_head_scale_throws();
  test_parametric_wavenet::test_config_parse_missing_parametric_block_throws();
  test_parametric_wavenet::test_config_parse_missing_condition_size_throws();
  test_parametric_wavenet::test_config_parse_condition_size_too_large_throws();
  test_parametric_wavenet::test_config_parse_parameters_must_be_array_of_objects();
  test_parametric_wavenet::test_config_parse_standard_channel_count();
  test_parametric_wavenet::test_config_parse_param_defs_optional();

  test_parametric_wavenet::test_config_parse_lora_block_parses();
  test_parametric_wavenet::test_config_parse_lora_absent_is_inactive();
  test_parametric_wavenet::test_config_parse_lora_missing_rank_throws();
  test_parametric_wavenet::test_config_parse_lora_zero_rank_throws();
  test_parametric_wavenet::test_construct_and_process_with_lora();
  test_parametric_wavenet::test_lora_active_output_differs_from_lora_inactive();
  test_parametric_wavenet::test_lora_weight_count_matches_layout();

  test_parametric_wavenet::test_construct_with_one_param();
  test_parametric_wavenet::test_construct_with_multiple_params();
  test_parametric_wavenet::test_get_parameter_defs();
  test_parametric_wavenet::test_get_parameter_defs_includes_steps();
  test_parametric_wavenet::test_processes_finite_output();
  test_parametric_wavenet::test_processes_zero_input_finite();
  test_parametric_wavenet::test_different_buffer_sizes();
  test_parametric_wavenet::test_set_knob_values_changes_output();
  test_parametric_wavenet::test_set_knob_values_excess_values_no_crash();
  test_parametric_wavenet::test_set_knob_values_empty_no_crash();
  test_parametric_wavenet::test_default_knobs_midpoint();
  test_parametric_wavenet::test_registry_has_parametric_wavenet();
  test_parametric_wavenet::test_factory_creates_and_processes_nano();
  test_parametric_wavenet::test_factory_nano_knobs_change_output();
  test_parametric_wavenet::test_standard_wavenet_unaffected();
  test_parametric_wavenet::test_dc_blocker_removes_dc_at_silence();
  test_parametric_wavenet::test_dc_blocker_passes_ac_signal();

  // [Fork] Knob smoothing convergence (Gap 1, ParametricWaveNet's independent
  // smoother implementation).
  test_parametric_wavenet::test_knob_smoothing_converges();

  // [Fork] render_parametric CLI integration test (Gap 3): exercises the real
  // compiled tool as a subprocess.
  test_render_parametric::test_render_parametric_respects_knobs();

  std::cout << "Success!" << std::endl;
#ifdef ADDASSERT
  std::cerr << "===============================================================" << std::endl;
  std::cerr << "Checking that we're successfully asserting. We should now fail." << std::endl;
  assert(false);
#endif
  return 0;
}
