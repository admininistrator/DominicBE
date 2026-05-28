from app.services.chat_service import _calculate_max_output_tokens


def test_chat_token_budget_uses_selected_model_limits_instead_of_global_fixed_value():
    assert _calculate_max_output_tokens(
        estimated_input_tokens=100,
        remaining_tokens=10000,
        context_window=1000,
        model_max_output_tokens=250,
    ) == 250

    assert _calculate_max_output_tokens(
        estimated_input_tokens=100,
        remaining_tokens=10000,
        context_window=300,
        model_max_output_tokens=5000,
    ) == 200

    assert _calculate_max_output_tokens(
        estimated_input_tokens=100,
        remaining_tokens=250,
        context_window=1000,
        model_max_output_tokens=5000,
    ) == 150
