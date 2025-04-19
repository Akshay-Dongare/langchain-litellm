from litellm import Router


def test_router() -> Router:
    model_group_gpt4 = "gpt-4"
    model_group_to_test = "gpt-3.5-turbo"
    fake_model_prefix = "azure/fake-deployment-name-"
    fake_models_names = [fake_model_prefix + suffix for suffix in ["1", "2"]]
    fake_api_key = "fakekeyvalue"
    fake_api_version = "XXXX-XX-XX"
    fake_api_base = "https://faketesturl/"

    model_list = [
        {
            "model_name": model_group_gpt4,
            "litellm_params": {
                "model": fake_models_names[0],
                "api_key": fake_api_key,
                "api_version": fake_api_version,
                "api_base": fake_api_base,
            },
        },
        {
            "model_name": model_group_to_test,
            "litellm_params": {
                "model": fake_models_names[1],
                "api_key": fake_api_key,
                "api_version": fake_api_version,
                "api_base": fake_api_base,
            },
        },
    ]
    return Router(model_list)
