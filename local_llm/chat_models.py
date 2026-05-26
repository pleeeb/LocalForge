from langchain_ollama import ChatOllama

_local_model_cache = {}

def get_local_model(model_name: str, temperature: float = 0.2, context_window: int = 8192, timeout: float = 120.0):
    cache_key = f"{model_name}_{temperature}"
    
    if cache_key in _local_model_cache:
        return _local_model_cache[cache_key]
    
    model = ChatOllama(
        model = model_name,
        temperature = temperature,
        num_ctx=context_window,
        keep_alive="1h",
        client_kwargs={
            "timeout": timeout
        }
    )

    _local_model_cache[cache_key] = model
    return model