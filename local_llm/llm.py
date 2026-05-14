from llama_index.llms.ollama import Ollama

class OllamaLLM:
    def __init__(self, model_name: str, request_timeout: int = 60, context_window: int = 8000, temperature: float = 0.2):
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.context_window = context_window
        self.temperature = temperature

        self.model = Ollama(
            model = self.model_name,
            request_timeout = self.request_timeout,
            context_window = self.context_window,
            temperature = self.temperature
        )
    
    def get_model(self):
        return self.model