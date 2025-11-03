from transformers import MarianMTModel, MarianTokenizer

_loaded_models = {}

def load_model_and_tokenizer(model_name):
    if model_name in _loaded_models:
        return _loaded_models[model_name]
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    _loaded_models[model_name] = (model, tokenizer)
    return model, tokenizer
