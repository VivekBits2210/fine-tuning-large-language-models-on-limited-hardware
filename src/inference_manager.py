import torch

from config import TextGenConfiguration


class InferenceManager:
    def __init__(self, text_gen_config: TextGenConfiguration):
        self.text_gen_config = text_gen_config

    def generate_tokens(self, model, inputs):
        return model.generate(
            **inputs,
            bos_token_id=self.text_gen_config.beginning_of_sentence_token_id,
            pad_token_id=self.text_gen_config.pad_token_id,
            do_sample=self.text_gen_config.do_sample,
            top_k=self.text_gen_config.top_k,
            min_length=self.text_gen_config.min_tokens_to_generate,
            max_length=self.text_gen_config.max_tokens_to_generate,
            top_p=self.text_gen_config.top_p,
            num_return_sequence=self.text_gen_config.num_return_sequence
        )

    def infer(self, model, tokenizer, device: str, prompt: str = "This "):
        model.config.use_cache = True
        model.eval()

        inputs = tokenizer(tokenizer.eos_token + prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_sequence = self.generate_tokens(model, inputs)

        generated_text = tokenizer.decode(output_sequence[0],
                                          skip_special_tokens=self.text_gen_config.skip_special_tokens)

        if self.text_gen_config.should_remove_new_lines:
            generated_text = generated_text.replace('\n', '')
        if self.text_gen_config.should_remove_tabs:
            generated_text = generated_text.replace('\t', ' ')

        model.train()
        model.config.use_cache = False

        return generated_text