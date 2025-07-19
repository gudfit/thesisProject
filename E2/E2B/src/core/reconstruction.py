# src/core/reconstruction.py
import torch
import time
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class SentenceReconstructor:

    @staticmethod
    def reconstruct_sentence(
        model, tokenizer, sentence: str, prompt_len: int
    ) -> Tuple[str, float, float, float]:
        device = model.device
        inputs = tokenizer(sentence, return_tensors="pt")
        full_ids = inputs.input_ids[0].to(device)
        if prompt_len >= len(full_ids):
            return sentence, 0.0, 0.0, 1.0  

        prompt_ids = full_ids[:prompt_len].unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids)
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=len(full_ids) - prompt_len + 5,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
            )
        latency = (time.time() - start_time) * 1000
        reconstructed_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        gen_ids = output_ids[0][prompt_len:]
        ref_cont = full_ids[prompt_len:]
        if len(gen_ids) == 0 or len(ref_cont) == 0:
            ppl = 0.0
        else:
            labels = torch.full_like(gen_ids, -100)
            labels[:len(ref_cont)] = ref_cont
            loss = model(output_ids[0].unsqueeze(0), labels=labels.unsqueeze(0)).loss
            ppl = torch.exp(loss).item()
        
        with torch.no_grad():
            logits = model(output_ids[0].unsqueeze(0)).logits
            probs = torch.softmax(logits[0, prompt_len-1:-1], dim=-1)
            conf = torch.gather(probs, 1, gen_ids.unsqueeze(1)).mean().item()
        
        return reconstructed_text, latency, ppl, conf


