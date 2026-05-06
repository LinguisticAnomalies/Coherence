from coherencecalculator.tools.vecloader import VecLoader
import torch
from typing import List, Tuple
from tqdm.auto import tqdm
from dataclasses import dataclass


@dataclass
class TokenizerOutput:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    
class PerplexityGenerator(object):
    def __init__(self, vecLoader:VecLoader):
        self.model = vecLoader.llm
        self.tokenizer = vecLoader.llm_tokenizer
        self.summarizer = vecLoader.summarizer
        self.device = vecLoader.device

    def encode_transcript(self, transcript: str) -> List[TokenizerOutput]:
        """
        Encode a transcript.
        
        Args:
            transcript: Transcript strings
            tokenizer: HuggingFace tokenizer
        
        Returns:
            TokenizerOutput objects
        """

        encodings = self.tokenizer(
            transcript,
            # truncation=True,
            # max_length=1024,
            padding=False,  # We'll handle padding separately if needed
            return_tensors=None  # Return list of lists instead of tensors
        )

        # Convert to tensors efficiently
        input_ids = torch.tensor([encodings['input_ids']], dtype=torch.long)
        attention_mask = torch.tensor([encodings['attention_mask']], dtype=torch.long)
            
        output = TokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
            
        
        return output


    def sliding_window_perplexity(
            self, transcript:str, window_size=60, window_batch_size=1) -> Tuple[float, List[float]]:
        """
        Calculate both average and per-window perplexities for multiple transcripts.
        
        Returns:
            Tuple containing:
            - List of average perplexities for each transcript
            - List of lists containing all window perplexities for each transcript
        """
        encodings = self.encode_transcript(transcript)
        seq_len = encodings.input_ids.size(1)
        
        # Handle short sequences with global perplexity
        if seq_len < window_size:
            input_ids = encodings.input_ids.to(self.device)
            target_ids = input_ids.clone()
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                
            ppl = torch.exp(neg_log_likelihood)
            return ppl.item(), [ppl.item()]
        
        # Process longer sequences with sliding windows
        window_nlls = []
        window_ppls = []  # Store perplexity for each window
        window_positions = list(range(0, seq_len - window_size + 1))
        
        for window_batch_start in range(0, len(window_positions), window_batch_size):
            window_batch_end = min(window_batch_start + window_batch_size, len(window_positions))
            current_windows = window_positions[window_batch_start:window_batch_end]
            
            batch_input_ids = []
            batch_target_ids = []
            
            for begin_loc in current_windows:
                end_loc = begin_loc + window_size
                window_input_ids = encodings.input_ids[:, begin_loc:end_loc]
                batch_input_ids.append(window_input_ids)
                batch_target_ids.append(window_input_ids.clone())
            
            batch_input_ids = torch.cat(batch_input_ids, dim=0).to(self.device)

            batch_target_ids = torch.cat(batch_target_ids, dim=0).to(self.device)
            
            with torch.no_grad():
                
                outputs = self.model(batch_input_ids, labels=batch_target_ids)
                
                
                neg_log_likelihoods = outputs.loss.view(-1)

                # Calculate perplexity for each window
                window_perplexities = torch.exp(neg_log_likelihoods)
                window_ppls.extend(window_perplexities.cpu().tolist())
                window_nlls.extend(neg_log_likelihoods.cpu().tolist())
        
        # Calculate average perplexity for the transcript
        avg_nll = sum(window_nlls) / len(window_nlls)
        avg_ppl = torch.exp(torch.tensor(avg_nll))
        
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
        return avg_ppl.item(), window_ppls

    # def summarize(self, transcript:str, min_length=20, max_length=60) -> str:
        
    #     est_length = len(self.tokenizer.encode(transcript))
    #     # max_length = int(est_length/2) # make summary = half of input length
    #     if est_length <= max_length:
    #         if est_length <= min_length: # use full transcript length
    #             return transcript # return transcript if entire transcript is less than min_length tokens
    #         else:
    #             result = self.summarizer(transcript, max_length=est_length, min_length=min_length, do_sample=False, truncation=True)
    #     else:
    #         result = self.summarizer(transcript, max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        
    #     return result[0]['summary_text']
    
    def __ppl(self, context, prompt):
    
        # 1) Combine them into one full string
        full_text = context +' '+ prompt

        # 2) Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        max_len = self.model.config.max_position_embeddings
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, -max_len:].to(self.device)
        else:
            input_ids = input_ids.to(self.device)

        # 3) Make labels the same as input_ids initially
        labels = input_ids.clone()

        # 4) Compute the length of the prompt (so we can mask it out)
        prompt_len = len(self.tokenizer(context)["input_ids"])

        # 5) Mask the prompt tokens so they do not contribute to the loss
        #    -100 tells the loss function to ignore these positions
        labels[0, :prompt_len] = -100
        with torch.no_grad():
            outputs = self.model(input_ids, labels=labels)
        return outputs.loss.item()


    def sentence_level_perplexity(self, sentences:list) -> Tuple[List[float], List[float]]:
        if len(sentences) == 0:
            return ([0], [0])
        summary = sentences[0]
        i = 1
        while len(summary.split(' ')) <= 10:
            if i >= len(sentences):
                break
            summary = summary +' '+ sentences[i]
            i+=1
            
        ppls1 = [] #context model ppl
        ppls2 = [] #topic model ppl
        for i, sent in enumerate(sentences):
            if i ==0:
                prompt = sent
                context = summary
                ppl = self.__ppl(context, prompt)
                ppls1.append(ppl)
                ppls2.append(ppl)
            else:
                context = context + ' ' +prompt
                prompt = sent

            
                ppls1.append(self.__ppl(context, prompt))
                ppls2.append(self.__ppl(summary, prompt))
        return ppls1, ppls2  
    
    def get_sentence(self, sentences:list) -> Tuple[List[Tuple[str]], List[Tuple[str]]]:
        
        summary = sentences[0]
        i = 1
        while len(summary.split(' ')) <= 10:
            summary = summary +' '+ sentences[i]
            i+=1
        s1 = [] #context model ppl
        s2 = [] #topic model ppl
        for i, sent in enumerate(sentences):
            if i ==0:
                prompt = sent
                context = summary
                s1.append((context, prompt))
                s2.append((context, prompt))
            else:
                context = context + ' ' +prompt
                prompt = sent

                s1.append((context, prompt))
                s2.append((summary, prompt))
        return s1, s2  
    

