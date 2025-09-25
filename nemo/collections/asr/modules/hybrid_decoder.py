
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from torch.nn.utils.rnn import pad_sequence
import torch

def find_first_difference(a, b, pad_token: int = 2, eos_token: int = 3):
    """
    Find the first index where sequences a and b differ (ignoring pad tokens).

    Notes:
    - Accepts 1D tensors/lists (single sequence) or 2D tensors/lists (batched).
    - Positions with value equal to pad_token are ignored in the comparison.
    - If no non-pad difference is found, returns min(effective_length(a), effective_length(b)).

    Returns a tuple:
    - first_diff_index: LongTensor [batch] first differing index (or min effective length if no diff)
    - effective_length_a: LongTensor [batch] number of non-pad tokens in a
    - eos_at_first_diff_in_b: bool indicating whether b[first_diff_index] equals eos_token for all batch items
    """
    # Convert inputs to tensors if needed
    a_is_tensor = torch.is_tensor(a)
    b_is_tensor = torch.is_tensor(b)
    if not a_is_tensor:
        a = torch.as_tensor(a)
    if not b_is_tensor:
        b = torch.as_tensor(b)

    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("first_difference expects 1D or 2D inputs (optionally batched).")

    # Align batch sizes (broadcasting batch of 1 if necessary)
    if a.size(0) != b.size(0):
        if a.size(0) == 1:
            a = a.expand(b.size(0), -1)
        elif b.size(0) == 1:
            b = b.expand(a.size(0), -1)
        else:
            raise ValueError("Batch sizes must match or one input must be a single sequence.")

    batch_size = a.size(0)
    len_a = a.size(1)
    len_b = b.size(1)
    max_len = max(len_a, len_b)

    # Choose device for temporary padded tensors
    device = a.device if a.device.type != 'cpu' else b.device
    dtype = a.dtype

    # Right-pad both to the same length with pad_token
    apad = torch.full((batch_size, max_len), pad_token, dtype=dtype, device=device)
    bpad = torch.full((batch_size, max_len), pad_token, dtype=dtype, device=device)
    apad[:, :len_a] = a.to(device)
    bpad[:, :len_b] = b.to(device)

    # Effective non-pad lengths
    nonpad_a = (apad != pad_token)
    nonpad_b = (bpad != pad_token)
    eff_len_a = nonpad_a.sum(dim=1)
    eff_len_b = nonpad_b.sum(dim=1)
    min_eff_len = torch.minimum(eff_len_a, eff_len_b)

    # Differences only where both are non-pad
    diff_mask = (apad != bpad) & nonpad_a & nonpad_b

    has_diff = diff_mask.any(dim=1)
    # argmax on float finds first True index (since False->0, True->1)
    first_diff_pos = diff_mask.float().argmax(dim=1)
    result = torch.where(has_diff, first_diff_pos, min_eff_len)

    # Whether b has eos_token at the first differing index for all batch items
    eos_at_first_diff_in_b = (b[torch.arange(len(result)), result] == eos_token).all().item()

    return result, eff_len_a, eos_at_first_diff_in_b

def find_first_index_of_b_last_in_a(a, b):
    """
    Return the first index in a where the last element of b appears.
    If not found, return 0.

    Args:
        a: Tensor [batch, len_a] or [len_a]
        b: Tensor [batch, len_b] or [len_b]

    Returns:
        Tensor [batch] with the first occurrence index per batch (0 if no match)
    """
    # Handle single sequence case
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    
    batch_size, len_a = a.size()

    # edge case: empty sequence a
    if len_a == 0:
        return torch.zeros(batch_size, dtype=torch.long, device=a.device)
    
    # Get last element of b for each batch
    target = b[:, -1:]  # shape: [batch_size, 1]
    
    # Find matches using vectorized comparison
    matches = (a == target)  # shape: [batch_size, seq_len_a]
    
    # Use argmax to find first occurrence, with fallback to 0
    first_indices = matches.long().argmax(dim=1)
    
    # If no match found, argmax returns 0, but we need to check if position 0 actually matches
    # If position 0 doesn't match and no other position matches, keep 0 as fallback
    no_match_mask = ~matches.any(dim=1)
    first_indices[no_match_mask] = 0
    
    return first_indices


class HybridDecoder:
    def __init__(self, decoder_model, classifier_model, tokenizer, beam_size=1, length_penalty=0.0, max_delta_length=50, batch_size=1):
        self.decoder = decoder_model.decoder
        self.classifier = classifier_model
        self.embedding = decoder_model.embedding
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.len_pen = length_penalty
        self.max_delta_len = max_delta_length
        self.max_seq_length = decoder_model.max_sequence_length
        self.batch_size = batch_size
        self.bos = tokenizer.bos_id
        self.pad = tokenizer.pad_id
        self.eos = tokenizer.eos_id

    def _mask_padded_tokens(self, tokens, pad_id):
        mask = tokens != pad_id
        return mask

    def _one_step_forward(
        self,
        decoder_input_ids=None,
        encoder_hidden_states=None,
        encoder_input_mask=None,
        decoder_mems_list=None,
        pos=0,
        return_scores: bool = True,
    ):
        """
        One step of autoregressive output generation.

        Args:
            decoder_input_ids: starting sequence of tokens to generate from;
                if None, generation will start from a batch of <bos> tokens
            encoder_hidden_states: output of the encoder for conditional
                sequence generation; if None, generator will use unconditional
                mode (e.g., language modeling)
            encoder_input_mask: input mask used in the encoder
            decoder_mems_list: list of size num_layers with cached activations
                of sequence (x[1], ..., x[k-1]) for fast generation of x[k]
            pos: starting position in positional encoding
        """

        decoder_hidden_states = self.embedding.forward(decoder_input_ids, start_pos=pos)
        decoder_input_mask = self._mask_padded_tokens(decoder_input_ids, self.pad).float()

        if encoder_hidden_states is not None:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states,
                decoder_input_mask,
                encoder_hidden_states,
                encoder_input_mask,
                decoder_mems_list,
                return_mems=False,
            )
        else:
            decoder_mems_list = self.decoder.forward(
                decoder_hidden_states, decoder_input_mask, decoder_mems_list, return_mems=False
            )
        with self.classifier.with_log_softmax_enabled(return_scores) as classifier:
            # Use the most recent hidden states to compute logits
            logits = classifier.forward(hidden_states=decoder_mems_list)

        return logits, decoder_mems_list

    def _forward(self, decoder_input_ids=None, encoder_hidden_states=None, encoder_input_mask=None, prev_y_sequence=None, cached_decoder_mems_list=None):

        if cached_decoder_mems_list is None:
            cur_pos = 0
        else:
            cur_pos = cached_decoder_mems_list[0].size(1)

        # decoder_tokens = decoder_input_ids
        batch_size, decoder_input_len = decoder_input_ids.size()

        logits, decoder_mems_list = self._one_step_forward(
            decoder_input_ids, encoder_hidden_states, encoder_input_mask, cached_decoder_mems_list, cur_pos
        )
        logits = logits[:, -decoder_input_ids.size(1):, :]
        topk_scores, topk_indices = torch.topk(logits, self.beam_size, dim=-1)
        topk_indices = topk_indices.view(batch_size, -1)

        y_sequence = topk_indices[:, self.prompt_token_num - 1 :]
        # if cur_pos == 0:  # remove prompt tokens
        #     y_sequence = topk_indices[:, self.prompt_token_num - 1 :]
        # else:
        #     y_sequence = torch.cat([prev_y_sequence, topk_indices], dim=1)

        return y_sequence, decoder_mems_list
         

    def forward(self, reference_sequences=None, decoder_input_ids=None, enc_states=None, enc_mask=None, K=1):

        self.decoder.eval()
        self.classifier.eval()
        self.embedding.eval()
        
        device = enc_mask.device
        forward_counter = 0
        self.prompt_token_num = decoder_input_ids.size(1)

        tmp_refereunce_sequences = []
        for idx, seq in enumerate(reference_sequences):
            if len(seq) == 0: # no reference sequence from fast decoder
                tmp_refereunce_sequences.append(torch.tensor([self.eos], device=device))
            else:
                tmp_refereunce_sequences.append(torch.tensor(seq, device=device))
        reference_sequences = tmp_refereunce_sequences
        
        hypotheses = []
        loop_checker = torch.full((decoder_input_ids.size(0),), -1, device=device)

        while True:
            reference_sequences_tensor = pad_sequence(reference_sequences, batch_first=True, padding_value=self.pad)

            tmp_reference_sequences_tensor = torch.cat([decoder_input_ids, reference_sequences_tensor], dim=1)
            # verify process
            tmp_y_sequence, tmp_decoder_mems_list = self._forward(
                tmp_reference_sequences_tensor, enc_states, enc_mask, None, None
            )

            first_difference_index, eff_len_reference, eos_check = find_first_difference(
                reference_sequences_tensor, tmp_y_sequence, self.pad, self.eos
            )
            seq_check = torch.equal(first_difference_index, eff_len_reference)

            
            # passed
            if seq_check and eos_check:
                for i in range(tmp_y_sequence.size(0)):
                    mask = (tmp_y_sequence[i] < 1123).nonzero(as_tuple=True)[0]
                    if len(mask) > 0:
                        final_sequence = tmp_y_sequence[i][:mask[0].item()]
                    else:
                        final_sequence = tmp_y_sequence[i]
                    hypotheses.append(Hypothesis(
                        score=0.0, 
                        y_sequence=final_sequence,
                        text=self.tokenizer.ids_to_text(final_sequence.tolist())))
                break
            # this case is
            #    1) fast decoder may have deletion issue
            #    2) transformer decoder may have repetition issue
            # so inference at most K times more and stop
            elif seq_check and not eos_check:
                tmp_y_sequence = torch.cat([decoder_input_ids, tmp_y_sequence], dim=1)
                tmp_y_sequence, tmp_decoder_mems_list = self._forward(
                    tmp_y_sequence, enc_states, enc_mask, None, None
                )
                forward_counter += 1
                for i in range(tmp_y_sequence.size(0)):
                    mask = (tmp_y_sequence[i] < 1123).nonzero(as_tuple=True)[0]
                    if len(mask) > 0:
                        final_sequence = tmp_y_sequence[i][:min(first_difference_index[i] + K, mask[0].item())]
                    else:
                        final_sequence = tmp_y_sequence[i]
                    hypotheses.append(Hypothesis(
                        score=0.0, 
                        y_sequence=final_sequence,
                        text=self.tokenizer.ids_to_text(final_sequence.tolist())))
                break
            # unexpected loop
            elif not torch.any(loop_checker < first_difference_index):
                break

            loop_checker = first_difference_index.clone()

            # correction process
            for i in range(reference_sequences_tensor.size(0)):
                idx = first_difference_index[i]
                if idx < reference_sequences_tensor.size(1): 
                    reference_sequences_tensor[i, idx] = tmp_y_sequence[i, idx]

            confirmed_sequence = torch.full((reference_sequences_tensor.size(0), reference_sequences_tensor.size(1)), self.pad, dtype=reference_sequences_tensor.dtype, device=reference_sequences_tensor.device)
            for i in range(reference_sequences_tensor.size(0)):
                idx = first_difference_index[i]
                length = min(idx+1, reference_sequences_tensor.size(1))
                confirmed_sequence[i, :length] = reference_sequences_tensor[i, :length]

            # patch generation
            tmp_confirmed_sequence = torch.cat([decoder_input_ids, confirmed_sequence], dim=1)
            tmp_y_sequence, tmp_decoder_mems_list = self._forward(
                tmp_confirmed_sequence, enc_states, enc_mask, None, None
            )
            forward_counter += 1

            # patch application
            reference_sequences = []
            for i in range(tmp_y_sequence.size(0)):  # batch
                patch = tmp_y_sequence[i, first_difference_index[i] : first_difference_index[i] + 2]
                eos_check = patch[patch >= 1123].numel() == 0
                if eos_check:
                    reference_sequences.append(reference_sequences_tensor[i][:first_difference_index[i]])
                else:
                    e = find_first_index_of_b_last_in_a(
                        reference_sequences_tensor[
                            i, first_difference_index[i] : first_difference_index[i] + 2 * len(patch)
                        ],
                        patch,
                    ).item()
                    new_ref = torch.cat(
                        [
                            reference_sequences_tensor[i, : first_difference_index[i]],
                            patch,
                            reference_sequences_tensor[i, first_difference_index[i] + e + 1 :],
                        ]
                    )
                    mask = (new_ref < 1123).nonzero(as_tuple=True)[0]
                    if len(mask) > 0:
                        new_ref = new_ref[:mask[0].item()]
                    reference_sequences.append(new_ref)

        return hypotheses
        
