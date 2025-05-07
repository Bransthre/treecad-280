from train_autoregressive import BaselineCADGenerator
import torch


def model_inference(
    model: BaselineCADGenerator, renders, rolls, elevations, tgt_masks, tokenizer
):
    beam_size = 5
    batch_size = renders.shape[0]
    seq_len = 512

    # Initialize best sequence contexts with the start token for beam search
    best_sequence_contexts = torch.full(
        (beam_size, 1), tokenizer._sos_token, dtype=torch.long
    ).cuda()  # (B, beam_size, 1)

    # Initialize beam scores (log probabilities of sequences)
    beam_scores = torch.zeros(beam_size).cuda()  # (B, beam_size)
    renders_repeat = renders.repeat(
        (beam_size, 1, 1, 1, 1)
    )  # (B * beam_size, k, 3, 128, 128)
    rolls_repeat = rolls.repeat(beam_size, 1)  # (B * beam_size,)
    elevations_repeat = elevations.repeat(beam_size, 1)  # (B * beam_size,)

    print("renders.shape", renders_repeat.shape)
    print(
        "tokenized_input",
        best_sequence_contexts.reshape(-1, best_sequence_contexts.shape[-1]).shape,
    )
    print("rolls.shape", rolls_repeat.shape)
    print("elevations.shape", elevations_repeat.shape)

    for i in range(seq_len):
        print("best_sequence_contexts.shape", best_sequence_contexts.shape)
        padding_mask = torch.zeros((beam_size, best_sequence_contexts.shape[-1])).cuda()

        # Forward pass with current best sequences
        next_token_logits = model(
            renders_repeat,
            best_sequence_contexts,
            rolls_repeat,
            elevations_repeat,
            padding_mask,
        )  # Reshape to (B, beam_size, vocab_size)

        # Apply beam search: get the top `beam_size` token indices
        print("L236 next_token_logits.shape", next_token_logits.shape)
        next_token_logits = next_token_logits[
            :, :, :-1
        ]  # Remove unwanted dimension (vocab size) if needed
        print("L240 next_token_logits.shape", next_token_logits.shape)
        log_probs = torch.log_softmax(next_token_logits, dim=-1)  # Log probabilities
        print("log_probs.shape", log_probs.shape)

        # Compute total log probability scores for beam search
        beam_scores = (
            beam_scores.unsqueeze(-1) + log_probs
        )  # (B, beam_size, vocab_size)

        # Select top `beam_size` from each batch
        print("beam_scores:", beam_scores.shape)
        top_scores, top_indices = torch.topk(beam_scores.view(-1), beam_size, dim=-1)

        # Extract the corresponding batch and beam indices
        beam_idx = top_indices // tokenizer._vocabulary_size  # Get the batch index
        token_idx = top_indices % tokenizer._vocabulary_size  # Get the token index

        # Update the sequences
        best_sequence_contexts = torch.cat(
            [
                best_sequence_contexts[beam_idx],
                token_idx.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Update the scores
        beam_scores = top_scores

        # Print shapes for debugging
        print("best_sequence_contexts.shape", best_sequence_contexts.shape)
        print("beam_scores.shape", beam_scores.shape)

    # Convert to list of decoded tokens
    best_sequence_contexts = best_sequence_contexts.squeeze().cpu().numpy()  # (B, T)
    decoded_texts = []
    for i in range(best_sequence_contexts.shape[0]):
        decoded_texts.append(
            [
                tokenizer._index_to_token.get(token_id, tokenizer._pad_token)
                for token_id in best_sequence_contexts[i]
            ]
        )
    decoded_texts = ["".join(decoded_text) for decoded_text in decoded_texts]
    return decoded_texts
