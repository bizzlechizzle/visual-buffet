"""RAM++ inference with configurable flat threshold.

The default RAM++ inference uses per-class thresholds (0.65-0.90) which are
calibrated for high precision but low recall. This module provides inference
with a configurable flat threshold for better recall.
"""

import numpy as np
import torch
import torch.nn.functional as F


def inference_ram_flat_threshold(
    image: torch.Tensor,
    model: torch.nn.Module,
    threshold: float = 0.5,
) -> tuple[list[str], list[float]]:
    """
    Run RAM++ inference with a flat threshold instead of per-class thresholds.

    Args:
        image: Preprocessed image tensor [B, C, H, W]
        model: Loaded RAM++ model
        threshold: Flat probability threshold (default 0.5)

    Returns:
        Tuple of (tag_names, probabilities) sorted by probability descending
    """
    device = image.device

    # Get image embeddings
    image_embeds = model.image_proj(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

    image_cls_embeds = image_embeds[:, 0, :]

    bs = 1  # Assume single image
    des_per_class = int(model.label_embed.shape[0] / model.num_class)

    # Normalize and compute reweighted logits
    image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
    reweight_scale = model.reweight_scale.exp()
    logits_per_image = (reweight_scale * image_cls_embeds @ model.label_embed.t())
    logits_per_image = logits_per_image.view(bs, -1, des_per_class)

    # Compute normalized weights
    weight_normalized = F.softmax(logits_per_image, dim=2)
    label_embed_reweight = torch.empty(bs, model.num_class, 512).to(device).to(image.dtype)

    reshaped_value = model.label_embed.view(-1, des_per_class, 512)
    product = weight_normalized[0].unsqueeze(-1) * reshaped_value
    label_embed_reweight[0] = product.sum(dim=1)

    label_embed = torch.nn.functional.relu(model.wordvec_proj(label_embed_reweight))

    # Generate tagging predictions
    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )

    # Get logits and convert to probabilities
    logits = model.fc(tagging_embed[0]).squeeze(-1)
    probs = torch.sigmoid(logits)[0]  # Shape: [num_class]

    # Apply FLAT threshold (not per-class)
    mask = probs >= threshold

    # Zero out deleted tags
    if len(model.delete_tag_index) > 0:
        mask[model.delete_tag_index] = False

    # Get passing indices
    indices = torch.where(mask)[0].cpu().numpy()

    if len(indices) == 0:
        return [], []

    # Get tags and probabilities
    probs_np = probs.cpu().numpy()
    tags = model.tag_list[indices].tolist()
    scores = probs_np[indices].tolist()

    # Sort by probability (descending)
    sorted_pairs = sorted(zip(scores, tags), reverse=True)
    scores, tags = zip(*sorted_pairs)

    return list(tags), list(scores)
