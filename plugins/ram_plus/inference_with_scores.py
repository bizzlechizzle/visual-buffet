"""Custom RAM++ inference that returns confidence scores.

The standard inference_ram() function discards the sigmoid probabilities.
This module provides inference_ram_with_scores() which returns them.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple


def inference_ram_with_scores(
    image: torch.Tensor,
    model: torch.nn.Module,
) -> Tuple[list[str], list[float], list[str], list[float]]:
    """
    Run RAM++ inference and return tags WITH confidence scores.

    This is a modified version of the generate_tag method that exposes
    the sigmoid probabilities instead of discarding them.

    Args:
        image: Preprocessed image tensor [B, C, H, W]
        model: Loaded RAM++ model

    Returns:
        Tuple of:
        - tags: List of tag names (English)
        - scores: List of confidence scores [0.0-1.0] for each tag
        - tags_chinese: List of tag names (Chinese)
        - thresholds: List of per-class thresholds used (for reference)
    """
    device = image.device

    # Get image embeddings
    image_embeds = model.image_proj(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

    image_cls_embeds = image_embeds[:, 0, :]
    image_spatial_embeds = image_embeds[:, 1:, :]

    bs = image_spatial_embeds.shape[0]
    des_per_class = int(model.label_embed.shape[0] / model.num_class)

    # Normalize and compute reweighted logits
    image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
    reweight_scale = model.reweight_scale.exp()
    logits_per_image = (reweight_scale * image_cls_embeds @ model.label_embed.t())
    logits_per_image = logits_per_image.view(bs, -1, des_per_class)

    # Compute normalized weights
    weight_normalized = F.softmax(logits_per_image, dim=2)
    label_embed_reweight = torch.empty(bs, model.num_class, 512).to(device).to(image.dtype)

    for i in range(bs):
        reshaped_value = model.label_embed.view(-1, des_per_class, 512)
        product = weight_normalized[i].unsqueeze(-1) * reshaped_value
        label_embed_reweight[i] = product.sum(dim=1)

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
    probs = torch.sigmoid(logits)  # <-- THE CONFIDENCE SCORES!

    # Get per-class thresholds
    class_threshold = model.class_threshold.to(device)

    # Threshold to get binary predictions
    targets = (probs > class_threshold).float()

    # Zero out deleted tags
    tag = targets.cpu().numpy()
    tag[:, model.delete_tag_index] = 0

    probs_np = probs.cpu().numpy()
    thresholds_np = class_threshold.cpu().numpy()

    # Build output with scores
    all_tags = []
    all_scores = []
    all_tags_chinese = []
    all_thresholds = []

    for b in range(bs):
        indices = np.argwhere(tag[b] == 1).flatten()

        tags = model.tag_list[indices].tolist() if len(indices) > 0 else []
        scores = probs_np[b, indices].tolist() if len(indices) > 0 else []
        tags_chinese = model.tag_list_chinese[indices].tolist() if len(indices) > 0 else []
        thresholds = thresholds_np[indices].tolist() if len(indices) > 0 else []

        # Sort by confidence (descending)
        if tags:
            sorted_pairs = sorted(zip(scores, tags, tags_chinese, thresholds), reverse=True)
            scores, tags, tags_chinese, thresholds = zip(*sorted_pairs)
            scores = list(scores)
            tags = list(tags)
            tags_chinese = list(tags_chinese)
            thresholds = list(thresholds)

        all_tags.append(tags)
        all_scores.append(scores)
        all_tags_chinese.append(tags_chinese)
        all_thresholds.append(thresholds)

    # Return first batch item (typical use case is batch_size=1)
    return all_tags[0], all_scores[0], all_tags_chinese[0], all_thresholds[0]


def get_all_scores(
    image: torch.Tensor,
    model: torch.nn.Module,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Get raw probability scores for ALL tags (not just those above threshold).

    Useful for analysis or custom thresholding.

    Args:
        image: Preprocessed image tensor [B, C, H, W]
        model: Loaded RAM++ model

    Returns:
        Tuple of:
        - probs: Array of shape [num_tags] with probabilities
        - thresholds: Array of shape [num_tags] with per-class thresholds
        - tag_list: List of all tag names
    """
    device = image.device

    # Abbreviated inference to get logits
    image_embeds = model.image_proj(model.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    image_cls_embeds = image_embeds[:, 0, :]

    bs = 1  # Assume single image
    des_per_class = int(model.label_embed.shape[0] / model.num_class)

    image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
    reweight_scale = model.reweight_scale.exp()
    logits_per_image = (reweight_scale * image_cls_embeds @ model.label_embed.t())
    logits_per_image = logits_per_image.view(bs, -1, des_per_class)

    weight_normalized = F.softmax(logits_per_image, dim=2)
    label_embed_reweight = torch.empty(bs, model.num_class, 512).to(device).to(image.dtype)

    reshaped_value = model.label_embed.view(-1, des_per_class, 512)
    product = weight_normalized[0].unsqueeze(-1) * reshaped_value
    label_embed_reweight[0] = product.sum(dim=1)

    label_embed = torch.nn.functional.relu(model.wordvec_proj(label_embed_reweight))

    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )

    logits = model.fc(tagging_embed[0]).squeeze(-1)
    probs = torch.sigmoid(logits)

    return (
        probs[0].cpu().numpy(),
        model.class_threshold.cpu().numpy(),
        model.tag_list.tolist()
    )
