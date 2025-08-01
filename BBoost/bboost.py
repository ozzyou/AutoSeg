import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import torch_kmeans

from lavis.models import load_model_and_preprocess
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from positional_encodings.torch_encodings import PositionalEncoding2D
from scipy.optimize import linear_sum_assignment
from crfseg import CRF

from .caption_filtering import filter_captions

def clean_clusters(tensor, footprint_size=3, max_iter=8):
    tensor = tensor.float()  # Convert to float for precision in calculations
    batch_size = tensor.shape[0]

    for i in range(max_iter):
        new_tensor = []
        for b in range(batch_size):
            image = tensor[b]
            new_image = majority_filter(image, footprint_size)
            new_tensor.append(new_image.unsqueeze(0))
        new_tensor = torch.cat(new_tensor, dim=0)
        mask = torch.abs(tensor - new_tensor) > 1e-5  # Tolerance for floating point errors
        if not torch.any(mask):
            break
        tensor = new_tensor
    return tensor


def resize(clusters, new_shape):
    resized_tensor = np.empty(new_shape, dtype=np.int64)
    for (k, image) in enumerate(clusters):
         resized_tensor[k] = cv2.resize(image, new_shape[1:], interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized_tensor)

def compute_cost(clustering1, clustering2):
    return torch.sum(clustering1 != clustering2)

def align_clusterings(clusterings):
    # Find the reference clustering (the one with the most unique clusters)
    ref_clustering_idx = np.argmax([len(np.unique(clustering)) for clustering in clusterings])
    ref_clustering = clusterings[ref_clustering_idx]

    # Align each clustering to the reference clustering
    aligned_clusterings = []
    for i, clustering in enumerate(clusterings):
        if i == ref_clustering_idx:
            aligned_clusterings.append(clustering)  # No need to align the reference clustering
            continue

        # Compute the cost matrix
        unique_clusters_ref = np.unique(ref_clustering)
        unique_clusters = np.unique(clustering)
        cost_matrix = np.zeros((len(unique_clusters_ref), len(unique_clusters)))
        for i, label1 in enumerate(unique_clusters_ref):
            for j, label2 in enumerate(unique_clusters):
                cost_matrix[i, j] = compute_cost(clustering == label2, ref_clustering == label1)

        # Apply the Hungarian algorithm to find the best alignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create the aligned clustering
        aligned_clustering = clustering.clone()
        for old_label, new_label in zip(unique_clusters[col_ind], unique_clusters_ref[row_ind]):
            aligned_clustering[clustering == old_label] = new_label

        aligned_clusterings.append(aligned_clustering)

    return aligned_clusterings

def interpolate_pos_encoding(pos_embed, emb_size):
    # Assuming pos_embed is of shape (1, npatch + 1, dim)
    npatch = pos_embed.shape[1] - 1
    N = npatch
    dim = pos_embed.shape[-1]

    # New dimensions
    w = emb_size
    h = emb_size

    if npatch == w * h:
        return pos_embed

    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]

    w0 = w
    h0 = h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        size=(int(w0), int(h0)),
        mode='bicubic',
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

def create_probability_map(clusterings, epsilon=1e-6):
    num_clusters = max([torch.max(clustering) for clustering in clusterings]) + 1
    prob_map = torch.zeros(list(clusterings[0].shape) + [num_clusters]).cuda()

    for clustering in clusterings:
        clustering = clustering.cuda()
        for label in range(num_clusters):
            prob_map[:,:,label] += (clustering == label)

    prob_map /= len(clusterings)
    prob_map += epsilon

    return prob_map / torch.sum(prob_map, axis=-1, keepdims=True)

class BBoost(torch.nn.Module):
    def __init__(self, device, scales=[384, 512], cluster_range=(2, 8), smoothness_weight=6, smoothness_theta=0.8,
                 pos_emb_dim=256, use_nucleus=True, num_beams=3, top_p=1, repetition_penalty=100.0,
                 attention_mode="global", use_background=True, n_passes=5, min_length=4, max_length=25):

        super().__init__()
        model, vis_processors, txt_processors = load_model_and_preprocess("blip_caption", "base_coco")

        self.device = device
        self.BLIPcap = model.to(device)
        self.captioning = True
        self.prompt = self.init_prompt()
        self.vis_processors = vis_processors
        self.txt_processors = txt_processors

        self.n_passes = n_passes
        self.min_length = min_length
        self.max_length = max_length

        #clustering
        self.crf = CRF(n_spatial_dims=2, requires_grad=False, smoothness_weight=smoothness_weight, smoothness_theta=smoothness_theta)
        self.scales = scales
        self.img_size = max(self.scales)
        self.output_size = (max(self.scales) // 16, max(self.scales) // 16)
        self.cluster_range = cluster_range
        self.pos_emb_dim = pos_emb_dim
        self.BLIPcap.visual_encoder.pos_embed = nn.Parameter(
            interpolate_pos_encoding(self.BLIPcap.visual_encoder.pos_embed, self.output_size[0]))

        self.use_nucleus = use_nucleus
        self.num_beams = num_beams
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.attention_mode = attention_mode
        self.use_background = use_background

    def init_prompt(self):
        prompt = [self.BLIPcap.prompt]
        prompt = self.BLIPcap.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt.input_ids[:, 0] = self.BLIPcap.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]
        return prompt

    def forward(self, raw_images, gt_mask=None, clean=True, return_clusters=False):
        raw_images = raw_images['image'].float().to(self.device)
        raw_images = raw_images / 255
        raw_images = raw_images.to(self.device).unsqueeze(0)
        batch_size = raw_images.shape[0]
        clusterings = []
        max_emb_size = max(self.scales) // 16
        if gt_mask is None:
            for img_size in self.scales:
                emb_size = img_size // 16
                p_enc_2d = PositionalEncoding2D(self.pos_emb_dim).to(self.device)

                self.BLIPcap.visual_encoder.pos_embed = nn.Parameter(
                    interpolate_pos_encoding(self.BLIPcap.visual_encoder.pos_embed, emb_size))
                self.BLIPcap.visual_encoder.patch_embed.img_size = (img_size, img_size)

                image = Resize(size=(img_size, img_size), antialias=True)(raw_images).to(self.device)
                embs = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]
                embs = embs.reshape(batch_size, emb_size, emb_size, -1)
                p_enc = p_enc_2d(embs)
                embs = torch.cat([embs, p_enc], dim=-1)

                for n_clust in range(self.cluster_range[0], self.cluster_range[1] + 1):
                    kmeans = torch_kmeans.KMeans(n_clusters=n_clust, verbose=False)
                    result = kmeans(embs.flatten(1,2)).labels
                    result_np = result.reshape(batch_size, emb_size, emb_size).cpu().numpy()
                    result_np = resize(result_np, (batch_size, max_emb_size, max_emb_size))
                    clusterings.append(result_np)
                    del result, result_np
                    torch.cuda.empty_cache()

                del embs, image, p_enc, kmeans
                # del embs, image, kmeans
                torch.cuda.empty_cache()
            prob_maps = []

            # Alignment
            for i in range(batch_size):
                aligned = align_clusterings([clusterings[j][i] for j in range(len(clusterings))])
                prob_map = create_probability_map(aligned)
                prob_maps.append(prob_map)

            prob_maps = torch.stack(prob_maps)
            crf_clusters = self.crf(prob_maps)
            final_clusters = torch.argmax(crf_clusters, dim=-1)
        else:
            final_clusters = torch.tensor(gt_mask).unsqueeze(0).to(self.device)
            final_clusters = Resize(size=self.output_size, antialias=True, interpolation=InterpolationMode.NEAREST_EXACT)(final_clusters)

        if clean:
            final_clusters = clean_clusters(final_clusters)

        if self.captioning:
            captions_list = self.generate_captions(raw_images, final_clusters)
            auto_vocabulary = filter_captions(captions_list)
            if return_clusters:
                return clusterings, prob_maps, crf_clusters, final_clusters, auto_vocabulary
            else:
                return auto_vocabulary
        else:
            return final_clusters

    # Get captions from already generated clusters, useful to generate multiple captions from the same cluster
    def generate_captions(self, image, clusters):
        image = Resize(size=(self.img_size, self.img_size), antialias=True)(image).to(self.device)
        image_emb = self.BLIPcap.forward_encoder({"image": image})[:, :-1, :]

        token_list = []
        nr_captions_per_img = []

        for idx, c in enumerate(clusters):
            c = c.unsqueeze(0)
            # get flattened indices of each cluster
            cluster_indices = []

            for i in torch.unique(c):
                cluster_indices.append(torch.where(c.flatten() == i)[0].to(self.device))

            # slice image_emb using cluster indices
            cluster_embs = []

            if self.attention_mode in ["local", "concat", "cls"]:
                pre_attention = self.BLIPcap.visual_encoder.patch_embed(image)
                B = pre_attention.shape[0]
                encoder = self.BLIPcap.visual_encoder
                for i in range(len(cluster_indices)):
                    register_blk = -1
                    x = torch.index_select(pre_attention, 1, cluster_indices[i])
                    cls_tokens = encoder.cls_token.expand(
                        B, -1, -1
                    )  # stole cls_tokens impl from Phil Wang, thanks

                    x = torch.cat((cls_tokens, x), dim=1)
                    x = x + encoder.pos_embed[:, : x.size(1), :]
                    x = encoder.pos_drop(x)

                    for j, blk in enumerate(encoder.blocks):
                        x = blk(x, register_blk == j)
                    x = encoder.norm(x).squeeze()
                    if self.attention_mode == "local":
                        cluster_embs.append(x)
                    elif self.attention_mode == "concat":
                        global_attention_emb = image_emb[idx].squeeze()[cluster_indices[i]]
                        x = torch.concat((x, global_attention_emb),0)
                        cluster_embs.append(x)
                    elif self.attention_mode == "cls":
                        cluster_embs.append(x[0])

            # Every cluster becomes its own tensor with embeddings
            else:
                for i in range(len(cluster_indices)):
                    cluster_embs.append(image_emb[idx].squeeze()[cluster_indices[i]])
            for i in range(self.n_passes):
                for emb in cluster_embs:
                    decoder_out = self.BLIPcap.text_decoder.generate_from_encoder(
                        tokenized_prompt=self.prompt,
                        visual_embeds=emb.clone().detach().unsqueeze(0),
                        sep_token_id=self.BLIPcap.tokenizer.sep_token_id,
                        pad_token_id=self.BLIPcap.tokenizer.pad_token_id,
                        use_nucleus_sampling=self.use_nucleus,
                        num_beams=self.num_beams,
                        max_length=self.max_length,
                        min_length=self.min_length,
                        top_p=self.top_p,
                        repetition_penalty=self.repetition_penalty,
                    )
                    token_list.append(list(decoder_out[0]))
            nr_captions_per_img.append(len(cluster_embs))

        outputs = self.BLIPcap.tokenizer.batch_decode(token_list, skip_special_tokens=True)
        return outputs

def majority_filter(tensor, footprint_size):
    padding_size = footprint_size // 2
    height, width = tensor.shape

    # Padding tensor to handle boundaries
    tensor = F.pad(tensor.unsqueeze(0), (padding_size,) * 4, mode='replicate').squeeze(0)

    # Create a tensor to hold the results
    result = torch.zeros_like(tensor)

    for y in range(padding_size, height + padding_size):
        for x in range(padding_size, width + padding_size):
            # Apply the filter by taking a slice
            window = tensor[y - padding_size:y + padding_size + 1, x - padding_size:x + padding_size + 1]

            # Find the histogram of the window
            hist = torch.histc(window.flatten(), bins=256, min=0, max=255)

            # Find the mode from the histogram
            mode = torch.argmax(hist)

            # Set the result at the center of the window as the mode
            result[y, x] = mode

    # Removing the padding
    result = result[padding_size:-padding_size, padding_size:-padding_size]

    return result

