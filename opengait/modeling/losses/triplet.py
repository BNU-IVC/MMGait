import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from itertools import combinations
from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist


class Cross_TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(Cross_TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings1, embeddings2, labels):
        # embeddings: [n, c, p], label: [n]
        embeddings1 = embeddings1.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
        embeddings2 = embeddings2.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        dist = self.ComputeDistance(embeddings1, embeddings2)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        ap_dist, an_dist = self.Convert2Triplets(labels, labels, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)
        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist

class New_Cross_TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(New_Cross_TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings1, embeddings2, labels1, labels2):
        # embeddings1: [n1, c, p], labels1: [n1]
        # embeddings2: [n2, c, p], labels2: [n2]
        embeddings1 = embeddings1.permute(2, 0, 1).contiguous().float()  # [p, n1, c]
        embeddings2 = embeddings2.permute(2, 0, 1).contiguous().float()  # [p, n2, c]

        dist = self.ComputeDistance(embeddings1, embeddings2)  # [p, n1, n2]
        mean_dist = dist.mean((1, 2))  # [p]
        
        ap_dist, an_dist = self.Convert2Triplets(labels1, labels2, dist)
        
        dist_diff = ap_dist - an_dist  # [p, num_triplets]
        loss = F.relu(dist_diff + self.margin)
        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_labels, dist):
        """
            row_labels: tensor with size [n_r] (来自 embeddings1)
            clo_labels: tensor with size [n_c] (来自 embeddings2)
            dist: [p, n_r, n_c]
        """
        matches = (row_labels.unsqueeze(1) == clo_labels.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        
        p, n_r, n_c = dist.size()
        
        ap_dist_list = []
        an_dist_list = []
        
        for i in range(n_r):
            pos_mask = matches[i]  # [n_c]
            neg_mask = diffenc[i]  # [n_c]
            
            num_pos = pos_mask.sum().item()
            num_neg = neg_mask.sum().item()
            
            if num_pos > 0 and num_neg > 0:
                ap = dist[:, i, pos_mask]  # [p, num_pos]
                an = dist[:, i, neg_mask]  # [p, num_neg]
                
                # 为这个anchor构建所有可能的 (positive, negative) 对
                ap_expanded = ap.unsqueeze(2).expand(p, num_pos, num_neg)
                an_expanded = an.unsqueeze(1).expand(p, num_pos, num_neg)
                
                ap_dist_list.append(ap_expanded.reshape(p, -1))
                an_dist_list.append(an_expanded.reshape(p, -1))
        
        if len(ap_dist_list) > 0:
            ap_dist = torch.cat(ap_dist_list, dim=1)  # [p, total_triplets]
            an_dist = torch.cat(an_dist_list, dim=1)  # [p, total_triplets]
        else:
            # 没有有效的三元组
            ap_dist = torch.zeros(p, 0).to(dist.device)
            an_dist = torch.zeros(p, 0).to(dist.device)
        
        return ap_dist, an_dist

class MultiModal_TripletLoss(BaseLoss):
    """
    Multi-modal Triplet Loss for cross-modal gait recognition.
    Computes pairwise triplet losses between all modality combinations.
    
    Args:
        margin: Margin for triplet loss
        loss_term_weight: Weight for this loss term (default: 1.0)
        combine_method: Method to combine losses from different modality pairs
                       'mean' (default) or 'sum'
    """
    def __init__(self, margin, loss_term_weight=1.0, combine_method='mean'):
        super(MultiModal_TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.combine_method = combine_method
        assert combine_method in ['mean', 'sum'], \
            "combine_method must be 'mean' or 'sum'"

    @gather_and_scale_wrapper
    def forward(self, labels, **embeddings_dict):
        """
        Args:
            labels: Label tensor [n]
            **embeddings_dict: Keyword arguments for embeddings from different modalities
                              Keys should be like 'emb_0', 'emb_1', etc.
                   
        Example:
            forward(labels=labels, emb_0=embeddings1, emb_1=embeddings2, emb_2=embeddings3)
            
        Returns:
            loss_avg: Average loss across all modality pairs
            info: Dictionary containing loss statistics
        """
        # Extract embeddings from keyword arguments and sort by key
        embeddings_keys = sorted([k for k in embeddings_dict.keys() if k.startswith('emb_')])
        embeddings_list = [embeddings_dict[k] for k in embeddings_keys]
        
        num_modalities = len(embeddings_list)
        
        if num_modalities < 2:
            raise ValueError(f"Need at least 2 modalities, got {num_modalities}")
        
        # Preprocess all embeddings: [n, c, p] -> [p, n, c]
        embeddings_list = [
            emb.permute(2, 0, 1).contiguous().float() 
            for emb in embeddings_list
        ]
        
        # Generate all pairwise combinations of modalities
        # Note: Triplet loss is asymmetric (anchor from first modality),
        # so we need to compute both (i,j) and (j,i)
        modality_pairs = []
        for i, j in combinations(range(num_modalities), 2):
            modality_pairs.append((i, j))
            modality_pairs.append((j, i))  # Add reverse direction
        
        # Storage for losses from all pairs
        all_losses = []
        all_hard_losses = []
        all_loss_nums = []
        all_mean_dists = []
        
        # Compute triplet loss for each modality pair (both directions)
        for idx, (i, j) in enumerate(modality_pairs):
            emb_i = embeddings_list[i]
            emb_j = embeddings_list[j]
            
            # Compute distance: emb_i as anchor (row), emb_j as reference (col)
            dist = self.ComputeDistance(emb_i, emb_j)  # [p, n, n]
            mean_dist = dist.mean((1, 2))  # [p]
            
            # Convert to triplets: emb_i is anchor
            ap_dist, an_dist = self.Convert2Triplets(labels, labels, dist)
            dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
            loss = F.relu(dist_diff + self.margin)
            
            hard_loss = torch.max(loss, -1)[0]
            loss_avg, loss_num = self.AvgNonZeroReducer(loss)
            
            # Store results
            all_losses.append(loss_avg)
            all_hard_losses.append(hard_loss)
            all_loss_nums.append(loss_num)
            all_mean_dists.append(mean_dist)
        
        # Combine losses from all modality pairs
        stacked_losses = torch.stack(all_losses)  # [num_pairs, p]
        
        if self.combine_method == 'mean':
            combined_loss = stacked_losses.mean(0)  # [p]
        else:  # sum
            combined_loss = stacked_losses.sum(0)  # [p]
        
        # Aggregate information
        combined_hard_loss = torch.stack(all_hard_losses).mean(0)
        combined_loss_num = torch.stack(all_loss_nums).mean(0)
        combined_mean_dist = torch.stack(all_mean_dists).mean(0)
        
        # Update info with detailed pair-wise statistics
        self.info.update({
            'loss': combined_loss.detach().clone(),
            'hard_loss': combined_hard_loss.detach().clone(),
            'loss_num': combined_loss_num.detach().clone(),
            'mean_dist': combined_mean_dist.detach().clone(),
            'num_modality_pairs': len(modality_pairs)
        })
        
        # Add per-pair statistics for debugging
        for idx, (i, j) in enumerate(modality_pairs):
            self.info[f'loss_pair_{i}_{j}'] = all_losses[idx].detach().clone()
            self.info[f'mean_dist_pair_{i}_{j}'] = all_mean_dists[idx].detach().clone()
        
        return combined_loss.mean(), self.info

    def AvgNonZeroReducer(self, loss):
        """
        Compute average of non-zero losses
        
        Args:
            loss: Loss tensor [p, num_triplets]
            
        Returns:
            loss_avg: Average loss [p]
            loss_num: Number of non-zero losses [p]
        """
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
        Compute Euclidean distance between embeddings
        
        Args:
            x: [p, n_x, c]
            y: [p, n_y, c]
            
        Returns:
            dist: [p, n_x, n_y]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
        Convert distance matrix to triplet format
        
        Args:
            row_labels: tensor with size [n_r]
            clo_label: tensor with size [n_c]
            dist: distance matrix [p, n_r, n_c]
            
        Returns:
            ap_dist: anchor-positive distances [p, n, num_pos, 1]
            an_dist: anchor-negative distances [p, n, 1, num_neg]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist


class Fast_Mask_MultiModal_TripletLoss(BaseLoss):
    """
    Multi-modal Triplet Loss for cross-modal gait recognition.
    Computes pairwise triplet losses between all modality combinations.
    
    Args:
        margin: Margin for triplet loss
        loss_term_weight: Weight for this loss term (default: 1.0)
        combine_method: Method to combine losses from different modality pairs
                       'mean' (default) or 'sum'
    """
    def __init__(self, margin, loss_term_weight=1.0, combine_method='sum'):
        super(Fast_Mask_MultiModal_TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin
        self.combine_method = combine_method
        assert combine_method in ['mean', 'sum'], \
            "combine_method must be 'mean' or 'sum'"

    @gather_and_scale_wrapper
    def forward(self, labels, **kwargs):
        """
        Args:
            labels: Label tensor [n]
            **kwargs: Keyword arguments containing:
                     - emb_0, emb_1, ..., emb_n: embeddings from different modalities
                     - modal_0_valid, modal_1_valid, ...: validity masks for each modality [n]
                   
        Example:
            forward(labels=labels, 
                   emb_0=embeddings1, emb_1=embeddings2, emb_2=embeddings3,
                   modal_0_valid=valid_mask1, modal_1_valid=valid_mask2, modal_2_valid=valid_mask3)
            
        Returns:
            loss_avg: Average loss across all modality pairs
            info: Dictionary containing loss statistics
        """
        # Extract embeddings and validity masks
        embeddings_keys = sorted([k for k in kwargs.keys() if k.startswith('emb_')])
        embeddings_list = [kwargs[k] for k in embeddings_keys]
        
        # Extract validity masks (optional)
        modal_valid_list = []
        for i in range(len(embeddings_keys)):
            valid_key = f'modal_{i}_valid'
            if valid_key in kwargs:
                modal_valid_list.append(kwargs[valid_key])
            else:
                # If not provided, assume all samples are valid
                modal_valid_list.append(torch.ones(labels.size(0), dtype=torch.bool, device=labels.device))
        
        num_modalities = len(embeddings_list)
        
        if num_modalities < 2:
            raise ValueError(f"Need at least 2 modalities, got {num_modalities}")
        
        # Preprocess all embeddings: [n, c, p] -> [p, n, c]
        embeddings_list = [
            emb.permute(2, 0, 1).contiguous().float() 
            for emb in embeddings_list
        ]
        
        # Generate all pairwise combinations of modalities
        # Note: Triplet loss is asymmetric (anchor from first modality),
        # so we need to compute both (i,j) and (j,i)
        modality_pairs = []
        for i, j in combinations(range(num_modalities), 2):
            modality_pairs.append((i, j))
            modality_pairs.append((j, i))  # Add reverse direction
        
        # Storage for losses from all pairs
        all_losses = []
        all_hard_losses = []
        all_loss_nums = []
        all_mean_dists = []
        
        # Compute triplet loss for each modality pair (both directions)
        for idx, (i, j) in enumerate(modality_pairs):
            # Get validity masks for this pair
            valid_i = modal_valid_list[i]  # [n]
            valid_j = modal_valid_list[j]  # [n]
            
            # Skip if either modality has no valid samples
            if not valid_i.any() or not valid_j.any():
                continue
            
            # Filter embeddings and labels by validity (more efficient)
            emb_i_full = embeddings_list[i]  # [p, n, c]
            emb_j_full = embeddings_list[j]  # [p, n, c]
            
            emb_i = emb_i_full[:, valid_i, :]  # [p, n_valid_i, c]
            emb_j = emb_j_full[:, valid_j, :]  # [p, n_valid_j, c]
            
            labels_i = labels[valid_i]  # [n_valid_i]
            labels_j = labels[valid_j]  # [n_valid_j]
            
            # Compute distance only for valid samples
            dist = self.ComputeDistance(emb_i, emb_j)  # [p, n_valid_i, n_valid_j]
            mean_dist = dist.mean((1, 2))  # [p]
            
            # Convert to triplets (now without validity mask since already filtered)
            ap_dist, an_dist = self.Convert2TripletsSimple(labels_i, labels_j, dist)
            
            # Check if we have valid triplets
            if ap_dist.numel() == 0 or an_dist.numel() == 0:
                continue
            
            # Compute triplet loss
            dist_diff = ap_dist - an_dist  # [p, num_triplets]
            loss = F.relu(dist_diff + self.margin)  # [p, num_triplets]
            
            hard_loss = torch.max(loss, -1)[0]
            loss_avg, loss_num = self.AvgNonZeroReducer(loss)
            
            # Store results
            all_losses.append(loss_avg)
            all_hard_losses.append(hard_loss)
            all_loss_nums.append(loss_num)
            all_mean_dists.append(mean_dist)
        
        # Combine losses from all modality pairs
        if len(all_losses) == 0:
            # No valid modality pairs, return zero loss
            p = embeddings_list[0].size(0)
            zero_loss = torch.zeros(p, device=embeddings_list[0].device)
            self.info.update({
                'loss': zero_loss.detach().clone(),
                'hard_loss': zero_loss.detach().clone(),
                'loss_num': zero_loss.detach().clone(),
                'mean_dist': zero_loss.detach().clone()
            })
            return zero_loss.mean(), self.info
            
        stacked_losses = torch.stack(all_losses)  # [num_valid_pairs, p]
        
        if self.combine_method == 'mean':
            combined_loss = stacked_losses.mean(0)  # [p]
        else:  # sum
            combined_loss = stacked_losses.sum(0)  # [p]
        
        # Aggregate information - only keep averaged statistics
        combined_hard_loss = torch.stack(all_hard_losses).mean(0)
        combined_loss_num = torch.stack(all_loss_nums).mean(0)
        combined_mean_dist = torch.stack(all_mean_dists).mean(0)
        
        # Update info with simplified statistics
        self.info.update({
            'loss': combined_loss.detach().clone(),
            'hard_loss': combined_hard_loss.detach().clone(),
            'loss_num': combined_loss_num.detach().clone(),
            'mean_dist': combined_mean_dist.detach().clone()
        })
        
        return combined_loss, self.info

    def AvgNonZeroReducer(self, loss):
        """
        Compute average of non-zero losses
        
        Args:
            loss: Loss tensor [p, num_triplets]
            
        Returns:
            loss_avg: Average loss [p]
            loss_num: Number of non-zero losses [p]
        """
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
        Compute Euclidean distance between embeddings
        
        Args:
            x: [p, n_x, c]
            y: [p, n_y, c]
            
        Returns:
            dist: [p, n_x, n_y]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2TripletsSimple(self, row_labels, clo_label, dist):
        """
        Convert distance matrix to triplet format - flattened version (simplified)
        
        Args:
            row_labels: tensor with size [n_r]
            clo_label: tensor with size [n_c]
            dist: distance matrix [p, n_r, n_c]
            
        Returns:
            ap_dist: anchor-positive distances [p, num_triplets]
            an_dist: anchor-negative distances [p, num_triplets]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        
        p, n_r, n_c = dist.size()
        
        ap_list = []
        an_list = []
        
        # For each anchor, generate all valid (anchor, positive, negative) triplets
        for i in range(n_r):
            pos_indices = torch.where(matches[i])[0]
            neg_indices = torch.where(diffenc[i])[0]
            
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            
            if num_pos > 0 and num_neg > 0:
                # Get distances for this anchor
                ap_i = dist[:, i, pos_indices]  # [p, num_pos]
                an_i = dist[:, i, neg_indices]  # [p, num_neg]
                
                # Expand to create all combinations: [p, num_pos, num_neg]
                ap_i_expanded = ap_i.unsqueeze(-1).expand(-1, -1, num_neg)  # [p, num_pos, num_neg]
                an_i_expanded = an_i.unsqueeze(1).expand(-1, num_pos, -1)  # [p, num_pos, num_neg]
                
                # Flatten the pos-neg combinations: [p, num_pos * num_neg]
                ap_list.append(ap_i_expanded.reshape(p, -1))
                an_list.append(an_i_expanded.reshape(p, -1))
        
        if len(ap_list) == 0:
            # No valid triplets
            empty_tensor = torch.tensor([], device=dist.device)
            return empty_tensor, empty_tensor
        
        # Concatenate all triplets: [p, total_num_triplets]
        ap_dist = torch.cat(ap_list, dim=1)
        an_dist = torch.cat(an_list, dim=1)
        
        return ap_dist, an_dist

    def Convert2Triplets(self, row_labels, clo_label, dist, row_valid=None, clo_valid=None):
        """
        Convert distance matrix to triplet format - flattened version with validity mask
        
        Args:
            row_labels: tensor with size [n_r]
            clo_label: tensor with size [n_c]
            dist: distance matrix [p, n_r, n_c]
            row_valid: validity mask for row (anchor) samples [n_r], optional
            clo_valid: validity mask for col (reference) samples [n_c], optional
            
        Returns:
            ap_dist: anchor-positive distances [p, num_triplets]
            an_dist: anchor-negative distances [p, num_triplets]
            triplet_mask: validity mask for each triplet [num_triplets]
                         True if both anchor and positive/negative are valid
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        
        p, n_r, n_c = dist.size()
        
        # If no validity masks provided, assume all valid
        if row_valid is None:
            row_valid = torch.ones(n_r, dtype=torch.bool, device=dist.device)
        if clo_valid is None:
            clo_valid = torch.ones(n_c, dtype=torch.bool, device=dist.device)
        
        ap_list = []
        an_list = []
        mask_list = []
        
        # For each anchor, generate all valid (anchor, positive, negative) triplets
        for i in range(n_r):
            pos_indices = torch.where(matches[i])[0]
            neg_indices = torch.where(diffenc[i])[0]
            
            num_pos = len(pos_indices)
            num_neg = len(neg_indices)
            
            if num_pos > 0 and num_neg > 0:
                # Get distances for this anchor
                ap_i = dist[:, i, pos_indices]  # [p, num_pos]
                an_i = dist[:, i, neg_indices]  # [p, num_neg]
                
                # Expand to create all combinations: [p, num_pos, num_neg]
                ap_i_expanded = ap_i.unsqueeze(-1).expand(-1, -1, num_neg)  # [p, num_pos, num_neg]
                an_i_expanded = an_i.unsqueeze(1).expand(-1, num_pos, -1)  # [p, num_pos, num_neg]
                
                # Flatten the pos-neg combinations: [p, num_pos * num_neg]
                ap_list.append(ap_i_expanded.reshape(p, -1))
                an_list.append(an_i_expanded.reshape(p, -1))
                
                # Compute validity mask for these triplets
                # A triplet is valid if: anchor is valid AND positive is valid AND negative is valid
                anchor_valid = row_valid[i]  # scalar
                pos_valid = clo_valid[pos_indices]  # [num_pos]
                neg_valid = clo_valid[neg_indices]  # [num_neg]
                
                # Expand validity: [num_pos, num_neg]
                pos_valid_expanded = pos_valid.unsqueeze(-1).expand(-1, num_neg)  # [num_pos, num_neg]
                neg_valid_expanded = neg_valid.unsqueeze(0).expand(num_pos, -1)  # [num_pos, num_neg]
                
                # Triplet is valid if all three are valid
                triplet_valid = anchor_valid & pos_valid_expanded & neg_valid_expanded  # [num_pos, num_neg]
                mask_list.append(triplet_valid.reshape(-1))  # [num_pos * num_neg]
        
        if len(ap_list) == 0:
            # No valid triplets
            empty_tensor = torch.tensor([], device=dist.device)
            empty_mask = torch.tensor([], dtype=torch.bool, device=dist.device)
            return empty_tensor, empty_tensor, empty_mask
        
        # Concatenate all triplets: [p, total_num_triplets]
        ap_dist = torch.cat(ap_list, dim=1)
        an_dist = torch.cat(an_list, dim=1)
        triplet_mask = torch.cat(mask_list, dim=0)  # [total_num_triplets]
        
        return ap_dist, an_dist, triplet_mask