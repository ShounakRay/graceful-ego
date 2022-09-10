import torch
import torch.nn as nn

from utils.model_utils import MLP_3, CG_stacked
import copy
import numpy as np
from utils.utils import get_agent_pos_from_vec
from random import choices
copy_func = copy.deepcopy


class initializer(nn.Module):
    """ A transformer model with wider latent space """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # input embedding stem
        hidden_dim = 1024
        self.CG_agent = CG_stacked(5, hidden_dim)
        self.CG_line = CG_stacked(5, hidden_dim)
        self.agent_encode = MLP_3([17, 256, 512, hidden_dim])
        self.line_encode = MLP_3([4, 256, 512, hidden_dim])
        self.type_embedding = nn.Embedding(20, hidden_dim)
        self.traf_embedding = nn.Embedding(4, hidden_dim)

        self.apply(self._init_weights)
        middle_layer_shape = [hidden_dim * 2, hidden_dim, 256]

        self.prob_head = MLP_3([*middle_layer_shape, 1])
        self.pos_head = MLP_3([*middle_layer_shape, 10 * (1 + 5)])
        self.bbox_head = MLP_3([*middle_layer_shape, 10 * (1 + 5)])
        self.heading_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])
        self.vel_heading_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])
        self.speed_head = MLP_3([*middle_layer_shape, 10 * (1 + 2)])

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def output_to_dist(self, para, n, range):
        # if n = 2, dim = 5 = 2 + 3, if n = 1, dim = 2 = 1 + 1

        if n == 2:
            loc, tril, diag = para[..., :2], para[..., 2], para[..., 3:]
            loc = torch.clip(loc, min=range[0], max=range[1])

            diag = torch.clip(1 + nn.functional.elu(diag), min=1e-4, max=5)

            z = torch.zeros([*loc.shape[:-1]], device=para.device)
            scale_tril = torch.stack([
                diag[..., 0], z,
                tril, diag[..., 1]
            ], dim=-1).view(*loc.shape[:-1], 2, 2)

            distri = torch.distributions.multivariate_normal.MultivariateNormal(loc=loc, scale_tril=scale_tril)
            return distri

        if n == 1:
            loc, scale = para[..., 0], para[..., 1]
            loc = torch.clip(loc, min=range[0], max=range[1])

            scale = torch.clip(1 + nn.functional.elu(scale), min=1e-4, max=5)
            # scale = torch.clip(scale, min=1e-4,max=5)

            distri = torch.distributions.Normal(loc, scale)

            return distri

    def feature_to_dists(self, feature, K):

        # feature dim (batch,128,2048)
        # vector distribution
        prob_pred = self.prob_head(feature).squeeze(-1)

        # position distribution： 2 dimension x y, range(-1/2,1/2)
        # pos_out dim (batch,128,10,6),
        # 10 mixtures
        # 0: mixture weight
        # 1-2: mean
        # 3-5: variance and covariance
        pos_out = self.pos_head(feature).view([*feature.shape[:-1], K, -1])
        pos_weight = pos_out[..., 0]
        pos_param = pos_out[..., 1:]
        pos_distri = self.output_to_dist(pos_param, 2, [-0.5, 0.5])
        pos_weight = torch.distributions.Categorical(logits=pos_weight)
        pos_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(pos_weight, pos_distri)

        # bbox distribution： 2 dimension length width
        bbox_out = self.bbox_head(feature).view([*feature.shape[:-1], K, -1])
        bbox_weight = bbox_out[..., 0]
        bbox_param = bbox_out[..., 1:]
        bbox_distri = self.output_to_dist(bbox_param, 2, [1, 30])
        bbox_weight = torch.distributions.Categorical(logits=bbox_weight)
        bbox_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(bbox_weight, bbox_distri)

        # heading distribution: 1 dimension,range(-pi/2,pi/2)
        heading_out = self.heading_head(feature).view([*feature.shape[:-1], K, -1])
        heading_weight = heading_out[..., 0]
        heading_param = heading_out[..., 1:]
        heading_distri = self.output_to_dist(heading_param, 1, [-np.pi / 2, np.pi / 2])
        heading_weight = torch.distributions.Categorical(logits=heading_weight)
        heading_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(heading_weight, heading_distri)

        # speed distribution: 1 dimension
        speed_out = self.speed_head(feature).view([*feature.shape[:-1], K, -1])
        speed_weight = speed_out[..., 0]
        speed_param = speed_out[..., 1:]
        speed_distri = self.output_to_dist(speed_param, 1, [0, 50])
        speed_weight = torch.distributions.Categorical(logits=speed_weight)
        speed_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(speed_weight, speed_distri)

        # vel heading distribution: 1 dimension,range(-pi/2,pi/2)
        vel_heading_out = self.vel_heading_head(feature).view([*feature.shape[:-1], K, -1])
        vel_heading_weight = vel_heading_out[..., 0]
        vel_heading_param = vel_heading_out[..., 1:]
        vel_heading_distri = self.output_to_dist(vel_heading_param, 1, [-np.pi / 2, np.pi / 2])
        vel_heading_weight = torch.distributions.Categorical(logits=vel_heading_weight)
        vel_heading_gmm = torch.distributions.mixture_same_family.MixtureSameFamily(vel_heading_weight,
                                                                                    vel_heading_distri)
        return {'prob': prob_pred, 'pos': pos_gmm, 'bbox': bbox_gmm, 'heading': heading_gmm, 'speed': speed_gmm,
                'vel_heading': vel_heading_gmm}

    def feature_extract(self, data, random_mask):
        agent = data['agent_feat'][..., :-2]

        agent_line_type = data['agent_feat'][..., -2].to(int)
        agent_line_traf = data['agent_feat'][..., -1].to(int)
        # agent_line_traf = torch.zeros_like(agent_line_traf).to(agent.device)

        agent_line_type_embed = self.type_embedding(agent_line_type)
        agent_line_traf_embed = self.traf_embedding(agent_line_traf)
        agent_mask = data['agent_mask']

        min_agent_num = self.cfg['min_agent']
        if random_mask:
            agent_mask[:, 0] = 1
            for i in range(agent_mask.shape[0]):
                masked_num = i % min_agent_num
                agent_mask[i, 1 + masked_num:] = 0

        polyline = data['lane_inp'][..., :4]
        polyline_type = data['lane_inp'][..., 4].to(int)
        polyline_traf = data['lane_inp'][..., 5].to(int)

        # polyline_traf = torch.zeros_like(polyline_traf).to(agent.device)

        polyline_type_embed = self.type_embedding(polyline_type)
        polyline_traf_embed = self.traf_embedding(polyline_traf)
        polyline_traf_embed = torch.zeros_like(polyline_traf_embed).to(agent.device)

        line_mask = data['center_mask']

        # agent features
        agent_enc = self.agent_encode(agent) + agent_line_type_embed + agent_line_traf_embed
        # map features
        line_enc = self.line_encode(polyline) + polyline_traf_embed + polyline_type_embed
        b, a, d = agent_enc.shape

        device = agent_enc.device

        context_agent = torch.ones([b, d]).to(device)
        # agent information fusion with CG block
        agent_enc, context_agent = self.CG_agent(agent_enc, context_agent, agent_mask)

        # map information fusion with CG block
        line_enc, context_line = self.CG_line(line_enc, context_agent, line_mask)
        # map context feature
        context_line = context_line.unsqueeze(1).repeat(1, line_enc.shape[1], 1)
        feature = torch.cat([line_enc, context_line], dim=-1)

        return feature

    def compute_loss(self, data, pred_dists):
        BCE = torch.nn.BCEWithLogitsLoss()
        prob_loss = BCE(pred_dists['prob'], data['gt_distribution'])

        line_mask = data['center_mask']
        prob_loss = torch.sum(prob_loss * line_mask) / max(torch.sum(line_mask), 1)
        if torch.isnan(prob_loss):
            print(data['gt_distribution'])

        gt_mask = data['gt_distribution']
        gt_sum = torch.clip(torch.sum(gt_mask, dim=1).unsqueeze(-1), min=1)
        # long lat loss
        pos_loss = -pred_dists['pos'].log_prob(data['gt_long_lat'])
        pos_loss = (torch.sum(pos_loss * gt_mask, dim=1) / gt_sum).mean()

        bbox_loss = -pred_dists['bbox'].log_prob(data['gt_bbox'])
        bbox_loss = (torch.sum(bbox_loss * gt_mask, dim=1) / gt_sum).mean()

        speed_loss = -pred_dists['speed'].log_prob(data['gt_speed'])
        speed_loss = (torch.sum(speed_loss * gt_mask, dim=1) / gt_sum).mean()

        vel_heading_loss = -pred_dists['vel_heading'].log_prob(data['gt_vel_heading'])
        vel_heading_loss = (torch.sum(vel_heading_loss * gt_mask, dim=1) / gt_sum).mean()

        heading_loss = -pred_dists['heading'].log_prob(data['gt_heading'])
        heading_loss = (torch.sum(heading_loss * gt_mask, dim=1) / gt_sum).mean()

        losses = {}

        losses['prob_loss'] = prob_loss
        losses['pos_loss'] = pos_loss
        losses['heading_loss'] = heading_loss
        losses['speed_loss'] = speed_loss
        losses['vel_heading_loss'] = vel_heading_loss
        losses['bbox_loss'] = bbox_loss

        total_loss = prob_loss + pos_loss + heading_loss + speed_loss + vel_heading_loss + bbox_loss

        return losses, total_loss

    def forward(self, data, random_mask=True):

        feature = self.feature_extract(data, random_mask)

        # Sample location, bounding box, heading and velocity.
        K = 10

        pred_dists = self.feature_to_dists(feature, K)
        losses, total_loss = self.compute_loss(data, pred_dists)

        pred = pred_dists
        pred['prob'] = nn.Sigmoid()(pred['prob'])

        return pred, total_loss, losses
