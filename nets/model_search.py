import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many
import copy
from problems.vrp.problem_vrp import CVRP
from problems.tsp.problem_tsp import TSP
from problems.op.problem_op import OP
from problems.vrp.problem_vrp import SDVRP

from problems.pctsp.problem_pctsp import PCTSPDet
from problems.pctsp.problem_pctsp import PCTSPStoch

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 n_paths=None,
                 n_EG=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_tsp = problem.NAME == 'tsp'
        self.allow_partial = problem.NAME == 'sdvrp'
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_op = self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        if self.is_pctsp:
            self.stochastic = problem.stochastic

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.n_paths = n_paths
        self.n_EG = n_EG

        # Problem specific context parameters (placeholder and step context dimension)
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            # Embedding of last node + remaining_capacity / remaining length / remaining prize to collect
            step_context_dim = embedding_dim + 1

            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embedding_dim)
            
            if self.is_vrp and self.allow_partial:  # Need to include the demand if split delivery allowed
                self.project_node_step = nn.Linear(1, 3 * embedding_dim, bias=False)
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim  # Embedding of first and last node
            node_dim = 2  # x, y
            
            # Learned input symbols for first action
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)  # Placeholder should be in range of activations

        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = [nn.Linear(embedding_dim, 3 * embedding_dim, bias=False) for i in range(self.n_paths)]
        self.project_fixed_context = [nn.Linear(embedding_dim, embedding_dim, bias=False) for i in range(self.n_paths)]
        self.project_step_context = [nn.Linear(step_context_dim, embedding_dim, bias=False) for i in range(self.n_paths)]
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = [nn.Linear(embedding_dim, embedding_dim, bias=False) for i in range(self.n_paths)]
        
        self.project_node_embeddings = nn.ModuleList(self.project_node_embeddings)
        self.project_fixed_context = nn.ModuleList(self.project_fixed_context)
        self.project_step_context = nn.ModuleList(self.project_step_context)
        self.project_out = nn.ModuleList(self.project_out)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, opts=None, baseline=None, bl_val=None, n_EG=None, return_pi=False, return_kl=False, beam_size=None, fst=None):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        assert fst==0 or fst==1

        assert beam_size!=None

        if self.is_vrp and n_EG:
            n_EG = n_EG
        else:
            n_EG = self.n_EG

        embeddings_init, init_context, attn, V, h_old = self.embedder(self._init_embed(input))

        batch_size, graph_size, _ = embeddings_init.size()
        expand_size = min(graph_size, beam_size)
        expand_size = 3
        if self.is_vrp:
            demand = input['demand']
            loc_with_depot = torch.cat((input['depot'][:, None, :], input['loc']), 1)
        if self.is_op:
            loc_with_depot = torch.cat((input['depot'][:, None, :], input['loc']), 1)
            max_length=input["max_length"][:, None] - (loc_with_depot[:, 0:1] - loc_with_depot).norm(p=2, dim=-1)
            prize_with_depot = torch.cat((torch.zeros_like(input['prize'][:, :1]), input['prize']), 1)
        if self.is_pctsp:
            loc_with_depot = torch.cat((input['depot'][:, None, :], input['loc']), 1)
            penalty_with_depot = torch.cat((torch.zeros_like(input['penalty'][:, :1]), input['penalty']), 1)
            expected_prize = input['deterministic_prize']
            real_prize = input['stochastic_prize' if self.stochastic else 'deterministic_prize']
            real_prize_with_depot = torch.cat((torch.zeros_like(real_prize[:, :1]), real_prize), -1)

        costs = []
        seq_beam_list = []

        for i in range(self.n_paths):
            seq_beam = torch.zeros(batch_size, 1, 1, dtype=torch.long, device=embeddings_init.device)
            ll_beam = torch.zeros(batch_size, 1, device=embeddings_init.device)
            distance_beam = torch.zeros(batch_size, 1, device=embeddings_init.device)
            fixed_beam = self._precompute(embeddings_init, path_index=i)
            mask_beam = torch.zeros(batch_size, 1, graph_size, dtype=torch.uint8, device=embeddings_init.device) > 0
            if self.is_vrp:
                used_beam = demand.new_zeros(batch_size, 1)
                visited_beam = torch.zeros(batch_size, 1, graph_size, dtype=torch.uint8, device=embeddings_init.device)
                if self.allow_partial:
                    demand_with_depot = torch.cat((demand.new_zeros(batch_size, 1), demand[:, :]), 1)[:, None, :]
                    mask_beam[:, :, 0] = True
            if self.is_op:
                visited_beam = torch.zeros(batch_size, 1, graph_size, dtype=torch.uint8, device=embeddings_init.device)
                coord_all = loc_with_depot.view(batch_size, 1, -1, 2)
                coord_cur = loc_with_depot[torch.arange(batch_size).view(-1, 1), seq_beam[:, :, -1]].view(batch_size, seq_beam.size(1), 1, -1)
                mask_beam = ((coord_all - coord_cur).norm(p=2, dim=-1)) > max_length.view(batch_size, 1, -1)
            if self.is_pctsp:
                visited_beam = torch.zeros(batch_size, 1, graph_size, dtype=torch.uint8, device=embeddings_init.device)
                prize_beam = penalty_with_depot.new_zeros(batch_size, 1)
                mask_beam[:, :, 0] = True

            j = 0

            while (self.is_op and (j == 0 or not (seq_beam[:, :, -1][ll_beam!=float("-inf")] == 0).all())) \
                    or (not self.is_vrp and not self.is_op and not self.is_pctsp and not mask_beam[ll_beam!=float("-inf")].all()) \
                    or (self.is_vrp and not self.allow_partial and not visited_beam[ll_beam!=float("-inf")].all()) \
                    or (self.is_vrp and self.allow_partial and (j < loc_with_depot.size(1) or (demand_with_depot[ll_beam!=float("-inf")] > 0).any())) \
                    or (self.is_pctsp and (j == 0 or not (seq_beam[:, :, -1][ll_beam!=float("-inf")] == 0).all())):

                assert (ll_beam.max(-1)[0]!=float("-inf")).all()
                ll_beam_new = []
                fixed_beam_new = []
                node_index_beam = []

                n_beam = distance_beam.size(-1)
                distance_beam = distance_beam.view(batch_size, n_beam, -1)

                for beam_index in range(ll_beam.size(1)):
                    ll = ll_beam[:, beam_index]
                    seq = seq_beam[:, beam_index, :]
                    mask = mask_beam[:, beam_index, :].view(batch_size, 1, -1)
                    fixed = fixed_beam[beam_index * batch_size:(beam_index + 1) * batch_size]
                    if self.is_vrp:
                        if self.allow_partial:
                            log_p = self._get_log_p(fixed, mask, seq, i, used=used_beam[:, beam_index], demand_with_depot=demand_with_depot[:, beam_index:beam_index+1, :])[:, 0, :]
                        else:
                            log_p = self._get_log_p(fixed, mask, seq, i, used=used_beam[:, beam_index])[:, 0, :]
                        # log_p[(ll_beam[:, beam_index]!=float("-inf")) & (visited_beam[:, beam_index].all(dim=-1)), 0] = 0
                    elif self.is_op:
                        remaining_beam = input["max_length"].view(batch_size, 1) - distance_beam[:, beam_index]

                        log_p = self._get_log_p(fixed, mask, seq, i, remaining = remaining_beam)[:, 0, :]
                    elif self.is_pctsp: 
                        remaining_beam = torch.clamp(1 - prize_beam[:, beam_index, None], min = 0)
                        mask[ll==float("-inf"), :, 0] = 0
                        log_p = self._get_log_p(fixed, mask, seq, i, remaining = remaining_beam)[:, 0, :]
                    else:
                        log_p = self._get_log_p(fixed, mask, seq, i)[:, 0, :]

                    log_p, node_index = torch.topk(log_p, expand_size)

                    ll_beam_new.append(ll.view(batch_size, 1) + log_p)
                    node_index_beam.append(node_index)

                node_index_beam = torch.cat(node_index_beam, dim=-1)
                ll_beam_new = torch.cat(ll_beam_new, dim=-1)

                if self.is_vrp or self.is_op or self.is_pctsp:
                    distance_beam = distance_beam + (
                            loc_with_depot[torch.arange(batch_size).view(-1, 1), seq_beam[:, :, -1]].view(batch_size, n_beam, 1, -1)
                            - loc_with_depot[torch.arange(batch_size).view(-1, 1), node_index_beam].view(batch_size, n_beam, expand_size, -1)
                            ).norm(p=2, dim=-1)
                else:
                    distance_beam = distance_beam + (
                            input[torch.arange(batch_size).view(-1, 1), seq_beam[:, :, -1]].view(batch_size, n_beam, 1, -1)
                            - input[torch.arange(batch_size).view(-1, 1), node_index_beam].view(batch_size, n_beam, expand_size, -1)
                            ).norm(p=2, dim=-1)
                distance_beam = distance_beam.view(batch_size, -1)

                if self.is_vrp:
                    if self.allow_partial:
                        used_beam_last = used_beam
                        selected_demand = demand_with_depot.view(-1, graph_size)[torch.arange(batch_size * demand_with_depot.size(1)).view(-1, 1), node_index_beam.view(batch_size * demand_with_depot.size(1), -1)].view(batch_size, -1)
                    else:
                        selected_demand = demand[torch.arange(batch_size).view(-1, 1), torch.clamp(node_index_beam - 1, 0, graph_size - 1)]
                    used_beam = torch.min((used_beam.view(batch_size, n_beam, 1).expand(batch_size, n_beam, expand_size).contiguous().view(batch_size, -1) + selected_demand) * (node_index_beam != 0).float(), self.problem.VEHICLE_CAPACITY * torch.ones_like(selected_demand))
                if fst == 1 and j > 0:
                    distance_beam[ll_beam_new==float("-inf")] = 99999
                    ll_beam_new1 = ll_beam_new + 0.0
                    if self.is_vrp or self.is_op or self.is_pctsp:
                        if self.allow_partial:
                            b_index = (demand_with_depot.view(batch_size, 1, n_beam, -1)==demand_with_depot.view(batch_size, n_beam, 1, -1)).all(dim=-1)
                        else:
                            b_index = (visited_beam.view(batch_size, 1, n_beam, -1)==visited_beam.view(batch_size, n_beam, 1, -1)).all(dim=-1)
                        b_index[:, torch.arange(n_beam), torch.arange(n_beam)] = 0

                        b_index = b_index.view(batch_size, n_beam, 1, n_beam, 1).expand(-1, -1, expand_size, -1, expand_size)
                        b_index = b_index.contiguous().view(batch_size, n_beam * expand_size, n_beam * expand_size)
                        n_index = (node_index_beam.view(batch_size, 1, -1) == node_index_beam.view(batch_size, -1, 1))
                        n_index[:, torch.arange(n_beam * expand_size), torch.arange(n_beam * expand_size)] = 0
                        bn_index = b_index & n_index

                        index1 = bn_index & (distance_beam.view(batch_size, n_beam * expand_size, 1) >= distance_beam.view(batch_size, 1, n_beam * expand_size))
                        index1[torch.tril(index1 & index1.transpose(-2, -1))] = 0
                        if self.is_vrp:
                            index1 = index1 & (used_beam.view(batch_size, n_beam * expand_size, 1) >= used_beam.view(batch_size, 1, n_beam * expand_size))
                    else:
                        b_index = (mask_beam.view(batch_size, 1, n_beam, -1)==mask_beam.view(batch_size, n_beam, 1, -1)).all(dim=-1)
                        b_index[:, torch.arange(n_beam), torch.arange(n_beam)] = 0
                        f_index = (seq_beam[:, :, 1].view(batch_size, 1, -1) == seq_beam[:, :, 1].view(batch_size, -1, 1))
                        b_index = b_index & f_index

                        b_index = b_index.view(batch_size, n_beam, 1, n_beam, 1).expand(-1, -1, expand_size, -1, expand_size)
                        b_index = b_index.contiguous().view(batch_size, n_beam * expand_size, n_beam * expand_size)
                        n_index = (node_index_beam.view(batch_size, 1, -1) == node_index_beam.view(batch_size, -1, 1))
                        n_index[:, torch.arange(n_beam * expand_size), torch.arange(n_beam * expand_size)] = 0

                        bn_index = b_index & n_index

                        index1 = bn_index & (distance_beam.view(batch_size, n_beam * expand_size, 1) >= distance_beam.view(batch_size, 1, n_beam * expand_size))
                        index1[torch.tril(index1 & index1.transpose(-2, -1))] = 0
                    index11 = index1.sum(1) > 0
                    index12 = index1.sum(-1) > 0
                    index2 = index1.float()

                    index2[index2 == 0] = float("-inf")
                    index2[index2 > 0] = 0

                    l_t = ll_beam_new1[index11] + 0.0
                    l_t[l_t!=float("-inf")] = 0.0
                    ll_beam_new1[index11] = l_t + torch.max((ll_beam_new.view(batch_size, n_beam * expand_size, 1) + index2).max(-2)[0][index11], ll_beam_new[index11])

                    ll_beam_new1[index12] = float("-inf")

                    ll_beam_new = ll_beam_new1

                ll_beam, top_index = torch.topk(ll_beam_new, min(beam_size, ll_beam_new.size(1)))
                node_to_add = node_index_beam.gather(1, top_index)

                mask_beam = mask_beam[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                seq_beam = seq_beam[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                seq_beam = torch.cat([seq_beam, node_to_add.view(batch_size, -1, 1)], dim=-1)

                distance_beam = distance_beam[torch.arange(batch_size).view(-1, 1), top_index]

                j += 1
                if j > 1 and j % n_EG == 0:
                    fixed_beam = []
                    for beam_index in range(mask_beam.size(1)):
                        mask_attn = mask_beam[:, beam_index, :]
                        if self.is_pctsp:
                            mask_attn[torch.arange(batch_size), seq_beam[:, beam_index, 0]] = mask_attn[torch.arange(batch_size), seq_beam[:, beam_index, 0]] == 0
                        embeddings, init_context = self.embedder.change(attn, V, h_old, mask_attn, self.is_tsp)
                        fixed_beam.append(self._precompute(embeddings, path_index=i))

                    fixed_beam = AttentionModelFixed(
                        node_embeddings=torch.cat([fixed.node_embeddings for fixed in fixed_beam], dim=0),
                        context_node_projected=torch.cat([fixed.context_node_projected for fixed in fixed_beam], dim=0),
                        glimpse_key=torch.cat([fixed.glimpse_key for fixed in fixed_beam], dim=1),  # dim 0 are the heads
                        glimpse_val=torch.cat([fixed.glimpse_val for fixed in fixed_beam], dim=1),  # dim 0 are the heads
                        logit_key=torch.cat([fixed.logit_key for fixed in fixed_beam], dim=0)
                    )

                else:
                    fixed_index = top_index // expand_size * batch_size + torch.arange(batch_size, device=top_index.device).view(-1, 1)
                    fixed_index = fixed_index.transpose(dim0=1,dim1=0).contiguous().view(-1)
                    fixed_beam = fixed_beam[fixed_index]

                if self.is_op:
                    visited_beam = visited_beam[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                    visited_beam = visited_beam.scatter(-1, seq_beam[:, :, -1, None], 1)
                    coord_all = loc_with_depot.view(batch_size, 1, -1, 2)
                    coord_cur = loc_with_depot[torch.arange(batch_size).view(-1, 1), seq_beam[:, :, -1]].view(batch_size, seq_beam.size(1), 1, -1)
                    exceeds_length = (distance_beam.view(batch_size, -1, 1) + (coord_all - coord_cur).norm(p=2, dim=-1)) > max_length.view(batch_size, 1, -1)
                    visited_ = visited_beam.to(exceeds_length.dtype)
                    mask_beam = visited_ | visited_[:, :, 0:1] | exceeds_length
                    mask_beam[:, :, 0] = 0
                elif self.is_pctsp:
                    prize_beam = prize_beam[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                    prize_to_collect = real_prize_with_depot[torch.arange(batch_size).view(-1, 1), seq_beam[:, :, -1]]
                    prize_beam = prize_beam + prize_to_collect
                    visited_beam = visited_beam[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                    visited_beam = visited_beam.scatter(-1, seq_beam[:, :, -1, None], 1)
                    mask_beam = visited_beam | visited_beam[:, :, 0:1]
                    mask_beam[:, :, 0] = (prize_beam < 1.) & (visited_beam[:, :, 1:].int().sum(-1) < visited_beam[:, :, 1:].size(-1))
                    mask_beam = mask_beam > 0
                elif self.is_vrp and self.allow_partial:
                    used_beam = used_beam[torch.arange(batch_size).view(-1, 1), top_index]
                    used_beam_last = used_beam_last[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                    demand_with_depot = demand_with_depot[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                    selected_demand = selected_demand.gather(1, top_index)
                    delivered_demand = torch.min(selected_demand, self.problem.VEHICLE_CAPACITY - used_beam_last)
                    demand_with_depot = demand_with_depot.scatter(
                            -1,
                            seq_beam[:, :, -1, None],
                            demand_with_depot.gather(-1, seq_beam[:, :, -1, None]) - delivered_demand[:, :, None]
                            )


                    mask_loc = (demand_with_depot[:, :, 1:] == 0) | (used_beam[:, :, None] >= self.problem.VEHICLE_CAPACITY)
                    mask_depot = (seq_beam[:, :, -1] == 0) & ((mask_loc == 0).int().sum(-1) > 0)
                    mask_beam = torch.cat((mask_depot[:, :, None], mask_loc), -1)
                elif self.is_vrp and not self.allow_partial:
                    used_beam = used_beam[torch.arange(batch_size).view(-1, 1), top_index]
                    visited_beam = visited_beam[torch.arange(batch_size).view(-1, 1), top_index // expand_size]
                    visited_beam = visited_beam.scatter(-1, seq_beam[:, :, -1, None], 1)
                    mask_beam = ((visited_beam[:, :, 1:] > 0)
                           | (demand.view(batch_size, 1, graph_size - 1) + used_beam.view(batch_size, -1, 1)
                               > self.problem.VEHICLE_CAPACITY))
                    mask_depot = (seq_beam[:, :, -1] == 0) & ((mask_beam == 0).int().sum(-1) > 0)
                    mask_beam = torch.cat((mask_depot[:, :, None], mask_beam), -1)
                else:
                    mask_beam = mask_beam.scatter(-1, seq_beam[:, :, -1, None], 1)

            for beam_index in range(beam_size):
                if not self.is_vrp and not self.is_op and not self.is_pctsp:
                    inf_mask = (ll_beam[:, beam_index] == float("-inf"))
                    d = input.gather(1, seq_beam[:, beam_index, 1:].unsqueeze(-1).expand_as(input))
                    costs.append((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1))
                    costs[-1][inf_mask] = float("inf")
                elif self.is_op:
                    inf_mask = (ll_beam[:, beam_index] == float("-inf"))
                    p = -prize_with_depot.gather(1, seq_beam[:, beam_index])
                    costs.append(p.sum(1))
                    costs[-1][inf_mask] = float("inf")
                elif self.is_pctsp:
                    inf_mask = (ll_beam[:, beam_index] == float("-inf"))
                    d = loc_with_depot.gather(1, seq_beam[:, beam_index, 1:, None].expand(batch_size, seq_beam.size(2) - 1, loc_with_depot.size(-1)))
                    costs.append((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1)
                            + (d[:, 0] - input['depot']).norm(p=2, dim=1)
                            + (d[:, -1] - input['depot']).norm(p=2, dim=1))
                    pen = penalty_with_depot.gather(1, seq_beam[:, beam_index, 1:])
                    costs[-1] = costs[-1] + input["penalty"].sum(-1) - pen.sum(-1)
                    costs[-1][inf_mask] = float("inf")
                else:
                    inf_mask = (ll_beam[:, beam_index] == float("-inf"))
                    d = loc_with_depot.gather(1, seq_beam[:, beam_index, 1:, None].expand(batch_size, seq_beam.size(2) - 1, loc_with_depot.size(-1)))
                    costs.append((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) 
                            + (d[:, 0] - input['depot']).norm(p=2, dim=1)
                            + (d[:, -1] - input['depot']).norm(p=2, dim=1))
                    costs[-1][inf_mask] = float("inf")

            seq_beam_list.append(seq_beam)

        costs = torch.stack(costs, 1)

        if self.is_vrp or self.is_op or self.is_pctsp:
            max_length = max([seq_beam.size(-1) for seq_beam in seq_beam_list])
            seq_beam_list = [torch.cat([seq_beam, torch.zeros(batch_size, beam_size, max_length-seq_beam.size(-1), device=seq_beam.device, dtype=seq_beam.dtype)], dim=-1) for seq_beam in seq_beam_list]
        seq_beam_list = torch.cat(seq_beam_list, 1)

        cost, index = costs.min(1)
        seq = seq_beam_list[torch.arange(batch_size), index]

        if self.is_vrp:
            if self.allow_partial:
                p = SDVRP()
                p, _ = p.get_costs(input, seq[:, 1:])
            else:
                p = CVRP()
                p, _ = p.get_costs(input, seq[:, 1:])
        elif self.is_op:
            p = OP()
            p, _ = p.get_costs(input, seq[:, 1:])
        elif self.is_pctsp:
            if not self.stochastic:
                p = PCTSPDet()
            else:
                p = PCTSPStoch()
            p, _ = p.get_costs(input, seq[:, 1:])
        else:
            p = TSP()
            p, _ = p.get_costs(input, seq[:, 1:])

        return None, p

    def beam_search(self, *args, **kwargs):
        return self.problem.beam_search(*args, **kwargs, model=self)

    def precompute_fixed(self, input):
        embeddings, _ = self.embedder(self._init_embed(input))
        # Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do
        # the lookup once... this is the case if.sum()) all elements in the batch have maximum batch size
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        # First dim = batch_size * cur_beam_size
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        assert log_p_topk.size(1) == 1, "Can only have single step"
        # This will broadcast, calculate log_p (score) of expansions
        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]

        # We flatten the action as we need to filter and this cannot be done in 2d
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10  # != -math.inf triggers

        # Parent is row idx of ind_topk, can be found by enumerating elements and dividing by number of columns
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)

        # Filter infeasible
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            # Too bad, no feasible expansions at all :(
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]

        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):

        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        # TSP
        return self.init_embed(input)

    def sample_many(self, input, batch_rep=1, iter_rep=1):
        """
        :param input: (batch_size, graph_size, node_dim) input node features
        :return:
        """
        # Bit ugly but we need to pass the embeddings as well.
        # Making a tuple will not work with the problem.get_cost function
        return sample_many(
            lambda input: self._inner(*input),  # Need to unpack tuple into arguments
            lambda input, pi: self.problem.get_costs(input[0], pi),  # Don't need embeddings as input to get_costs
            (input, self.embedder(self._init_embed(input))[0]),  # Pack input with embeddings (additional input)
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1, path_index=None):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)

        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context[path_index](graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings[path_index](embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        fixed = AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)
        return fixed

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)

        # Return topk
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)

        # Return all, note different from torch.topk this does not give error if less than k elements along dim
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, mask, seq, path_index, used=None, remaining=None, demand_with_depot=None, normalize=True):
        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context[path_index](self._get_parallel_step_context(fixed.node_embeddings, seq, used, remaining))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, demand_with_depot)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, path_index)



        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p

    def _get_parallel_step_context(self, embeddings, seq, used, remaining, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)
        
        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        #current_node = state.get_current_node()
        #batch_size, num_steps = current_node.size()

        current_node = seq[:, -1, None]
        batch_size = seq.size(0)
        num_steps = seq.size(1) - 1

        if self.is_vrp:
            # Embedding of previous node + remaining capacity
            if from_depot:
                # 1st dimension is node idx, but we do not squeeze it since we want to insert step dimension
                # i.e. we actually want embeddings[:, 0, :][:, None, :] which is equivalent
                return torch.cat(
                    (
                        embeddings[:, 0:1, :].expand(batch_size, num_steps, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, None])
                    ),
                    -1
                )
            else:
                num_steps += 1
                return torch.cat(
                    (
                        torch.gather(
                            embeddings,
                            1,
                            current_node.contiguous()
                                .view(batch_size, 1, 1)
                                .expand(batch_size, 1, embeddings.size(-1))
                        ).view(batch_size, 1, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - used[:, None, None]
                    ),
                    -1
                )
        elif self.is_orienteering or self.is_pctsp:
            return torch.cat(
                (
                    torch.gather(
                        embeddings,
                        1,
                        current_node.contiguous()
                            .view(batch_size, 1, 1)
                            .expand(batch_size, 1, embeddings.size(-1))
                    ).view(batch_size, 1, embeddings.size(-1)),
                    (
                        remaining[:, :, None]
                    )
                ),
                -1
            )
        else:  # TSP
        

            if num_steps == 0:
                # First and only step, ignore prev_a (this is a placeholder)
                return self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1))
            else:
                return embeddings.gather(
                    1,
                    torch.cat((seq[:, 1, None], current_node), 1)[:, :, None].expand(batch_size, 2, embeddings.size(-1))
                ).view(batch_size, 1, -1)
            # More than one step, assume always starting with first
            embeddings_per_step = embeddings.gather(
                1,
                current_node[:, 1:, None].expand(batch_size, num_steps - 1, embeddings.size(-1))
            )
            return torch.cat((
                # First step placeholder, cat in dim 1 (time steps)
                self.W_placeholder[None, None, :].expand(batch_size, 1, self.W_placeholder.size(-1)),
                # Second step, concatenate embedding of first with embedding of current/previous (in dim 2, context dim)
                torch.cat((
                    embeddings_per_step[:, 0:1, :].expand(batch_size, num_steps - 1, embeddings.size(-1)),
                    embeddings_per_step
                ), 2)
            ), 1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, path_index):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        assert not torch.isnan(glimpse_Q).any()
        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))



        assert not torch.isnan(glimpse_K).any()

        assert not torch.isnan(compatibility).any()
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
        assert not torch.isnan(compatibility).any()
        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)
        assert not torch.isnan(F.softmax(compatibility, dim=-1)).any()
        assert not torch.isnan(glimpse_V).any()
        assert not torch.isnan(heads).any()
        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out[path_index](
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        assert not torch.isnan(final_Q).any()
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        assert not torch.isnan(logits).any()
        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = F.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, demand_with_depot):

        if self.is_vrp and self.allow_partial:

            # Need to provide information of how much each node has already been served
            # Clone demands as they are needed by the backprop whereas they are updated later
            glimpse_key_step, glimpse_val_step, logit_key_step = \
                self.project_node_step(demand_with_depot[:, :, :, None].clone()).chunk(3, dim=-1)

            # Projection of concatenation is equivalent to addition of projections but this is more efficient
            return (
                fixed.glimpse_key + self._make_heads(glimpse_key_step),
                fixed.glimpse_val + self._make_heads(glimpse_val_step),
                fixed.logit_key + logit_key_step,
            )

        # TSP or VRP without split delivery
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )
