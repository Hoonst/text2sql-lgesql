#coding=utf8
import copy, math
from gettext import lgettext
import torch, dgl
import torch.nn as nn
import dgl.function as fn
from utils.dropedge import DropEdge

from model.model_utils import Registrable, FFN
from model.encoder.rgatsql import RGATLayer, MultiViewRGATLayer
from model.encoder.functions import *
from torch.cuda.amp import autocast, GradScaler
use_fp16 = True

@Registrable.register('lgesql')
class LGESQL(nn.Module):
    """ Compared with RGAT, we utilize a line graph to explicitly model the propagation among edges:
    1. aggregate info from in-edges
    2. aggregate info from src nodes
    """
    def __init__(self, args):
        super(LGESQL, self).__init__()
        self.num_layers, self.num_heads = args.gnn_num_layers, args.num_heads
        self.relation_share_heads = args.relation_share_heads
        self.graph_view = args.local_and_nonlocal
        self.ndim = args.gnn_hidden_size # node feature dim
        self.edim = self.ndim // self.num_heads if self.relation_share_heads else \
            self.ndim // 2 if self.graph_view == 'mmc' else self.ndim
        self.relation_num = args.relation_num
        self.relation_embed = nn.Embedding(self.relation_num, self.edim)
        self.gnn_layers = nn.ModuleList([
            DualRGATLayer(self.ndim, self.edim, num_heads=args.num_heads, feat_drop=args.dropout, graph_view=self.graph_view)
            for _ in range(self.num_layers)])

        self.local_transform = DropEdge(p=args.local_drop_edge_p)
        self.global_transform = DropEdge(p=args.global_drop_edge_p)

    def forward(self, x, batch):
        '''
        x: input of graph input layer > question / table / column
        batch.graph.global_edges: tensor([ 3, 27, 27,  ..., 10, 10,  9]
        > edge type

        * global_lgx:
        tensor([[-0.4086, -0.4426,  0.3816,  ..., -1.3158,  0.0257,  0.0076],
                [-0.0166,  0.4859, -0.8854,  ..., -0.4945, -0.9852, -2.0803],
                ...
        > 각 global_edges에 대한 embeddings

        * local_lgx:
        tensor([[-0.0166,  0.4859, -0.8854,  ..., -0.4945, -0.9852, -2.0803],
                [-0.0166,  0.4859, -0.8854,  ..., -0.4945, -0.9852, -2.0803],
                [-0.0166,  0.4859, -0.8854,  ..., -0.4945, -0.9852, -2.0803],
                ...
        > 각 local_edges에 대한 embeddings

        * global_g:
        Graph(num_nodes=265, num_edges=19721,
                ndata_schemes={}
                edata_schemes={})

        * local_g: 
        Graph(num_nodes=265, num_edges=7997,
                ndata_schemes={}
                edata_schemes={})

        local_g edge의 시작과 끝을 나타냄
        * src_ids: 
        tensor([  0,   0,   0,  ..., 264, 264, 264], device='cuda:0')   

        * dst_ids: 
        tensor([  5,   8,   9,  ..., 191, 192, 193], device='cuda:0')

        * global_edges:
        tensor([27, 27, 27,  ..., 10, 10,  9], device='cuda:0')

        * lg (line graph)
        Graph(num_nodes=4079, 
                num_edges=69647,      
                ndata_schemes={}      
                edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int32)})

        '''
        # copy를 안해서 발생하는 문제?
        global_lgx = self.relation_embed(batch.graph.global_edges)
        mask = batch.graph.local_mask
        local_lgx = global_lgx[mask]

        # local_lgx: global_lgx (relation embedding)중, local에 해당하는 부분

        # local_g, global_g, lg = batch.graph.local_g, batch.graph.global_g, batch.graph.lg
        local_g, global_g = batch.graph.local_g, batch.graph.global_g
        src_ids, dst_ids = batch.graph.src_ids, batch.graph.dst_ids

        if self.local_transform.p != 0.0 and self.global_transform.p != 0.0:
            global_g, global_eids_to_remove = self.global_transform(global_g)
            global_index = torch.ones(global_lgx.shape[0], dtype=bool)
            global_index[global_eids_to_remove.tolist()] = False

            # local relation이 있는 edge에서 
            # global edge와 local 부분 모두 살아있는 파트 살리자
            local_index = [True if v_1 and v_2 else False for v_1, v_2 in zip(global_index.tolist(), mask.tolist())]

            global_lgx, local_lgx = global_lgx[global_index], global_lgx[local_index]
            global_edges = batch.graph.global_edges[global_index]

            local_eids_to_remove = []
            local_eids_to_preserve = []

            local_i = -1
            
            for l,m in zip(local_index, mask.tolist()):
                # 원래 local이었으면
                if m:
                    local_i += 1
                    if not l:
                        local_eids_to_remove.append(local_i)
                    elif l:
                        local_eids_to_preserve.append(local_i)   

            '''
            local_mask는 global_index, 즉 한번 global_transform (dropedge)를 겪은 그래프에
            살아 남은 edge들 중, local edge인 애들만을 살린다.

            local_mask: 모든 global_edge들 중, dropedge 이후 원래 local 이었던 요소
            local_index: local_lgx를 필터링하기 위하여 True, False로 

            해당 global_lgx로부터 local_lgx를 만들어야 한다
            그럼 local_mask가 필요한데 local_mask는 어떻게 만드는가?
            '''

            local_g.remove_edges(local_eids_to_remove)

            src_ids, dst_ids = src_ids[local_eids_to_preserve], dst_ids[local_eids_to_preserve]   
            
            lg = local_g.line_graph(backtracking=False)
            matching_ids = list(range(17, 31))
            match_ids = [idx for idx, r in enumerate(global_edges) if r in matching_ids]
            '''
            'match'에 해당하는 relation은 제거를 해줘야 한다.
            이를 위해 relation 순서를 살펴봤을 때
            [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            에 match가 포함되어 있다.
            '''
            src, dst, eids = lg.edges(form='all', order='eid')
            eids = [e for u, v, e in zip(src.tolist(), dst.tolist(), eids.tolist()) if not (u in match_ids and v in match_ids)]
            lg = lg.edge_subgraph(eids, preserve_nodes=True).remove_self_loop().add_self_loop()
            # print(f'global_g: {global_g}')
            # print(f'local_g: {local_g}')
            # print(f'len(global_edges): {global_edges}')

            # print(f'lg: {lg}')
            # print(f'src_ids: {src_ids.shape}')
            # print(f'dst_ids: {dst_ids.shape}')
        else:
            lg = batch.graph.lg
                
        for i in range(self.num_layers):
            x, local_lgx = self.gnn_layers[i](x, local_lgx, global_lgx, local_g, global_g, lg, src_ids, dst_ids)
            
            if self.graph_view == 'msde':
                # update local edge embeddings in the global edge embeddings matrix
                global_lgx = global_lgx.masked_scatter_(mask.unsqueeze(-1), local_lgx)
        
        # x becomes nan... why?
        return x

class DualRGATLayer(nn.Module):

    def __init__(self, ndim, edim, num_heads=8, feat_drop=0.2, graph_view='mmc', local_drop_edge_p=0.0, global_drop_edge_p=0.0):
        super(DualRGATLayer, self).__init__()
        self.ndim, self.edim = ndim, edim
        self.num_heads, self.graph_view = num_heads, graph_view
        NodeRGATLayer = MultiViewRGATLayer if self.graph_view == 'mmc' else RGATLayer
        self.node_update = NodeRGATLayer(self.ndim, self.edim, self.num_heads, feat_drop=feat_drop, local_drop_edge_p=local_drop_edge_p, global_drop_edge_p=global_drop_edge_p)
        self.edge_update = EdgeRGATLayer(self.edim, self.ndim, self.num_heads, feat_drop=feat_drop, local_drop_edge_p=local_drop_edge_p, global_drop_edge_p=global_drop_edge_p)

    def forward(self, x, local_lgx, global_lgx, local_g, global_g, lg, src_ids, dst_ids):
        if self.graph_view == 'mmc':
            out_x, _ = self.node_update(x, local_lgx, global_lgx, local_g, global_g)
        elif self.graph_view == 'msde':
            out_x, _ = self.node_update(x, global_lgx, global_g)
        else:
            out_x, _ = self.node_update(x, local_lgx, local_g)

        src_x = torch.index_select(x, dim=0, index=src_ids)
        dst_x = torch.index_select(x, dim=0, index=dst_ids)

        out_local_lgx, _ = self.edge_update(local_lgx, src_x, dst_x, lg)
        return out_x, out_local_lgx

class EdgeRGATLayer(nn.Module):

    def __init__(self, edim, ndim, num_heads=8, feat_drop=0.2, local_drop_edge_p=0.0, global_drop_edge_p=0.0):
        super(EdgeRGATLayer, self).__init__()
        self.edim, self.ndim = edim, ndim
        self.num_heads = num_heads
        self.d_k = self.ndim // self.num_heads
        self.affine_q, self.affine_k, self.affine_v = nn.Linear(self.edim, self.ndim), \
            nn.Linear(self.edim, self.ndim, bias=False), nn.Linear(self.edim, self.ndim, bias=False)
        self.affine_o = nn.Linear(self.ndim, self.edim)
        self.layernorm = nn.LayerNorm(self.edim)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.ffn = FFN(self.edim)

    def forward(self, x, src_x, dst_x, g):
        q, k, v = self.affine_q(self.feat_dropout(x)), self.affine_k(self.feat_dropout(x)), self.affine_v(self.feat_dropout(x))
        with g.local_scope():
            g.ndata['q'] = (q + src_x).view(-1, self.num_heads, self.d_k)
            g.ndata['k'] = k.view(-1, self.num_heads, self.d_k)
            g.ndata['v'] = (v + dst_x).view(-1, self.num_heads, self.d_k)
            out_x = self.propagate_attention(g)
        out_x = self.layernorm(x + self.affine_o(out_x.view(-1, self.num_heads * self.d_k)))
        out_x = self.ffn(out_x)
        return out_x, (src_x, dst_x)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.ndata['o']
        return out_x
