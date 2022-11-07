#coding=utf8
import dgl, math, torch

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func

def src_sum_edge_mul_dst(src_field, dst_field, e_field, out_field):
    def func(edges):
        '''
        - src_field: k
        - dst_field: q
        - e_field: e
        - out_field: score

        '''
        return {out_field: ((edges.src[src_field] + edges.data[e_field]) * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}

    return func

def src_sum_edge_mul_edge(src_field, e_field1, e_field2, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] + edges.data[e_field1]) * edges.data[e_field2]}

    return func

def div_by_z(in_field, norm_field, out_field):
    def func(nodes):
        # return {out_field: nodes.data[in_field] / (nodes.data[norm_field]+1e-6)}
        # 해당 line에 1e-6을 더하지 않으면 forward pass 중에 nan으로 입력이 변함
        return {out_field: nodes.data[in_field] / (nodes.data[norm_field]+1e-6)}

    return func
