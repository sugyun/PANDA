# Adapted from PANCDR: https://github.com/DMCB-GIST/PANCDR
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.init as init
from typing import Optional

class GraphConv(nn.Module):
    def __init__(self,
                 in_channels,units,step_num=1):
        super(GraphConv,self).__init__()
        self.weight = Parameter(torch.empty((in_channels, units)))
        self.bias = Parameter(torch.empty(units))
        self.step_num = step_num
        self.reset_parameters()
        
    def reset_parameters(self)->None:
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)
        
    def _get_walked_edges(self, edges, step_num):
        if step_num <= 1:
            return edges
        deeper = self._get_walked_edges(torch.matmul(edges,edges), step_num//2) ##?
        if step_num %2 == 1:
            deeper += edges
        return torch.gt(deeper, 0.0) 
        
    def _forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        features, edges = input
        outputs = torch.matmul(features, weight) + bias
        if self.step_num > 1:
            edges = self._get_walked_edges(edges, self.step_num)
        outputs = torch.matmul(edges.permute(0,2,1),outputs)
        return outputs.permute(0,2,1)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._forward(input, self.weight, self.bias)
        
    def get_config(self):
        config = {'in_channels': self.inchannels, ' units':self.units}
        base_config = super(GraphConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DeepCDR(nn.Module):
    def __init__(self, input_dim, nz, in_channels, units_list, h_dim=100, n_hidden=256, is_regr=False):
        super(DeepCDR, self).__init__()
        self.nz = nz
        self.h_dim=h_dim
        self.fe = nn.Sequential(nn.Linear(input_dim, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.Tanh(),
                                nn.Dropout(0.1),
                                nn.Linear(n_hidden,nz),
                                nn.ReLU())
        self.fe.apply(self.init_weights)

        self.GC1 = GraphConv(in_channels,units=units_list[0],step_num=1)
        self.GC2 = GraphConv(units_list[0],units=units_list[1],step_num=1)
        self.GC3 = GraphConv(units_list[1],units=units_list[2],step_num=1)
        self.GC4 = GraphConv(units_list[2],units=h_dim,step_num=1)
        RBD1 = [nn.ReLU(),nn.BatchNorm1d(units_list[0]),nn.Dropout(0.1)]
        RBD2 = [nn.ReLU(),nn.BatchNorm1d(units_list[1]),nn.Dropout(0.1)]
        RBD3 = [nn.ReLU(),nn.BatchNorm1d(units_list[2]),nn.Dropout(0.1)]
        RBD4 = [nn.ReLU(),nn.BatchNorm1d(h_dim),nn.Dropout(0.1)]
        self.RBD1 = nn.Sequential(*RBD1)
        self.RBD2 = nn.Sequential(*RBD2)
        self.RBD3 = nn.Sequential(*RBD3)
        self.RBD4 = nn.Sequential(*RBD4)

        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(nn.Linear(nz+h_dim,300),
                                    nn.Dropout(0.1))
        self.CONV = nn.Sequential(nn.Conv2d(1, 30, kernel_size=(150,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2,1)),
                                  nn.Conv2d(30, 10, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Conv2d(10, 5, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Dropout(0.1),
                                  nn.Flatten(),
                                  nn.Dropout(0.2))
                                  
        if is_regr:
            self.fc = nn.Linear(30,1)
        else:
            self.fc = nn.Sequential(nn.Linear(30,1),
                                    nn.Sigmoid())
        self.Linear.apply(self.init_weights)
        self.CONV.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        
    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
   
    def forward(self, drug_feat, drug_adj, gexpr):
        GCN_layer = self.RBD1(self.GC1([drug_feat, drug_adj])).permute(0,2,1)
        GCN_layer = self.RBD2(self.GC2([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.RBD3(self.GC3([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.RBD4(self.GC4([GCN_layer, drug_adj]))
        GCN_layer = self.Pool(GCN_layer)
        x_drug = GCN_layer.view(GCN_layer.shape[0],-1)
        x_gexpr = self.fe(gexpr)
        x = torch.cat((x_drug,x_gexpr),1)
        x = self.Linear(x)
        x = x.view(-1,1,300,1)
        x = self.CONV(x)
        output = self.fc(x)
        return output
    

class FE(nn.Module):
    def __init__(self, n_input, nz, device, n_hidden=256):
        super(FE, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.device = device
        
        encoder = [nn.Linear(self.n_input,self.n_hidden),
                   nn.BatchNorm1d(self.n_hidden),
                   nn.ReLU()]
        n_layers = 3
        for i in range(n_layers):
            encoder += [nn.Linear(self.n_hidden,self.n_hidden),
                        nn.BatchNorm1d(self.n_hidden),
                        nn.ReLU()]
        encoder += [nn.Linear(self.n_hidden,self.n_hidden)]
        self.encoder = nn.Sequential(*encoder)
        
        self.fc = nn.Linear(self.n_hidden, self.nz)
        
    def forward(self, x):
        return self.fc(self.encoder(x))


class Encoder(nn.Module):
    def __init__(self, n_input, nz, device, n_hidden=256):
        super(Encoder, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.device = device
        
        encoder = [nn.Linear(self.n_input,self.n_hidden),
                   nn.BatchNorm1d(self.n_hidden),
                   nn.ReLU()]
        n_layers = 3
        for i in range(n_layers):
            encoder += [nn.Linear(self.n_hidden,self.n_hidden),
                        nn.BatchNorm1d(self.n_hidden),
                        nn.ReLU()]
        encoder += [nn.Linear(self.n_hidden,self.n_hidden)]
        self.encoder = nn.Sequential(*encoder)
        
        self.fc1 = nn.Linear(self.n_hidden, self.nz)
        self.fc2 = nn.Linear(self.n_hidden, self.nz)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar) if self.training else mu
        return z, mu, logvar
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)

        return eps * std + mu

    
class Decoder(nn.Module):
    def __init__(self, n_output, nz, n_hidden=256):
        super(Decoder,self).__init__()
        decoder = [nn.Linear(nz,n_hidden),
                   nn.BatchNorm1d(n_hidden),
                   nn.ReLU()]
        
        n_layers = 3
        for i in range(n_layers):
            decoder += [nn.Linear(n_hidden,n_hidden),
                        nn.BatchNorm1d(n_hidden),
                        nn.ReLU()]
        decoder += [nn.Linear(n_hidden,n_output)]
        self.decoder = nn.Sequential(*decoder)
        
    def forward(self,z):
        res = self.decoder(z)
        return res
    
    
        
class GCN(nn.Module):
    def __init__(self,in_channels,units_list, h_dims=[100,200], use_dropout=False, is_regr=False): #units_list = [256,256,256]
        super(GCN, self).__init__()
        self.GC1 = GraphConv(in_channels,units=units_list[0],step_num=1)
        self.GC2 = GraphConv(units_list[0],units=units_list[1],step_num=1)
        self.GC3 = GraphConv(units_list[1],units=units_list[2],step_num=1)
        self.GC4 = GraphConv(units_list[2],units=h_dims[0],step_num=1)
        BRD1 = [nn.BatchNorm1d(units_list[0]),nn.ReLU()]
        BRD2 = [nn.BatchNorm1d(units_list[1]),nn.ReLU()]
        BRD3 = [nn.BatchNorm1d(units_list[2]),nn.ReLU()]
        BRD4 = [nn.BatchNorm1d(h_dims[0]),nn.ReLU()]

        if use_dropout:
            BRD1 += [nn.Dropout(0.1)]
            BRD2 += [nn.Dropout(0.1)]
            BRD3 += [nn.Dropout(0.1)]
            BRD4 += [nn.Dropout(0.1)]
            
        self.BRD1 = nn.Sequential(*BRD1)
        self.BRD2 = nn.Sequential(*BRD2)
        self.BRD3 = nn.Sequential(*BRD3)
        self.BRD4 = nn.Sequential(*BRD4)
        
        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(nn.Linear(h_dims[1],300),
                                    nn.Dropout(0.1))
        self.CONV = nn.Sequential(nn.Conv2d(1, 30, kernel_size=(150,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((2,1)),
                                  nn.Conv2d(30, 10, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Conv2d(10, 5, kernel_size=(5,1), stride=(1,1)),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,1)),
                                  nn.Dropout(0.2))
                                  
        if is_regr:
            self.fc = nn.Linear(30,1)
        else:
            self.fc = nn.Sequential(nn.Linear(30,1),
                                    nn.Sigmoid())
        
        #self.Linear.apply(self.init_weights)
        #self.CONV.apply(self.init_weights)
        #self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        
         
    def forward(self, drug_feat, drug_adj, x_gexpr):
        GCN_layer = self.BRD1(self.GC1([drug_feat, drug_adj])).permute(0,2,1)
        GCN_layer = self.BRD2(self.GC2([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.BRD3(self.GC3([GCN_layer, drug_adj])).permute(0,2,1)
        GCN_layer = self.BRD4(self.GC4([GCN_layer, drug_adj]))
        GCN_layer = self.Pool(GCN_layer)
        x_drug = GCN_layer.view(GCN_layer.shape[0],-1)
        x = torch.cat((x_drug,x_gexpr),1)
        x = self.Linear(x)
        x = x.view(-1,1,300,1)
        output = self.CONV(x)
        output = self.fc(output.view(output.shape[0],-1))
        return output


class GCNWithSCF(nn.Module):
    def __init__(
        self,
        in_channels,
        units_list,
        scf_input_dim,
        scf_hidden_dim=256,
        scf_out_dim=128,
        h_dims=[100, 200],
        use_dropout=False,
    ):
        super(GCNWithSCF, self).__init__()
        self.GC1 = GraphConv(in_channels, units=units_list[0], step_num=1)
        self.GC2 = GraphConv(units_list[0], units=units_list[1], step_num=1)
        self.GC3 = GraphConv(units_list[1], units=units_list[2], step_num=1)
        self.GC4 = GraphConv(units_list[2], units=h_dims[0], step_num=1)

        BRD1 = [nn.BatchNorm1d(units_list[0]), nn.ReLU()]
        BRD2 = [nn.BatchNorm1d(units_list[1]), nn.ReLU()]
        BRD3 = [nn.BatchNorm1d(units_list[2]), nn.ReLU()]
        BRD4 = [nn.BatchNorm1d(h_dims[0]), nn.ReLU()]
        if use_dropout:
            BRD1 += [nn.Dropout(0.1)]
            BRD2 += [nn.Dropout(0.1)]
            BRD3 += [nn.Dropout(0.1)]
            BRD4 += [nn.Dropout(0.1)]

        self.BRD1 = nn.Sequential(*BRD1)
        self.BRD2 = nn.Sequential(*BRD2)
        self.BRD3 = nn.Sequential(*BRD3)
        self.BRD4 = nn.Sequential(*BRD4)

        # Small NN to compress high-dimensional scFoundation embeddings.
        self.scf_reducer = nn.Sequential(
            nn.Linear(scf_input_dim, scf_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(scf_hidden_dim, scf_out_dim),
            nn.ReLU(),
        )

        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(nn.Linear(h_dims[1], 300), nn.Dropout(0.1))
        self.CONV = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(150, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(30, 10, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(10, 5, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())

    def forward(self, drug_feat, drug_adj, x_gexpr, x_scf):
        GCN_layer = self.BRD1(self.GC1([drug_feat, drug_adj])).permute(0, 2, 1)
        GCN_layer = self.BRD2(self.GC2([GCN_layer, drug_adj])).permute(0, 2, 1)
        GCN_layer = self.BRD3(self.GC3([GCN_layer, drug_adj])).permute(0, 2, 1)
        GCN_layer = self.BRD4(self.GC4([GCN_layer, drug_adj]))
        GCN_layer = self.Pool(GCN_layer)
        x_drug = GCN_layer.view(GCN_layer.shape[0], -1)
        x_scf = self.scf_reducer(x_scf)
        x = torch.cat((x_drug, x_gexpr, x_scf), 1)
        x = self.Linear(x)
        x = x.view(-1, 1, 300, 1)
        output = self.CONV(x)
        output = self.fc(output.view(output.shape[0], -1))
        return output


class GCNWithSCFGatedFusion(nn.Module):
    def __init__(
        self,
        in_channels,
        units_list,
        gexpr_input_dim,
        scf_input_dim,
        scf_hidden_dim=256,
        scf_out_dim=128,
        h_dims=[100, 128],
        use_dropout=False,
    ):
        super(GCNWithSCFGatedFusion, self).__init__()
        self.GC1 = GraphConv(in_channels, units=units_list[0], step_num=1)
        self.GC2 = GraphConv(units_list[0], units=units_list[1], step_num=1)
        self.GC3 = GraphConv(units_list[1], units=units_list[2], step_num=1)
        self.GC4 = GraphConv(units_list[2], units=h_dims[0], step_num=1)

        BRD1 = [nn.BatchNorm1d(units_list[0]), nn.ReLU()]
        BRD2 = [nn.BatchNorm1d(units_list[1]), nn.ReLU()]
        BRD3 = [nn.BatchNorm1d(units_list[2]), nn.ReLU()]
        BRD4 = [nn.BatchNorm1d(h_dims[0]), nn.ReLU()]
        if use_dropout:
            BRD1 += [nn.Dropout(0.1)]
            BRD2 += [nn.Dropout(0.1)]
            BRD3 += [nn.Dropout(0.1)]
            BRD4 += [nn.Dropout(0.1)]

        self.BRD1 = nn.Sequential(*BRD1)
        self.BRD2 = nn.Sequential(*BRD2)
        self.BRD3 = nn.Sequential(*BRD3)
        self.BRD4 = nn.Sequential(*BRD4)

        self.scf_reducer = nn.Sequential(
            nn.Linear(scf_input_dim, scf_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(scf_hidden_dim, scf_out_dim),
            nn.ReLU(),
        )
        self.chem_proj = nn.Linear(h_dims[0], scf_out_dim)
        self.gexpr_proj = nn.Linear(gexpr_input_dim, scf_out_dim)
        self.gate_layer = nn.Linear(scf_out_dim * 3, scf_out_dim * 3)

        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(nn.Linear(h_dims[1], 300), nn.Dropout(0.1))
        self.CONV = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(150, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(30, 10, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(10, 5, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())

    def forward(self, drug_feat, drug_adj, x_gexpr, x_scf):
        GCN_layer = self.BRD1(self.GC1([drug_feat, drug_adj])).permute(0, 2, 1)
        GCN_layer = self.BRD2(self.GC2([GCN_layer, drug_adj])).permute(0, 2, 1)
        GCN_layer = self.BRD3(self.GC3([GCN_layer, drug_adj])).permute(0, 2, 1)
        GCN_layer = self.BRD4(self.GC4([GCN_layer, drug_adj]))
        GCN_layer = self.Pool(GCN_layer)
        x_drug = GCN_layer.view(GCN_layer.shape[0], -1)

        x_drug_proj = self.chem_proj(x_drug)
        x_gexpr_proj = self.gexpr_proj(x_gexpr)
        x_scf = self.scf_reducer(x_scf)
        gate_logits = self.gate_layer(torch.cat((x_drug_proj, x_gexpr_proj, x_scf), 1))
        gate_logits = gate_logits.view(-1, 3, x_drug_proj.shape[1])
        gate = torch.softmax(gate_logits, dim=1)
        x_fused = (
            gate[:, 0, :] * x_drug_proj
            + gate[:, 1, :] * x_gexpr_proj
            + gate[:, 2, :] * x_scf
        )

        x = x_fused
        x = self.Linear(x)
        x = x.view(-1, 1, 300, 1)
        output = self.CONV(x)
        output = self.fc(output.view(output.shape[0], -1))
        return output
        
class ADV(nn.Module):
    def __init__(self, nz):
        super(ADV, self).__init__()
        self.adv = nn.Sequential(nn.Linear(nz,nz//2),
                                 nn.ReLU(),
                                 nn.Linear(nz//2,nz//4),
                                 nn.ReLU(),
                                 nn.Linear(nz//4,1),
                                 nn.Sigmoid())
    
    def forward(self, x):
        output = self.adv(x)
        return output


class GraphGeneEncoderVAE(nn.Module):
    def __init__(
        self,
        gene_input_dim,
        nz,
        device,
        node_hidden_dim=256,
        node_out_dim=128,
        gnn_dims=[128, 128, 128],
    ):
        super(GraphGeneEncoderVAE, self).__init__()
        self.nz = nz
        self.device = device
        self.node_encoder = nn.Sequential(
            nn.Linear(gene_input_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_hidden_dim, node_out_dim),
            nn.ReLU(),
        )

        self.GC1 = GraphConv(node_out_dim, units=gnn_dims[0], step_num=1)
        self.GC2 = GraphConv(gnn_dims[0], units=gnn_dims[1], step_num=1)
        self.GC3 = GraphConv(gnn_dims[1], units=gnn_dims[2], step_num=1)
        self.BRD1 = nn.Sequential(nn.BatchNorm1d(gnn_dims[0]), nn.ReLU())
        self.BRD2 = nn.Sequential(nn.BatchNorm1d(gnn_dims[1]), nn.ReLU())
        self.BRD3 = nn.Sequential(nn.BatchNorm1d(gnn_dims[2]), nn.ReLU())
        self.Pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(gnn_dims[2], nz)
        self.fc2 = nn.Linear(gnn_dims[2], nz)

    def build_adjacency(self, node_repr):
        scale = torch.sqrt(
            torch.tensor(node_repr.shape[-1], dtype=node_repr.dtype, device=node_repr.device)
        )
        adj = torch.matmul(node_repr, node_repr.transpose(1, 2)) / scale
        adj = torch.softmax(adj, dim=-1)
        adj = 0.5 * (adj + adj.transpose(1, 2))
        eye = torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
        adj = adj + eye
        adj = adj / adj.sum(dim=-1, keepdim=True)
        return adj

    def encode(self, x):
        node_repr = self.node_encoder(x)
        adj = self.build_adjacency(node_repr)

        gnn_layer = self.BRD1(self.GC1([node_repr, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD2(self.GC2([gnn_layer, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD3(self.GC3([gnn_layer, adj]))
        pooled = self.Pool(gnn_layer).view(gnn_layer.shape[0], -1)
        return self.fc1(pooled), self.fc2(pooled)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps * std + mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar


class GenePTGraphEncoder(nn.Module):
    def __init__(
        self,
        gene_input_dim,
        out_dim,
        node_hidden_dim=256,
        node_out_dim=128,
        gnn_dims=[128, 128, 128],
    ):
        super(GenePTGraphEncoder, self).__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(gene_input_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(node_hidden_dim, node_out_dim),
            nn.ReLU(),
        )
        self.GC1 = GraphConv(node_out_dim, units=gnn_dims[0], step_num=1)
        self.GC2 = GraphConv(gnn_dims[0], units=gnn_dims[1], step_num=1)
        self.GC3 = GraphConv(gnn_dims[1], units=gnn_dims[2], step_num=1)
        self.BRD1 = nn.Sequential(nn.BatchNorm1d(gnn_dims[0]), nn.ReLU())
        self.BRD2 = nn.Sequential(nn.BatchNorm1d(gnn_dims[1]), nn.ReLU())
        self.BRD3 = nn.Sequential(nn.BatchNorm1d(gnn_dims[2]), nn.ReLU())
        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(gnn_dims[2], out_dim)

    def build_adjacency(self, weighted_gene_embeddings):
        adj = torch.matmul(
            weighted_gene_embeddings, weighted_gene_embeddings.transpose(1, 2)
        )
        adj = torch.softmax(adj, dim=-1)
        adj = 0.5 * (adj + adj.transpose(1, 2))
        eye = torch.eye(adj.shape[-1], device=adj.device).unsqueeze(0)
        adj = adj + eye
        adj = adj / adj.sum(dim=-1, keepdim=True)
        return adj

    def forward(self, weighted_gene_embeddings):
        adj = self.build_adjacency(weighted_gene_embeddings)
        node_repr = self.node_encoder(weighted_gene_embeddings)
        gnn_layer = self.BRD1(self.GC1([node_repr, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD2(self.GC2([gnn_layer, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD3(self.GC3([gnn_layer, adj]))
        pooled = self.Pool(gnn_layer).view(gnn_layer.shape[0], -1)
        return self.out_proj(pooled)


class GCNWithRawAndGenePTGraph(nn.Module):
    def __init__(
        self,
        in_channels,
        units_list,
        raw_gexpr_dim,
        genept_graph_dim,
        h_dims=[100, 200],
        use_dropout=False,
    ):
        super(GCNWithRawAndGenePTGraph, self).__init__()
        self.GC1 = GraphConv(in_channels, units=units_list[0], step_num=1)
        self.GC2 = GraphConv(units_list[0], units=units_list[1], step_num=1)
        self.GC3 = GraphConv(units_list[1], units=units_list[2], step_num=1)
        self.GC4 = GraphConv(units_list[2], units=h_dims[0], step_num=1)

        BRD1 = [nn.BatchNorm1d(units_list[0]), nn.ReLU()]
        BRD2 = [nn.BatchNorm1d(units_list[1]), nn.ReLU()]
        BRD3 = [nn.BatchNorm1d(units_list[2]), nn.ReLU()]
        BRD4 = [nn.BatchNorm1d(h_dims[0]), nn.ReLU()]
        if use_dropout:
            BRD1 += [nn.Dropout(0.1)]
            BRD2 += [nn.Dropout(0.1)]
            BRD3 += [nn.Dropout(0.1)]
            BRD4 += [nn.Dropout(0.1)]

        self.BRD1 = nn.Sequential(*BRD1)
        self.BRD2 = nn.Sequential(*BRD2)
        self.BRD3 = nn.Sequential(*BRD3)
        self.BRD4 = nn.Sequential(*BRD4)

        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(
            nn.Linear(h_dims[0] + raw_gexpr_dim + genept_graph_dim, 300),
            nn.Dropout(0.1),
        )
        self.CONV = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(150, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(30, 10, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(10, 5, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.2),
        )
        self.fc = nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())

    def forward(self, drug_feat, drug_adj, x_raw_gexpr, x_genept_graph):
        gcn_layer = self.BRD1(self.GC1([drug_feat, drug_adj])).permute(0, 2, 1)
        gcn_layer = self.BRD2(self.GC2([gcn_layer, drug_adj])).permute(0, 2, 1)
        gcn_layer = self.BRD3(self.GC3([gcn_layer, drug_adj])).permute(0, 2, 1)
        gcn_layer = self.BRD4(self.GC4([gcn_layer, drug_adj]))
        gcn_layer = self.Pool(gcn_layer)
        x_drug = gcn_layer.view(gcn_layer.shape[0], -1)
        x = torch.cat((x_drug, x_raw_gexpr, x_genept_graph), 1)
        x = self.Linear(x)
        x = x.view(-1, 1, 300, 1)
        output = self.CONV(x)
        output = self.fc(output.view(output.shape[0], -1))
        return output


class PrecomputedGeneGraphEncoder(nn.Module):
    def __init__(
        self,
        num_genes,
        out_dim,
        node_dim=64,
        gnn_dims=[64, 128, 128],
    ):
        super(PrecomputedGeneGraphEncoder, self).__init__()
        self.num_genes = num_genes
        self.node_embeddings = nn.Parameter(torch.randn(num_genes, node_dim) * 0.02)
        self.GC1 = GraphConv(node_dim, units=gnn_dims[0], step_num=1)
        self.GC2 = GraphConv(gnn_dims[0], units=gnn_dims[1], step_num=1)
        self.GC3 = GraphConv(gnn_dims[1], units=gnn_dims[2], step_num=1)
        self.BRD1 = nn.Sequential(nn.BatchNorm1d(gnn_dims[0]), nn.ReLU())
        self.BRD2 = nn.Sequential(nn.BatchNorm1d(gnn_dims[1]), nn.ReLU())
        self.BRD3 = nn.Sequential(nn.BatchNorm1d(gnn_dims[2]), nn.ReLU())
        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(gnn_dims[2], out_dim)

    def forward(self, adj):
        batch_size = adj.shape[0]
        node_repr = self.node_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        gnn_layer = self.BRD1(self.GC1([node_repr, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD2(self.GC2([gnn_layer, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD3(self.GC3([gnn_layer, adj]))
        pooled = self.Pool(gnn_layer).view(gnn_layer.shape[0], -1)
        return self.out_proj(pooled)


class GenePTNodeFeatureGraphEncoder(nn.Module):
    def __init__(
        self,
        genept_node_features,
        out_dim,
        node_dim=64,
        gnn_dims=[64, 128, 128],
    ):
        super(GenePTNodeFeatureGraphEncoder, self).__init__()
        self.register_buffer(
            "genept_node_features",
            torch.as_tensor(genept_node_features, dtype=torch.float32),
        )
        self.node_projector = nn.Sequential(
            nn.Linear(self.genept_node_features.shape[1], node_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.GC1 = GraphConv(node_dim, units=gnn_dims[0], step_num=1)
        self.GC2 = GraphConv(gnn_dims[0], units=gnn_dims[1], step_num=1)
        self.GC3 = GraphConv(gnn_dims[1], units=gnn_dims[2], step_num=1)
        self.BRD1 = nn.Sequential(nn.BatchNorm1d(gnn_dims[0]), nn.ReLU())
        self.BRD2 = nn.Sequential(nn.BatchNorm1d(gnn_dims[1]), nn.ReLU())
        self.BRD3 = nn.Sequential(nn.BatchNorm1d(gnn_dims[2]), nn.ReLU())
        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(gnn_dims[2], out_dim)

    def forward(self, adj):
        batch_size = adj.shape[0]
        node_repr = self.node_projector(self.genept_node_features)
        node_repr = node_repr.unsqueeze(0).expand(batch_size, -1, -1)
        gnn_layer = self.BRD1(self.GC1([node_repr, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD2(self.GC2([gnn_layer, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD3(self.GC3([gnn_layer, adj]))
        pooled = self.Pool(gnn_layer).view(gnn_layer.shape[0], -1)
        return self.out_proj(pooled)


class ExpressionWeightedGenePTNodeFeatureGraphEncoder(nn.Module):
    def __init__(
        self,
        genept_node_features,
        out_dim,
        node_dim=64,
        gnn_dims=[64, 128, 128],
    ):
        super(ExpressionWeightedGenePTNodeFeatureGraphEncoder, self).__init__()
        self.register_buffer(
            "genept_node_features",
            torch.as_tensor(genept_node_features, dtype=torch.float32),
        )
        self.node_projector = nn.Sequential(
            nn.Linear(self.genept_node_features.shape[1], node_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.GC1 = GraphConv(node_dim, units=gnn_dims[0], step_num=1)
        self.GC2 = GraphConv(gnn_dims[0], units=gnn_dims[1], step_num=1)
        self.GC3 = GraphConv(gnn_dims[1], units=gnn_dims[2], step_num=1)
        self.BRD1 = nn.Sequential(nn.BatchNorm1d(gnn_dims[0]), nn.ReLU())
        self.BRD2 = nn.Sequential(nn.BatchNorm1d(gnn_dims[1]), nn.ReLU())
        self.BRD3 = nn.Sequential(nn.BatchNorm1d(gnn_dims[2]), nn.ReLU())
        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(gnn_dims[2], out_dim)

    def forward(self, adj, z_scores):
        node_repr = self.node_projector(self.genept_node_features)
        node_repr = node_repr.unsqueeze(0) * z_scores.unsqueeze(-1)
        gnn_layer = self.BRD1(self.GC1([node_repr, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD2(self.GC2([gnn_layer, adj])).permute(0, 2, 1)
        gnn_layer = self.BRD3(self.GC3([gnn_layer, adj]))
        pooled = self.Pool(gnn_layer).view(gnn_layer.shape[0], -1)
        return self.out_proj(pooled)


class GenePTZScoreTransformerEncoder(nn.Module):
    def __init__(
        self,
        genept_node_features,
        out_dim=64,
        d_model=64,
        nhead=4,
        num_layers=1,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super(GenePTZScoreTransformerEncoder, self).__init__()
        self.register_buffer(
            "genept_node_features",
            torch.as_tensor(genept_node_features, dtype=torch.float32),
        )
        self.token_projector = nn.Sequential(
            nn.Linear(self.genept_node_features.shape[1], d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, z_scores):
        weighted_genept = self.genept_node_features.unsqueeze(0) * z_scores.unsqueeze(-1)
        tokens = self.token_projector(weighted_genept)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return self.out_proj(pooled)


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=256, dropout=0.1):
        super(LatentDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent):
        return self.decoder(latent)


class GCNWithRawAndAuxFusion(nn.Module):
    def __init__(
        self,
        in_channels,
        units_list,
        raw_gexpr_dim,
        aux_dim,
        h_dims=[100, 200],
        use_dropout=False,
        is_regr=False,
    ):
        super(GCNWithRawAndAuxFusion, self).__init__()
        self.GC1 = GraphConv(in_channels, units=units_list[0], step_num=1)
        self.GC2 = GraphConv(units_list[0], units=units_list[1], step_num=1)
        self.GC3 = GraphConv(units_list[1], units=units_list[2], step_num=1)
        self.GC4 = GraphConv(units_list[2], units=h_dims[0], step_num=1)

        BRD1 = [nn.BatchNorm1d(units_list[0]), nn.ReLU()]
        BRD2 = [nn.BatchNorm1d(units_list[1]), nn.ReLU()]
        BRD3 = [nn.BatchNorm1d(units_list[2]), nn.ReLU()]
        BRD4 = [nn.BatchNorm1d(h_dims[0]), nn.ReLU()]
        if use_dropout:
            BRD1 += [nn.Dropout(0.1)]
            BRD2 += [nn.Dropout(0.1)]
            BRD3 += [nn.Dropout(0.1)]
            BRD4 += [nn.Dropout(0.1)]

        self.BRD1 = nn.Sequential(*BRD1)
        self.BRD2 = nn.Sequential(*BRD2)
        self.BRD3 = nn.Sequential(*BRD3)
        self.BRD4 = nn.Sequential(*BRD4)

        self.Pool = nn.AdaptiveAvgPool1d(1)
        self.Linear = nn.Sequential(
            nn.Linear(h_dims[0] + raw_gexpr_dim + aux_dim, 300),
            nn.Dropout(0.1),
        )
        self.CONV = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=(150, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(30, 10, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Conv2d(10, 5, kernel_size=(5, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 1)),
            nn.Dropout(0.2),
        )
        if is_regr:
            self.fc = nn.Linear(30, 1)
        else:
            self.fc = nn.Sequential(nn.Linear(30, 1), nn.Sigmoid())

    def forward(self, drug_feat, drug_adj, x_raw_gexpr, x_aux):
        gcn_layer = self.BRD1(self.GC1([drug_feat, drug_adj])).permute(0, 2, 1)
        gcn_layer = self.BRD2(self.GC2([gcn_layer, drug_adj])).permute(0, 2, 1)
        gcn_layer = self.BRD3(self.GC3([gcn_layer, drug_adj])).permute(0, 2, 1)
        gcn_layer = self.BRD4(self.GC4([gcn_layer, drug_adj]))
        gcn_layer = self.Pool(gcn_layer)
        x_drug = gcn_layer.view(gcn_layer.shape[0], -1)
        x = torch.cat((x_drug, x_raw_gexpr, x_aux), 1)
        x = self.Linear(x)
        x = x.view(-1, 1, 300, 1)
        output = self.CONV(x)
        output = self.fc(output.view(output.shape[0], -1))
        return output


class GeneformerZScoreTransformerEncoder(nn.Module):
    def __init__(
        self,
        gene_embedding_matrix,
        out_dim=128,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super(GeneformerZScoreTransformerEncoder, self).__init__()
        self.register_buffer(
            "gene_embedding_matrix",
            torch.as_tensor(gene_embedding_matrix, dtype=torch.float32),
        )
        self.num_genes = int(self.gene_embedding_matrix.shape[0])
        self.token_projector = nn.Sequential(
            nn.Linear(self.gene_embedding_matrix.shape[1], d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_genes + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

        init.trunc_normal_(self.cls_token, std=0.02)
        init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, z_scores):
        weighted_embeddings = (
            self.gene_embedding_matrix.unsqueeze(0) * z_scores.unsqueeze(-1)
        )
        tokens = self.token_projector(weighted_embeddings)
        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        tokens = self.transformer(tokens)
        pooled = tokens[:, 0, :]
        return self.out_proj(pooled)


class ScFoundationZScoreTransformerEncoder(nn.Module):
    def __init__(
        self,
        gene_embedding_matrix,
        out_dim=32,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super(ScFoundationZScoreTransformerEncoder, self).__init__()
        self.register_buffer(
            "gene_embedding_matrix",
            torch.as_tensor(gene_embedding_matrix, dtype=torch.float32),
        )
        self.token_projector = nn.Sequential(
            nn.Linear(self.gene_embedding_matrix.shape[1], d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, z_scores):
        weighted_embeddings = (
            self.gene_embedding_matrix.unsqueeze(0) * z_scores.unsqueeze(-1)
        )
        tokens = self.token_projector(weighted_embeddings)
        tokens = self.transformer(tokens)
        pooled = tokens.mean(dim=1)
        return self.out_proj(pooled)


class GeneformerContextualEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_path,
        gene_token_ids,
        out_dim=128,
        d_model=256,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        cls_token_id=2,
        eos_token_id=3,
        freeze_backbone=True,
    ):
        super(GeneformerContextualEncoder, self).__init__()
        try:
            from transformers import BertModel
        except ImportError as exc:
            raise ImportError(
                "Geneformer contextual encoder requires the transformers package."
            ) from exc

        self.geneformer = BertModel.from_pretrained(
            pretrained_model_path,
            local_files_only=True,
        )
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for param in self.geneformer.parameters():
                param.requires_grad = False

        token_ids = torch.as_tensor(gene_token_ids, dtype=torch.long)
        self.register_buffer("gene_token_ids", token_ids)
        self.cls_token_id = int(cls_token_id)
        self.eos_token_id = int(eos_token_id)
        self.num_genes = int(token_ids.shape[0])
        hidden_size = int(self.geneformer.config.hidden_size)
        self.hidden_size = hidden_size

        self.context_projector = nn.Sequential(
            nn.Linear(hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine_cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_genes + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.refine_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
        )

        init.trunc_normal_(self.refine_cls_token, std=0.02)
        init.trunc_normal_(self.position_embedding, std=0.02)

    def _build_ranked_inputs(self, z_scores):
        batch_size = z_scores.shape[0]
        rank_indices = torch.argsort(z_scores, dim=1, descending=True)
        token_template = self.gene_token_ids.unsqueeze(0).expand(batch_size, -1)
        ordered_gene_tokens = torch.gather(token_template, 1, rank_indices)
        cls_tokens = torch.full(
            (batch_size, 1),
            self.cls_token_id,
            dtype=torch.long,
            device=z_scores.device,
        )
        eos_tokens = torch.full(
            (batch_size, 1),
            self.eos_token_id,
            dtype=torch.long,
            device=z_scores.device,
        )
        input_ids = torch.cat((cls_tokens, ordered_gene_tokens, eos_tokens), dim=1)
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def forward(self, z_scores):
        input_ids, attention_mask = self._build_ranked_inputs(z_scores)
        if self.freeze_backbone:
            self.geneformer.eval()
            with torch.no_grad():
                backbone_outputs = self.geneformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        else:
            backbone_outputs = self.geneformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        gene_hidden = backbone_outputs.last_hidden_state[:, 1:-1, :]
        tokens = self.context_projector(gene_hidden)
        cls_token = self.refine_cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        tokens = self.refine_transformer(tokens)
        pooled = tokens[:, 0, :]
        return self.out_proj(pooled)

    def extract_pooled_embedding(self, z_scores):
        input_ids, attention_mask = self._build_ranked_inputs(z_scores)
        if self.freeze_backbone:
            self.geneformer.eval()
            with torch.no_grad():
                backbone_outputs = self.geneformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        else:
            backbone_outputs = self.geneformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        return backbone_outputs.last_hidden_state[:, 0, :]


        
        
