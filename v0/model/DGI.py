from .GCN import * 
from .Discriminator import * 

from gh import * 


class DGI(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int, 
                 gcn_activation: Callable = nn.PReLU(),
                 readout_activation: Callable = nn.Sigmoid()):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.gcn = GCN(
            in_dim = in_dim, 
            out_dim = out_dim, 
            activation = gcn_activation, 
        )
        
        self.readout = lambda x: torch.mean(x, dim=0, keepdim=True)

        self.readout_activation = readout_activation 

        self.discriminator = Discriminator(out_dim)

    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatScalarTensor:
        num_nodes = g.num_nodes() 
        assert feat.shape == (num_nodes, self.in_dim)
                
        h_pos = self.gcn(g=g, feat=feat)
        assert h_pos.shape == (num_nodes, self.out_dim)
        
        summary = self.readout_activation(self.readout(h_pos))
        assert summary.shape == (1, self.out_dim)
        
        perm = np.random.permutation(num_nodes)
        feat_neg = feat[perm]
        h_neg = self.gcn(g=g, feat=feat_neg)
        assert h_neg.shape == (num_nodes, self.out_dim)

        score_pos, score_neg = self.discriminator(summary=summary, h_pos=h_pos, h_neg=h_neg)
        ones = torch.ones_like(score_pos)
        zeros = torch.zeros_like(score_neg)
        
        score = torch.cat([score_pos, score_neg])
        target = torch.cat([ones, zeros])
        assert score.shape == target.shape == (num_nodes * 2,)
        
        loss = F.binary_cross_entropy_with_logits(input=score, target=target)

        return loss 

    @torch.no_grad()
    def embed(self,
              g: dgl.DGLGraph,
              feat: FloatTensor):
        out = self.gcn(g=g, feat=feat)

        return out.detach()
