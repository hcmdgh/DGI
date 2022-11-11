from gh import * 


class GCN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: Callable = nn.PReLU()):
        super().__init__()
        
        self.conv = dglnn.GraphConv(
            in_feats = in_dim,
            out_feats = out_dim,
            activation = activation, 
        )
        
    def forward(self,
                g: dgl.DGLGraph,
                feat: FloatTensor) -> FloatTensor:
        out = self.conv(graph=g, feat=feat)
        
        return out 
