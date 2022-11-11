from gh import * 


class Discriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()

        self.in_dim = in_dim

        self.bfc = nn.Bilinear(in_dim, in_dim, 1)

    def forward(self, 
                summary: FloatTensor, 
                h_pos: FloatTensor, 
                h_neg: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        num_nodes = len(h_pos)
        assert summary.shape == (1, self.in_dim)
        assert h_pos.shape == (num_nodes, self.in_dim)
        assert h_neg.shape == (num_nodes, self.in_dim)
        
        summary = summary.expand_as(h_pos)

        score_pos = self.bfc(h_pos, summary).view(num_nodes)
        score_neg = self.bfc(h_neg, summary).view(num_nodes)

        return score_pos, score_neg 
