
import torch

class GUFM(torch.nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim, num_nodes, H_xn_gain=1):
        super().__init__()
        self.in_feature_dim = in_feature_dim
        self.out_feature_dim = out_feature_dim
        self.num_nodes = num_nodes
        self.params = torch.nn.ParameterDict({
            'W_1': torch.nn.Parameter(torch.empty(out_feature_dim, in_feature_dim)),
            # 'W_2': torch.nn.Parameter(torch.empty(out_feature_dim, in_feature_dim)),
            'H': torch.nn.Parameter(torch.empty(in_feature_dim, num_nodes))
        })
        torch.nn.init.normal_(self.params['W_1'])
        # torch.nn.init.normal_(self.params['W_2'])
        torch.nn.init.normal_(self.params['H'])
        # with torch.no_grad():
            # self.params['W_1'] = self.params['W_1']
            # self.params['H'] = self.params['H']
            # ignore structural information as of now
            # self.params['W_2'] = 0*self.params['W_2']

    def forward(self, A_hat):
        # return self.params['W_1']@self.params['H'] + self.params['W_2']@self.params['H']@A_hat
        return self.params['W_1']@self.params['H']
