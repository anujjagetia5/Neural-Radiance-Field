import torch
import torch.nn as nn

class NerfModel(nn.Module):
    def __init__(self, L_x, L_d):   
        super(NerfModel, self).__init__()
        
        self.block1 = nn.Sequential(nn.Linear(L_x * 6 + 3, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(), )
        # density estimation
        self.block2 = nn.Sequential(nn.Linear(L_x * 6 + 256 + 3, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256 + 1), )
        # color estimation
        self.block3 = nn.Sequential(nn.Linear(L_d * 6 + 256 + 3, 256 // 2), nn.ReLU(), )
        self.block4 = nn.Sequential(nn.Linear(256 // 2, 3), nn.Sigmoid(), )

        self.L_x = L_x
        self.L_d = L_d
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.L_x) # emb_x: [batch_size, L_x * 6]
        emb_d = self.positional_encoding(d, self.L_d) # emb_d: [batch_size, L_d * 6]
        emb_x = emb_x.to(torch.float32)
        emb_d = emb_d.to(torch.float32)
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        tmp = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma
