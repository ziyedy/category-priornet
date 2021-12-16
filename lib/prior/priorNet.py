import sys

sys.path.append("../../")
import lib.gcn3d as gcn3d

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEncoder(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int):
        super(PriorEncoder, self).__init__()

        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num=32, support_num=support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num=support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num=support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num=support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num=support_num)
        self.pool_3 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)

    def forward(self, vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace=True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace=True)
        vertices, fm_1 = self.pool_1(vertices, fm_1)

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace=True)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace=True)
        vertices, fm_3 = self.pool_2(vertices, fm_3)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        feature_global = fm_4.max(1)[0]
        # fm_4 = F.relu(fm_4, inplace=True)
        # vertices, fm_4 = self.pool_3(vertices, fm_4)

        return feature_global


class PriorDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts):
        super(PriorDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3 * n_pts)

    def forward(self, embedding):
        """
        Args:
            embedding: (B, 512)

        """
        bs = embedding.size()[0]
        out1 = F.relu(self.fc1(embedding))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        out_pc = out3.view(bs, -1, 3)
        return out_pc


class PriorNet(nn.Module):
    def __init__(self, emb_dim=512, n_pts=1024):
        super(PriorNet, self).__init__()
        self.encoder = PriorEncoder(1, 20)
        self.decoder = PriorDecoder(emb_dim, n_pts)

    def forward(self, in_pc):
        emb = self.encoder(in_pc)
        out_pc = self.decoder(emb)
        return emb, out_pc


if __name__ == '__main__':
    estimator = PriorEncoder(1, 1)
    xyz = torch.randn(32, 2048, 3)

    gg = estimator(xyz)