import sys
sys.path.append("../")
import lib.gcn3d as gcn3d

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(PriorEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(256, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        self.fc = nn.Linear(2048, emb_dim)

        self.neighbor_num = 20
        support_num = 1
        self.conv_0 = gcn3d.Conv_surface(kernel_num=32, support_num=support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num=support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num=support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num=support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d.Conv_layer(256, 1024, support_num=support_num)
        self.pool_3 = gcn3d.Pool_layer(pooling_rate=4, neighbor_num=4)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace= True),
            nn.Linear(512, 6)
        )

    def forward(self, xyz):
        """
        Args:
            xyz: (B, 3, N)

        """
        np = xyz.size()[2]
        x = F.relu(self.conv1(xyz))
        x = F.relu(self.conv2(x))
        global_feat = F.adaptive_max_pool1d(x, 1)

        gg = global_feat.repeat(1, 1, np)
        x = torch.cat((x, global_feat.repeat(1, 1, np)), dim=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)

        vertices = xyz.permute(0, 2, 1)
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

        fc_data = torch.cat((x, feature_global), dim=1)

        embedding = self.fc(fc_data)
        pred = self.classifier(fc_data)
        return embedding, pred


class PriorDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts):
        super(PriorDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 3*n_pts)

    def forward(self, embedding):
        """
        Args:
            embedding: (B, 512)

        """
        bs = embedding.size()[0]
        out = F.relu(self.fc1(embedding))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out_pc = out.view(bs, -1, 3)
        return out_pc


class PriorAE(nn.Module):
    def __init__(self, emb_dim=512, n_pts=1024):
        super(PriorAE, self).__init__()
        self.encoder = PriorEncoder(emb_dim)
        self.decoder = PriorDecoder(emb_dim, n_pts)

    def forward(self, in_pc, emb=None):
        """
        Args:
            in_pc: (B, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_pc: (B, n_pts, 3)

        """
        if emb is None:
            xyz = in_pc.permute(0, 2, 1)
            emb, pred = self.encoder(xyz)
        out_pc = self.decoder(emb)
        return emb, pred, out_pc


if __name__ == '__main__':
    estimator = PriorAE(512, 1024)
    xyz = torch.randn(32, 2048, 3)

    gg = estimator(xyz)