import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                  WhitenedVariationalStrategy)

from torch.nn import GRU, Linear, ReLU, Sequential
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops



class NeuralMessagePassing(nn.Module):
    def __init__(self, num_features, output_features, dim):
        super(NeuralMessagePassing, self).__init__()
        self.num_features = num_features
        self.output_features = output_features
        self.dim = dim
        self.lin0 = torch.nn.Linear(self.num_features, self.dim)

        seq_net = Sequential(Linear(6, 128), ReLU(),
                             Linear(128, self.dim * self.dim))
        self.conv = NNConv(self.dim, self.dim, seq_net, aggr='mean')
        self.gru = GRU(self.dim, self.dim)

        self.set2set = Set2Set(self.dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * self.dim, self.dim)
        # self.lin2 = torch.nn.Linear(self.dim, self.output_features)


    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(2):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = self.lin1(out)
        # out = self.lin2(out)

        return out


class Transform(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


class FeatureExtractor(nn.Sequential):
    def __init__(self, num_features, output_features, dim):
        super(FeatureExtractor, self).__init__()
        self.num_features = num_features
        self.output_features = output_features
        self.dim = dim
        self.add_module("neural_message_passing", NeuralMessagePassing(
            self.num_features, self.output_features, self.dim))


class GPRegressionLayer(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = WhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPRegressionLayer, self).__init__(variational_strategy)
        self.mean_func = gpytorch.means.ConstantMean()
        self.covar_func = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_func(x)
        covar_x = self.covar_func(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVDKL(gpytorch.Module):
    def __init__(self, inducing_points, feature_extractor):
        super(SVDKL, self).__init__()
        self.feature_extractor = feature_extractor
        self.inducing_points = inducing_points
        self.gp_layer = GPRegressionLayer(self.inducing_points)

    def forward(self, x):
        features = self.feature_extractor(x)
        res = self.gp_layer(features)
        return res


def gp_test(loader, std):
    errors = 0
    with torch.no_grad(), gpytorch.settings.max_cg_iterations(50), gpytorch.settings.use_toeplitz(False):
        for data in loader:
            data = data.to(device)
            preds = model(data)

            sum_error = torch.sum(torch.abs(preds.mean * std - data.y * std))
            errors += sum_error

    return errors / len(loader.dataset)


if __name__ == "__main__":

    path = "data/QM9"
    transform = T.Compose([Transform(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform).shuffle()
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std

    test_dataset = dataset[:20000]
    train_dataset = dataset[20000:]

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    num_features = dataset.num_features
    output_features = 5
    num_inducing_points = 500
    dim = 64
    feature_extractor = FeatureExtractor(num_features=num_features,
                                         output_features=output_features,
                                         dim=dim)

    inducing_loader = DataLoader(train_dataset[:num_inducing_points],
                                 batch_size=num_inducing_points)
    inducing_points = list(inducing_loader)[0]

    inducing_points = feature_extractor(inducing_points)
    model = SVDKL(inducing_points=inducing_points,
                  feature_extractor=feature_extractor)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # train
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-3},
        {'params': model.gp_layer.mean_func.parameters()},
        {'params': model.gp_layer.covar_func.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.005)

    mll = gpytorch.mlls.VariationalELBO(likelihood,
                                        model.gp_layer,
                                        num_data=len(train_dataset),
                                        combine_terms=False)

    i = 0
    max_iter = 1
    with gpytorch.settings.max_cg_iterations(50):
        for _ in range(max_iter):
            for data in train_loader:
                i += 1
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data)
                log_lik, kl_div, log_prior = mll(output, data.y)
                loss = -(log_lik - kl_div + log_prior)
                if i % 10 == 0:
                    print(loss.item())
                optimizer.step()
                loss.backward()

    # test 
    std = std.to(device)
    test_error = gp_test(test_loader, std)
    print("test mae: ", test_error.item())