# Graph Neural Networks — Deep Reference

## Core Message Passing Framework

All GNNs follow the message-passing neural network (MPNN) paradigm:

$$h_v^{(k)} = \text{UPDATE}\!\left(h_v^{(k-1)},\; \text{AGGREGATE}\!\left(\{h_u^{(k-1)} : u \in \mathcal{N}(v)\}\right)\right)$$

---

## Architecture Variants

### GCN (Graph Convolutional Network)
$$H^{(l+1)} = \sigma\!\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}\right)$$

where $\tilde{A} = A + I$ (self-loops), $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.

```python
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_channels if i == 0 else hidden, 
                    out_channels if i == num_layers-1 else hidden)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers-1)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)
```

### GAT (Graph Attention Network)
$$h_v^{(l)} = \sigma\!\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \alpha_{vu}^{(l)} W^{(l)} h_u^{(l-1)}\right)$$

Attention coefficients (normalized via softmax over neighbors):
$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(a^T[Wh_v \| Wh_u]))}{\sum_{k \in \mathcal{N}(v)}\exp(\text{LeakyReLU}(a^T[Wh_v \| Wh_k]))}$$

GAT v2 fixes the static attention problem — use GATv2Conv in practice.

### GraphSAGE (Inductive, Scalable)
$$h_v^{(k)} = \sigma\!\left(W \cdot \text{CONCAT}\!\left(h_v^{(k-1)},\; \text{MEAN}_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)\right)$$

Designed for inductive learning (unseen nodes at inference). Works with neighbor sampling.

### GIN (Graph Isomorphism Network) — Maximum Expressiveness
$$h_v^{(k)} = \text{MLP}^{(k)}\!\left((1+\epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)}\right)$$

As powerful as the Weisfeiler-Lehman graph isomorphism test (WL-1). Use for graph classification.

---

## Graph-Level Pooling (Graph Classification)

```python
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import TopKPooling, SAGPooling

class GraphClassifier(nn.Module):
    def __init__(self, in_channels, hidden, num_classes):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden)
        ))
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, edge_index, batch):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        # JK (Jumping Knowledge) concatenation
        graph_emb = torch.cat([
            global_add_pool(h1, batch),
            global_add_pool(h2, batch)
        ], dim=1)
        return self.classifier(graph_emb)
```

---

## Heterogeneous Graphs

Multiple node/edge types. Use HeteroConv wrapper:

```python
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGNN(nn.Module):
    def __init__(self, metadata, hidden):
        super().__init__()
        self.conv = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden)
            for edge_type in metadata[1]
        })
    
    def forward(self, x_dict, edge_index_dict):
        return self.conv(x_dict, edge_index_dict)
```

---

## Temporal / Dynamic Graphs

- **Discrete-time**: snapshot sequence → apply GNN per snapshot + temporal aggregation (LSTM/Transformer)
- **Continuous-time**: TGN (Temporal Graph Network) — memory modules + time encoding

$$\text{Time Encoding}: \Phi(t) = \sqrt{\frac{1}{d}}[\cos(\omega_1 t + \phi_1), \ldots, \cos(\omega_d t + \phi_d)]$$

---

## Scalability Techniques

| Technique | Idea | When to use |
|---|---|---|
| Neighbor sampling (GraphSAGE) | Sample fixed K neighbors per layer | Large graphs (millions of nodes) |
| Cluster-GCN | Partition graph into clusters, batch by cluster | Very large graphs |
| GraphSAINT | Random graph sampling with importance weighting | Accurate + scalable |
| SIGN | Precompute diffusion operators offline | Fastest inference |

---

## Domain Applications

| Domain | Task | Architecture |
|---|---|---|
| Molecular property prediction | Graph regression | MPNN / DimeNet++ / SchNet |
| Drug-drug interaction | Link prediction | GAT / HeteroGNN |
| Recommendation systems | Link prediction | LightGCN |
| Social network analysis | Node classification | GraphSAGE |
| Knowledge graph completion | Link prediction | RotatE / TransE |
| Traffic forecasting | Spatial-temporal | STGCN / DCRNN |
| Protein structure | 3D graph | SE(3)-equivariant GNN (e.g., EGNN) |

---

## Oversmoothing Problem

Deep GNNs suffer from oversmoothing — all node features converge.

Fixes:
- **Residual connections** (GCNII)
- **Initial residual + identity mapping**: $H^{(l)} = (1-\alpha)(\hat{A}H^{(l-1)}) + \alpha H^{(0)}$
- **DropEdge**: randomly drop edges during training
- **Pair Normalization**: normalize so features don't collapse
- **Limit depth**: 2-3 layers sufficient for most tasks
