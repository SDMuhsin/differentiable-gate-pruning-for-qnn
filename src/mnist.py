# quantum_dgp_mnist.py
# TorchQuantum ≥0.4.0  ▸  python quantum_dgp_mnist.py --epochs 3
import argparse, random, numpy as np, torch, torch.nn.functional as F
import torch.optim as optim
import torchquantum as tq
from torchquantum.dataset import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

# -------------------------------
# 1.  Utility: active-gate count
# -------------------------------
def count_qgates(qmodule):
    """Traverse all QuantumModule children and count how many calls to
       .rx/.ry/.rz/.crx appear in their forward graphs (Baseline),
       OR, for DGP layers, how many switch parameters are ≥0.5."""
    n = 0
    for m in qmodule.modules():
        if hasattr(m, "gate_switch"):
            n += int(torch.sigmoid(m.gate_switch).item() >= 0.5)
        elif isinstance(m, (tq.RX, tq.RY, tq.RZ, tq.CRX)):
            n += 1
    return n

# -----------------------------------------
# 2.  Differentiable-Gate-Pruning gate wrap
# -----------------------------------------
class PrunableRX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.θ = torch.nn.Parameter(torch.randn(1))       # rotation angle
        self.gate_switch = torch.nn.Parameter(torch.zeros(1))  # ζ
    def forward(self, qdev, w, static=False, graph=None):
        s = torch.sigmoid(self.gate_switch)               # 0‥1
        qdev.rx(wires=w, params=self.θ * s,
                static=static, parent_graph=graph)

class PrunableRY(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.θ = torch.nn.Parameter(torch.randn(1))
        self.gate_switch = torch.nn.Parameter(torch.zeros(1))
    def forward(self, qdev, w, static=False, graph=None):
        s = torch.sigmoid(self.gate_switch)
        qdev.ry(wires=w, params=self.θ * s,
                static=static, parent_graph=graph)

class PrunableRZ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.θ = torch.nn.Parameter(torch.randn(1))
        self.gate_switch = torch.nn.Parameter(torch.zeros(1))
    def forward(self, qdev, w, static=False, graph=None):
        s = torch.sigmoid(self.gate_switch)
        qdev.rz(wires=w, params=self.θ * s,
                static=static, parent_graph=graph)

class PrunableCRX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.θ = torch.nn.Parameter(torch.randn(1))
        self.gate_switch = torch.nn.Parameter(torch.zeros(1))
    def forward(self, qdev, w, static=False, graph=None):
        s = torch.sigmoid(self.gate_switch)
        qdev.crx(wires=w, params=self.θ * s,
                 static=static, parent_graph=graph)

# --------------------------------
# 3.  Baseline model (unchanged)
# --------------------------------
class BaselineModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(4)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
        def forward(self, qdev):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0,2])
            qdev.h(wires=3); qdev.sx(wires=2); qdev.cnot(wires=[3,0])
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_u3_h_rx"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x):
        bsz = x.shape[0]; x = F.avg_pool2d(x,6).view(bsz,16)
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz,
                                device=x.device, record_op=False)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        x = self.measure(qdev).reshape(bsz,2,2).sum(-1)
        return F.log_softmax(x, dim=1)

# -----------------------------------------------
# 4.  DGP model – same macro-architecture
# -----------------------------------------------
class DGPModel(tq.QuantumModule):
    λ_switch = 1e-2         # strength of sparsity regulariser
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(4)))
            # prunable gates
            self.rx0 = PrunableRX(); self.ry0 = PrunableRY()
            self.rz0 = PrunableRZ(); self.crx0 = PrunableCRX()
        def forward(self, qdev):
            self.random_layer(qdev)
            self.rx0(qdev, 0); self.ry0(qdev, 1)
            self.rz0(qdev, 3); self.crx0(qdev, [0,2])
            qdev.h(wires=3); qdev.sx(wires=2); qdev.cnot(wires=[3,0])
            # save L₀-like penalty to module attr so the outer loop can grab it
            penalties = [torch.sigmoid(g.gate_switch).mean()
                         for g in [self.rx0,self.ry0,self.rz0,self.crx0]]
            self.penalty = sum(penalties)
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_u3_h_rx"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x):
        bsz = x.shape[0]; x = F.avg_pool2d(x,6).view(bsz,16)
        qdev = tq.QuantumDevice(n_wires=4, bsz=bsz,
                                device=x.device, record_op=False)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        x = self.measure(qdev).reshape(bsz,2,2).sum(-1)
        return F.log_softmax(x, dim=1)
    def extra_loss(self):
        # grab the sparsity term collected in forward
        return self.q_layer.penalty * self.λ_switch

# --------------------------
# 5.  Train / evaluate utils
# --------------------------
def run_epoch(dataflow, model, opt=None):
    is_train = opt is not None
    loss_all, corr_all, n = 0,0,0
    model.train(is_train)
    for batch in dataflow:
        x = batch["image"].to(device); y = batch["digit"].to(device)
        out = model(x)
        loss = F.nll_loss(out, y)
        if isinstance(model, DGPModel):           # add sparsity loss
            loss = loss + model.extra_loss()
        if is_train:
            opt.zero_grad(); loss.backward(); opt.step()
        loss_all += loss.item()*y.size(0)
        corr_all += out.argmax(1).eq(y).sum().item()
        n += y.size(0)
    return corr_all/n, loss_all/n

# ---------------------
# 6.  Main experiment
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    # reproducibility
    seed = 0; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # data: 3 vs 6 MNIST
    ds = MNIST(root="./mnist_data",
               train_valid_split_ratio=[0.9,0.1],
               digits_of_interest=[3,6],
               n_test_samples=75)
    loaders = {split: torch.utils.data.DataLoader(
                    ds[split], batch_size=256, shuffle=True, num_workers=2)
               for split in ds}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Baseline ----------
    base = BaselineModel().to(device)
    opt_base = optim.Adam(base.parameters(), lr=5e-3, weight_decay=1e-4)
    sched_base = CosineAnnealingLR(opt_base, T_max=args.epochs)
    for ep in range(args.epochs):
        run_epoch(loaders["train"], base, opt_base); sched_base.step()
    acc_b, _ = run_epoch(loaders["test"], base)
    gates_b = count_qgates(base); params_b = sum(p.numel() for p in base.parameters())

    # ---------- DGP variant ----------
    dgp = DGPModel().to(device)
    opt_dgp = optim.Adam(dgp.parameters(), lr=5e-3, weight_decay=1e-4)
    sched_dgp = CosineAnnealingLR(opt_dgp, T_max=args.epochs)
    for ep in range(args.epochs):
        run_epoch(loaders["train"], dgp, opt_dgp); sched_dgp.step()
    acc_d, _ = run_epoch(loaders["test"], dgp)
    gates_d = count_qgates(dgp); params_d = sum(p.numel() for p in dgp.parameters())

    # ------------- Report -------------
    print("\n===  Results  ===")
    print(f"{'Model':13s} |  Acc  | Gates |  Params")
    print("-"*40)
    print(f"Baseline-4q   | {acc_b:6.3f} | {gates_b:5d} | {params_b:7d}")
    print(f"DGP-4q       | {acc_d:6.3f} | {gates_d:5d} | {params_d:7d}")
    acc_drop = 100*(acc_b-acc_d)/acc_b
    gate_drop = 100*(gates_b-gates_d)/gates_b
    print(f"\nAccuracy Δ: {acc_drop:+.2f}%   (≤ 3% goal)")
    print(f"Gate Δ:     {gate_drop:+.1f}%   (efficiency gain)")

