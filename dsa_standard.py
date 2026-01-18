import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.parametrize as parametrize
from typing import Literal

try:
    import ot  # optional dependency for wasserstein compare
except Exception:  # pragma: no cover - optional
    ot = None


def embed_signal_torch(data, n_delays, delay_interval=1):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    device = data.device

    if data.shape[int(data.ndim == 3)] - (n_delays - 1) * delay_interval < 1:
        raise ValueError("The number of delays is too large for the number of time points in the data!")

    if data.ndim == 3:
        embedding = torch.zeros(
            (data.shape[0], data.shape[1] - (n_delays - 1) * delay_interval, data.shape[2] * n_delays)
        ).to(device)
    else:
        embedding = torch.zeros(
            (data.shape[0] - (n_delays - 1) * delay_interval, data.shape[1] * n_delays)
        ).to(device)

    for d in range(n_delays):
        index = (n_delays - 1 - d) * delay_interval
        ddelay = d * delay_interval

        if data.ndim == 3:
            ddata = d * data.shape[2]
            embedding[:, :, ddata : ddata + data.shape[2]] = data[:, index : data.shape[1] - ddelay]
        else:
            ddata = d * data.shape[1]
            embedding[:, ddata : ddata + data.shape[1]] = data[index : data.shape[0] - ddelay]

    return embedding


class DMD:
    def __init__(
        self,
        data,
        n_delays,
        delay_interval=1,
        rank=None,
        rank_thresh=None,
        rank_explained_variance=None,
        reduced_rank_reg=False,
        lamb=0,
        device="cpu",
        verbose=False,
        send_to_cpu=False,
        steps_ahead=1,
    ):
        self.device = device
        self._init_data(data)

        self.n_delays = n_delays
        self.delay_interval = delay_interval
        self.rank = rank
        self.rank_thresh = rank_thresh
        self.rank_explained_variance = rank_explained_variance
        self.reduced_rank_reg = reduced_rank_reg
        self.lamb = lamb
        self.verbose = verbose
        self.send_to_cpu = send_to_cpu
        self.steps_ahead = steps_ahead

        self.H = None
        self.U = None
        self.S = None
        self.V = None
        self.S_mat = None
        self.S_mat_inv = None
        self.A_v = None
        self.A_havok_dmd = None

    def _init_data(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        self.data = data
        if self.data.ndim == 3:
            self.ntrials = self.data.shape[0]
            self.window = self.data.shape[1]
            self.n = self.data.shape[2]
        else:
            self.window = self.data.shape[0]
            self.n = self.data.shape[1]
            self.ntrials = 1

    def compute_hankel(self, data=None, n_delays=None, delay_interval=None):
        if self.verbose:
            print("Computing Hankel matrix ...")

        self.data = self.data if data is None else self._init_data(data)
        self.n_delays = self.n_delays if n_delays is None else n_delays
        self.delay_interval = self.delay_interval if delay_interval is None else delay_interval
        self.data = self.data.to(self.device)

        self.H = embed_signal_torch(self.data, self.n_delays, self.delay_interval)

        if self.verbose:
            print("Hankel matrix computed!")

    def compute_svd(self):
        if self.verbose:
            print("Computing SVD on Hankel matrix ...")
        if self.H.ndim == 3:
            H = self.H.reshape(self.H.shape[0] * self.H.shape[1], self.H.shape[2])
        else:
            H = self.H
        U, S, Vh = torch.linalg.svd(H.T, full_matrices=False)

        V = Vh.T
        self.U = U
        self.S = S
        self.V = V

        self.S_mat = torch.diag(S).to(self.device)
        self.S_mat_inv = torch.diag(1 / S).to(self.device)

        exp_variance_inds = self.S**2 / ((self.S**2).sum())
        cumulative_explained = torch.cumsum(exp_variance_inds, 0)
        self.cumulative_explained_variance = cumulative_explained

        if self.reduced_rank_reg:
            V = self.V
        else:
            V = self.V

        if self.ntrials > 1:
            if V.numel() < self.H.numel():
                raise ValueError(
                    "The dimension of the SVD of the Hankel matrix is smaller than the dimension of the Hankel matrix itself. \n \
                                 This is likely due to the number of time points being smaller than the number of dimensions. \n \
                                 Please reduce the number of delays."
                )

            V = V.reshape(self.H.shape)
            newshape = (self.H.shape[0] * (self.H.shape[1] - self.steps_ahead), self.H.shape[2])
            self.Vt_minus = V[:, : -self.steps_ahead].reshape(newshape)
            self.Vt_plus = V[:, self.steps_ahead :].reshape(newshape)
        else:
            self.Vt_minus = V[: -self.steps_ahead]
            self.Vt_plus = V[self.steps_ahead :]

        if self.verbose:
            print("SVD complete!")

    def recalc_rank(self, rank, rank_thresh, rank_explained_variance):
        none_vars = (rank is None) + (rank_thresh is None) + (rank_explained_variance is None)
        if none_vars != 3:
            self.rank = None
            self.rank_thresh = None
            self.rank_explained_variance = None

        self.rank = self.rank if rank is None else rank
        self.rank_thresh = self.rank_thresh if rank_thresh is None else rank_thresh
        self.rank_explained_variance = (
            self.rank_explained_variance if rank_explained_variance is None else rank_explained_variance
        )

        none_vars = (self.rank is None) + (self.rank_thresh is None) + (self.rank_explained_variance is None)
        if none_vars < 2:
            raise ValueError(
                "More than one value was provided between rank, rank_thresh, and rank_explained_variance. Please provide only one of these, and ensure the others are None!"
            )
        elif none_vars == 3:
            self.rank = len(self.S)

        if self.reduced_rank_reg:
            S = self.proj_mat_S
        else:
            S = self.S

        if self.rank_thresh is not None:
            if S[-1] > self.rank_thresh:
                self.rank = len(S)
            else:
                self.rank = torch.argmax(torch.arange(len(S), 0, -1).to(self.device) * (S < self.rank_thresh))

        if self.rank_explained_variance is not None:
            self.rank = int(
                torch.argmax((self.cumulative_explained_variance > self.rank_explained_variance).type(torch.int))
                .cpu()
                .numpy()
            )

        if self.rank is not None and self.rank > self.H.shape[-1]:
            self.rank = self.H.shape[-1]

        if self.rank is None:
            if S[-1] > self.rank_thresh:
                self.rank = len(S)
            else:
                self.rank = torch.argmax(torch.arange(len(S), 0, -1).to(self.device) * (S < self.rank_thresh))

    def compute_havok_dmd(self, lamb=None):
        if self.verbose:
            print("Computing least squares fits to HAVOK DMD ...")

        self.lamb = self.lamb if lamb is None else lamb

        A_v = (
            torch.linalg.pinv(
                self.Vt_minus[:, : self.rank].T @ self.Vt_minus[:, : self.rank]
                + self.lamb * torch.eye(self.rank).to(self.device)
            )
            @ self.Vt_minus[:, : self.rank].T
            @ self.Vt_plus[:, : self.rank]
        ).T
        self.A_v = A_v
        self.A_havok_dmd = (
            self.U @ self.S_mat[: self.U.shape[1], : self.rank] @ self.A_v @ self.S_mat_inv[: self.rank, : self.U.shape[1]] @ self.U.T
        )

        if self.verbose:
            print("Least squares complete! \n")

    def compute_proj_mat(self, lamb=None):
        if self.verbose:
            print("Computing Projector Matrix for Reduced Rank Regression")

        self.lamb = self.lamb if lamb is None else lamb

        self.proj_mat = (
            self.Vt_plus.T
            @ self.Vt_minus
            @ torch.linalg.pinv(
                self.Vt_minus.T @ self.Vt_minus
                + self.lamb * torch.eye(self.Vt_minus.shape[1]).to(self.device)
            )
            @ self.Vt_minus.T
            @ self.Vt_plus
        )

        self.proj_mat_S, self.proj_mat_V = torch.linalg.eigh(self.proj_mat)
        self.proj_mat_S = torch.flip(self.proj_mat_S, dims=(0,))
        self.proj_mat_V = torch.flip(self.proj_mat_V, dims=(1,))

        if self.verbose:
            print("Projector Matrix computed! \n")

    def compute_reduced_rank_regression(self, lamb=None):
        if self.verbose:
            print("Computing Reduced Rank Regression ...")

        self.lamb = self.lamb if lamb is None else lamb
        proj_mat = self.proj_mat_V[:, : self.rank] @ self.proj_mat_V[:, : self.rank].T
        B_ols = (
            torch.linalg.pinv(
                self.Vt_minus.T @ self.Vt_minus
                + self.lamb * torch.eye(self.Vt_minus.shape[1]).to(self.device)
            )
            @ self.Vt_minus.T
            @ self.Vt_plus
        )

        self.A_v = B_ols @ proj_mat
        self.A_havok_dmd = (
            self.U
            @ self.S_mat[: self.U.shape[1], : self.A_v.shape[1]]
            @ self.A_v.T
            @ self.S_mat_inv[: self.A_v.shape[0], : self.U.shape[1]]
            @ self.U.T
        )

        if self.verbose:
            print("Reduced Rank Regression complete! \n")

    def fit(
        self,
        data=None,
        n_delays=None,
        delay_interval=None,
        rank=None,
        rank_thresh=None,
        rank_explained_variance=None,
        lamb=None,
        device=None,
        verbose=None,
        steps_ahead=None,
    ):
        self.steps_ahead = self.steps_ahead if steps_ahead is None else steps_ahead
        self.device = self.device if device is None else device
        self.verbose = self.verbose if verbose is None else verbose

        self.compute_hankel(data, n_delays, delay_interval)
        self.compute_svd()

        if self.reduced_rank_reg:
            self.compute_proj_mat(lamb)
            self.recalc_rank(rank, rank_thresh, rank_explained_variance)
            self.compute_reduced_rank_regression(lamb)
        else:
            self.recalc_rank(rank, rank_thresh, rank_explained_variance)
            self.compute_havok_dmd(lamb)

        if self.send_to_cpu:
            self.all_to_device("cpu")

    def all_to_device(self, device="cpu"):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)


class LearnableSimilarityTransform(nn.Module):
    def __init__(self, n, orthog=True):
        super(LearnableSimilarityTransform, self).__init__()
        self.C = nn.Parameter(torch.eye(n).float())
        self.orthog = orthog

    def forward(self, B):
        if self.orthog:
            return self.C @ B @ self.C.transpose(-1, -2)
        return self.C @ B @ torch.linalg.inv(self.C)


class Skew(nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.L1 = nn.Linear(n, n, bias=False, device=device)
        self.L2 = nn.Linear(n, n, bias=False, device=device)
        self.L3 = nn.Linear(n, n, bias=False, device=device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X - X.transpose(-1, -2)


class Matrix(nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.L1 = nn.Linear(n, n, bias=False, device=device)
        self.L2 = nn.Linear(n, n, bias=False, device=device)
        self.L3 = nn.Linear(n, n, bias=False, device=device)

    def forward(self, X):
        X = torch.tanh(self.L1(X))
        X = torch.tanh(self.L2(X))
        X = self.L3(X)
        return X


class CayleyMap(nn.Module):
    def __init__(self, n, device):
        super().__init__()
        self.register_buffer("Id", torch.eye(n, device=device))

    def forward(self, X):
        return torch.linalg.solve(self.Id + X, self.Id - X)


class SimilarityTransformDist:
    def __init__(
        self,
        iters=200,
        score_method: Literal["angular", "euclidean", "wasserstein"] = "angular",
        lr=0.01,
        device: Literal["cpu", "cuda"] = "cpu",
        verbose=False,
        group: Literal["O(n)", "SO(n)", "GL(n)"] = "O(n)",
        wasserstein_compare=None,
    ):
        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.verbose = verbose
        self.device = device
        self.C_star = None
        self.A = None
        self.B = None
        self.group = group
        self.wasserstein_compare = wasserstein_compare

    def fit(self, A, B, iters=None, lr=None, group=None):
        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == B.shape[1]

        A = A.to(self.device)
        B = B.to(self.device)
        self.A, self.B = A, B
        lr = self.lr if lr is None else lr
        iters = self.iters if iters is None else iters
        group = self.group if group is None else group

        if group in {"SO(n)", "O(n)"}:
            self.losses, self.C_star, self.sim_net = self.optimize_C(
                A, B, lr, iters, orthog=True, verbose=self.verbose
            )
        if group == "O(n)":
            P = torch.eye(B.shape[0], device=self.device)
            if P.shape[0] > 1:
                P[[0, 1], :] = P[[1, 0], :]
            losses, C_star, sim_net = self.optimize_C(
                A, P @ B @ P.T, lr, iters, orthog=True, verbose=self.verbose
            )
            if losses[-1] < self.losses[-1]:
                self.losses = losses
                self.C_star = C_star @ P
                self.sim_net = sim_net
        if group == "GL(n)":
            self.losses, self.C_star, self.sim_net = self.optimize_C(
                A, B, lr, iters, orthog=False, verbose=self.verbose
            )

    def optimize_C(self, A, B, lr, iters, orthog, verbose):
        n = A.shape[0]
        sim_net = LearnableSimilarityTransform(n, orthog=orthog).to(self.device)
        if orthog:
            parametrize.register_parametrization(sim_net, "C", Skew(n, self.device))
            parametrize.register_parametrization(sim_net, "C", CayleyMap(n, self.device))
        else:
            parametrize.register_parametrization(sim_net, "C", Matrix(n, self.device))

        simdist_loss = nn.MSELoss(reduction="sum")

        optimizer = optim.Adam(sim_net.parameters(), lr=lr)

        losses = []
        A /= torch.linalg.norm(A)
        B /= torch.linalg.norm(B)
        for _ in range(iters):
            optimizer.zero_grad()
            loss = simdist_loss(A, sim_net(B))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if verbose:
            print("Finished optimizing C")

        C_star = sim_net.C.detach()
        return losses, C_star, sim_net

    def score(self, A=None, B=None, score_method=None, group=None):
        assert self.C_star is not None
        A = self.A if A is None else A
        B = self.B if B is None else B
        assert A is not None
        assert B is not None
        assert A.shape == self.C_star.shape
        assert B.shape == self.C_star.shape
        score_method = self.score_method if score_method is None else score_method
        group = self.group if group is None else group
        with torch.no_grad():
            if not isinstance(A, torch.Tensor):
                A = torch.from_numpy(A).float().to(self.device)
            if not isinstance(B, torch.Tensor):
                B = torch.from_numpy(B).float().to(self.device)
            C = self.C_star.to(self.device)

        if group in {"SO(n)", "O(n)"}:
            Cinv = C.T
        elif group in {"GL(n)"}:
            Cinv = torch.linalg.inv(C)
        else:
            raise AssertionError("Need proper group name")
        if score_method == "angular":
            num = torch.trace(A.T @ C @ B @ Cinv)
            den = torch.norm(A, p="fro") * torch.norm(B, p="fro")
            score = torch.arccos(num / den).cpu().numpy()
            if np.isnan(score):
                if num / den < 0:
                    score = np.pi
                else:
                    score = 0
        else:
            score = torch.norm(A - C @ B @ Cinv, p="fro").cpu().numpy().item()

        return score

    def fit_score(
        self,
        A,
        B,
        iters=None,
        lr=None,
        score_method=None,
        zero_pad=True,
        group=None,
    ):
        score_method = self.score_method if score_method is None else score_method
        group = self.group if group is None else group

        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).float()
        if isinstance(B, np.ndarray):
            B = torch.from_numpy(B).float()

        if zero_pad and A.shape != B.shape:
            target_rows = max(A.shape[0], B.shape[0])
            target_cols = max(A.shape[1], B.shape[1])
            if A.shape[0] < target_rows or A.shape[1] < target_cols:
                pad_rows = target_rows - A.shape[0]
                pad_cols = target_cols - A.shape[1]
                A = torch.nn.functional.pad(A, (0, pad_cols, 0, pad_rows), value=0)
            if B.shape[0] < target_rows or B.shape[1] < target_cols:
                pad_rows = target_rows - B.shape[0]
                pad_cols = target_cols - B.shape[1]
                B = torch.nn.functional.pad(B, (0, pad_cols, 0, pad_rows), value=0)

        if zero_pad and A.shape != B.shape:
            target_rows = max(A.shape[0], B.shape[0])
            target_cols = max(A.shape[1], B.shape[1])
            if A.shape[0] < target_rows or A.shape[1] < target_cols:
                pad_rows = target_rows - A.shape[0]
                pad_cols = target_cols - A.shape[1]
                A = torch.nn.functional.pad(A, (0, pad_cols, 0, pad_rows), value=0)
            if B.shape[0] < target_rows or B.shape[1] < target_cols:
                pad_rows = target_rows - B.shape[0]
                pad_cols = target_cols - B.shape[1]
                B = torch.nn.functional.pad(B, (0, pad_cols, 0, pad_rows), value=0)

        assert A.shape == B.shape or self.wasserstein_compare is not None
        if A.shape != B.shape:
            if self.wasserstein_compare is None:
                raise AssertionError("Matrices must be the same size unless using wasserstein distance")
            else:
                print(f"resorting to wasserstein distance over {self.wasserstein_compare}")

        if self.score_method == "wasserstein":
            assert self.wasserstein_compare in {"sv", "eig"}
            if self.wasserstein_compare == "sv":
                a = torch.svd(A).S.view(-1, 1)
                b = torch.svd(B).S.view(-1, 1)
            elif self.wasserstein_compare == "eig":
                a = torch.linalg.eig(A).eigenvalues
                a = torch.vstack([a.real, a.imag]).T

                b = torch.linalg.eig(B).eigenvalues
                b = torch.vstack([b.real, b.imag]).T
            else:
                raise AssertionError("wasserstein_compare must be 'sv' or 'eig'")
            device = a.device
            a = a
            b = b
            if ot is None:
                raise ImportError("POT (ot) is required for wasserstein score_method.")
            M = ot.dist(a, b)
            a, b = torch.ones(a.shape[0]) / a.shape[0], torch.ones(b.shape[0]) / b.shape[0]
            a, b = a.to(device), b.to(device)

            score_star = ot.emd2(a, b, M)
        else:
            self.fit(A, B, iters, lr, group)
            score_star = self.score(self.A, self.B, score_method=score_method, group=group)

        return score_star


class DSA:
    def __init__(
        self,
        X,
        Y=None,
        n_delays=1,
        delay_interval=1,
        rank=None,
        rank_thresh=None,
        rank_explained_variance=None,
        lamb=0.0,
        send_to_cpu=True,
        iters=1500,
        score_method: Literal["angular", "euclidean", "wasserstein"] = "angular",
        lr=5e-3,
        group: Literal["GL(n)", "O(n)", "SO(n)"] = "O(n)",
        zero_pad=True,
        device="cpu",
        verbose=False,
        reduced_rank_reg=False,
        kernel=None,
        num_centers=0.1,
        svd_solver="arnoldi",
        wasserstein_compare: Literal["sv", "eig", None] = None,
    ):
        self.X = X
        self.Y = Y
        if self.X is None and isinstance(self.Y, list):
            self.X, self.Y = self.Y, self.X

        self.check_method()
        if self.method == "self-pairwise":
            self.data = [self.X]
        else:
            self.data = [self.X, self.Y]

        self.n_delays = self.broadcast_params(n_delays, cast=int)
        self.delay_interval = self.broadcast_params(delay_interval, cast=int)
        self.rank = self.broadcast_params(rank, cast=int)
        self.rank_thresh = self.broadcast_params(rank_thresh)
        self.rank_explained_variance = self.broadcast_params(rank_explained_variance)
        self.lamb = self.broadcast_params(lamb)
        self.send_to_cpu = send_to_cpu
        self.iters = iters
        self.score_method = score_method
        self.lr = lr
        self.device = device
        self.verbose = verbose
        self.zero_pad = zero_pad
        self.group = group
        self.reduced_rank_reg = reduced_rank_reg
        self.kernel = kernel
        self.wasserstein_compare = wasserstein_compare

        if kernel is None:
            self.dmds = [
                [
                    DMD(
                        Xi,
                        self.n_delays[i][j],
                        delay_interval=self.delay_interval[i][j],
                        rank=self.rank[i][j],
                        rank_thresh=self.rank_thresh[i][j],
                        rank_explained_variance=self.rank_explained_variance[i][j],
                        reduced_rank_reg=self.reduced_rank_reg,
                        lamb=self.lamb[i][j],
                        device=self.device,
                        verbose=self.verbose,
                        send_to_cpu=self.send_to_cpu,
                    )
                    for j, Xi in enumerate(dat)
                ]
                for i, dat in enumerate(self.data)
            ]
        else:
            self.dmds = [
                [
                    KernelDMD(
                        Xi,
                        self.n_delays[i][j],
                        kernel=self.kernel,
                        num_centers=num_centers,
                        delay_interval=self.delay_interval[i][j],
                        rank=self.rank[i][j],
                        reduced_rank_reg=self.reduced_rank_reg,
                        lamb=self.lamb[i][j],
                        verbose=self.verbose,
                        svd_solver=svd_solver,
                    )
                    for j, Xi in enumerate(dat)
                ]
                for i, dat in enumerate(self.data)
            ]

        self.simdist = SimilarityTransformDist(
            iters, score_method, lr, device, verbose, group, wasserstein_compare
        )

    def check_method(self):
        tensor_or_np = lambda x: isinstance(x, (np.ndarray, torch.Tensor))

        if isinstance(self.X, list):
            if self.Y is None:
                self.method = "self-pairwise"
            elif isinstance(self.Y, list):
                self.method = "bipartite-pairwise"
            elif tensor_or_np(self.Y):
                self.method = "list-to-one"
                self.Y = [self.Y]
            else:
                raise ValueError("unknown type of Y")
        elif tensor_or_np(self.X):
            self.X = [self.X]
            if self.Y is None:
                raise ValueError("only one element provided")
            elif isinstance(self.Y, list):
                self.method = "one-to-list"
            elif tensor_or_np(self.Y):
                self.method = "default"
                self.Y = [self.Y]
            else:
                raise ValueError("unknown type of Y")
        else:
            raise ValueError("unknown type of X")

    def broadcast_params(self, param, cast=None):
        out = []
        if isinstance(param, (int, float, np.integer)) or param is None:
            out.append([param] * len(self.X))
            if self.Y is not None:
                out.append([param] * len(self.Y))
        elif isinstance(param, (tuple, list, np.ndarray)):
            if self.method == "self-pairwise" and len(param) >= len(self.X):
                out = [param]
            else:
                assert len(param) <= 2

                for i, data in enumerate([self.X, self.Y]):
                    if data is None:
                        continue
                    if isinstance(param[i], (int, float)):
                        out.append([param[i]] * len(data))
                    elif isinstance(param[i], (list, np.ndarray, tuple)):
                        assert len(param[i]) >= len(data)
                        out.append(param[i][: len(data)])
        else:
            raise ValueError("unknown type entered for parameter")

        if cast is not None and param is not None:
            out = [[cast(x) for x in dat] for dat in out]

        return out

    def fit_score(self):
        for dmd_sets in self.dmds:
            for dmd in dmd_sets:
                dmd.fit()

        return self.score()

    def score(self, iters=None, lr=None, score_method=None):
        iters = self.iters if iters is None else iters
        lr = self.lr if lr is None else lr
        score_method = self.score_method if score_method is None else score_method

        ind2 = 1 - int(self.method == "self-pairwise")

        self.sims = np.zeros((len(self.dmds[0]), len(self.dmds[ind2])))
        for i, dmd1 in enumerate(self.dmds[0]):
            for j, dmd2 in enumerate(self.dmds[ind2]):
                if self.method == "self-pairwise":
                    if j >= i:
                        continue
                if self.verbose:
                    print(f"computing similarity between DMDs {i} and {j}")

                self.sims[i, j] = self.simdist.fit_score(
                    dmd1.A_v, dmd2.A_v, iters, lr, score_method, zero_pad=self.zero_pad
                )

                if self.method == "self-pairwise":
                    self.sims[j, i] = self.sims[i, j]

        if self.method == "default":
            return self.sims[0, 0]

        return self.sims
