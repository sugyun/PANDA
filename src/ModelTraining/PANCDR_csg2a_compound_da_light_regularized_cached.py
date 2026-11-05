# Adapted from PANCDR (https://github.com/DMCB-GIST/PANCDR) and extended
# with CSG2A-derived post-treatment and chemical representation branches.
import itertools
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ModelTraining.model import ADV, Encoder, GCN
from ModelTraining.csg2a_utils import (
    CSG2AFeatureEncoder,
    IsNaN,
    binary_accuracy_from_probs,
    set_requires_grad,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RawCSG2ACompoundDataset(Dataset):
    def __init__(
        self,
        drug_feat,
        drug_adj,
        raw_expr,
        csg2a_perturbed,
        csg2a_compound,
        csg2a_row_idx,
        y,
        indices,
    ):
        self.drug_feat = drug_feat
        self.drug_adj = drug_adj
        self.raw_expr = raw_expr
        self.csg2a_perturbed = csg2a_perturbed
        self.csg2a_compound = csg2a_compound
        self.csg2a_row_idx = csg2a_row_idx
        self.y = y
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = int(self.indices[idx])
        csg2a_row = int(self.csg2a_row_idx[row])
        csg2a_pert = np.asarray(self.csg2a_perturbed[csg2a_row], dtype=np.float32)
        csg2a_comp = np.asarray(self.csg2a_compound[csg2a_row], dtype=np.float32)
        return (
            torch.as_tensor(self.drug_feat[row], dtype=torch.float32),
            torch.as_tensor(self.drug_adj[row], dtype=torch.float32),
            torch.as_tensor(self.raw_expr[row], dtype=torch.float32),
            torch.as_tensor(csg2a_pert, dtype=torch.float32),
            torch.as_tensor(csg2a_comp, dtype=torch.float32),
            torch.as_tensor(self.y[row], dtype=torch.float32),
        )


class train_PANCDR_csg2a_compound_da_light_regularized:
    def __init__(
        self,
        train_data,
        test_data,
        csg2a_state_hidden_dim=128,
        csg2a_state_out_dim=32,
        csg2a_comp_hidden_dim=128,
        csg2a_comp_out_dim=32,
        csg2a_dropout=0.2,
        fusion_dropout=0.2,
        use_gcn_dropout=True,
        weight_decay=1e-4,
        adv_weight_decay=1e-4,
        lam_csg2a_adv=0.001,
        da_start_epoch=0,
        da_warmup_epochs=50,
        max_epochs=60,
        patience=5,
        eval_batch_size=128,
        source_val_data=None,
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.source_val_data = source_val_data
        self.csg2a_state_hidden_dim = csg2a_state_hidden_dim
        self.csg2a_state_out_dim = csg2a_state_out_dim
        self.csg2a_comp_hidden_dim = csg2a_comp_hidden_dim
        self.csg2a_comp_out_dim = csg2a_comp_out_dim
        self.csg2a_dropout = csg2a_dropout
        self.fusion_dropout = fusion_dropout
        self.use_gcn_dropout = use_gcn_dropout
        self.weight_decay = weight_decay
        self.adv_weight_decay = adv_weight_decay
        self.lam_csg2a_adv = lam_csg2a_adv
        self.da_start_epoch = da_start_epoch
        self.da_warmup_epochs = da_warmup_epochs
        self.max_epochs = max_epochs
        self.patience = patience
        self.eval_batch_size = eval_batch_size

    def _da_weight(self, epoch):
        if epoch < self.da_start_epoch:
            return 0.0
        if self.da_warmup_epochs <= 0:
            return 1.0
        step = epoch - self.da_start_epoch + 1
        return min(1.0, step / float(self.da_warmup_epochs))

    def _predict_loader(
        self,
        loader,
        EN_model,
        CSG2AState_model,
        CSG2ACompound_model,
        Fusion_dropout,
        GCN_model,
    ):
        targets = []
        preds = []
        EN_model.eval()
        CSG2AState_model.eval()
        CSG2ACompound_model.eval()
        Fusion_dropout.eval()
        GCN_model.eval()
        with torch.no_grad():
            for drug_feat, drug_adj, gexpr, csg2a_pert, csg2a_comp, y_true in loader:
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                csg2a_pert = csg2a_pert.to(device)
                csg2a_comp = csg2a_comp.to(device)
                _, raw_mu, _ = EN_model(gexpr)
                raw_latents = raw_mu
                csg2a_state_latents = CSG2AState_model(csg2a_pert)
                csg2a_comp_latents = CSG2ACompound_model(csg2a_comp)
                fused_latents = torch.cat(
                    (raw_latents, csg2a_state_latents, csg2a_comp_latents),
                    dim=1,
                )
                fused_latents = Fusion_dropout(fused_latents)
                y_pred = GCN_model(drug_feat, drug_adj, fused_latents)
                targets.append(y_true.view(-1, 1).cpu())
                preds.append(y_pred.cpu())
        return torch.cat(targets).numpy(), torch.cat(preds).numpy()

    def train(self, params, weight_path="../checkpoint/model.pt"):
        nz, d_dim, lr, lr_adv, lam, batch_size = params.values()

        (
            X_drug_feat_data,
            X_drug_adj_data,
            X_gexpr_data,
            X_csg2a_pert_data,
            X_csg2a_comp_data,
            X_csg2a_row_idx,
            Y,
            t_gexpr_feature,
            t_csg2a_zscore_feature,
        ) = self.train_data
        (
            TX_drug_feat_data_test,
            TX_drug_adj_data_test,
            TX_gexpr_data_test,
            TX_csg2a_pert_data_test,
            TX_csg2a_comp_data_test,
            TX_csg2a_row_idx_test,
            TY_test,
        ) = self.test_data

        source_indices = np.arange(len(Y), dtype=np.int64)
        if self.source_val_data is None:
            source_train_idx, source_val_idx = train_test_split(
                source_indices, test_size=0.05, random_state=0
            )
            (
                VX_drug_feat_data,
                VX_drug_adj_data,
                VX_gexpr_data,
                VX_csg2a_pert_data,
                VX_csg2a_comp_data,
                VX_csg2a_row_idx,
                VY,
                val_indices,
            ) = (
                X_drug_feat_data,
                X_drug_adj_data,
                X_gexpr_data,
                X_csg2a_pert_data,
                X_csg2a_comp_data,
                X_csg2a_row_idx,
                Y,
                source_val_idx,
            )
        else:
            source_train_idx = source_indices
            (
                VX_drug_feat_data,
                VX_drug_adj_data,
                VX_gexpr_data,
                VX_csg2a_pert_data,
                VX_csg2a_comp_data,
                VX_csg2a_row_idx,
                VY,
            ) = self.source_val_data
            val_indices = np.arange(len(VY), dtype=np.int64)

        X_t_train, X_t_val = train_test_split(
            t_gexpr_feature.T.values.astype("float32"),
            test_size=0.05,
            random_state=0,
        )
        X_t_csg2a_train, X_t_csg2a_val = train_test_split(
            np.asarray(t_csg2a_zscore_feature, dtype=np.float32),
            test_size=0.05,
            random_state=0,
        )

        train_dataset = RawCSG2ACompoundDataset(
            X_drug_feat_data,
            X_drug_adj_data,
            X_gexpr_data,
            X_csg2a_pert_data,
            X_csg2a_comp_data,
            X_csg2a_row_idx,
            Y,
            source_train_idx,
        )
        val_dataset = RawCSG2ACompoundDataset(
            VX_drug_feat_data,
            VX_drug_adj_data,
            VX_gexpr_data,
            VX_csg2a_pert_data,
            VX_csg2a_comp_data,
            VX_csg2a_row_idx,
            VY,
            val_indices,
        )
        test_dataset = RawCSG2ACompoundDataset(
            TX_drug_feat_data_test,
            TX_drug_adj_data_test,
            TX_gexpr_data_test,
            TX_csg2a_pert_data_test,
            TX_csg2a_comp_data_test,
            TX_csg2a_row_idx_test,
            TY_test,
            np.arange(len(TY_test), dtype=np.int64),
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size[0],
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
        )
        target_loader = DataLoader(
            dataset=TensorDataset(torch.FloatTensor(X_t_train)),
            batch_size=batch_size[1],
            shuffle=True,
            drop_last=True,
        )
        target_csg2a_loader = DataLoader(
            dataset=TensorDataset(torch.FloatTensor(X_t_csg2a_train)),
            batch_size=batch_size[1],
            shuffle=True,
            drop_last=True,
        )

        X_gexpr_val_raw = torch.FloatTensor(VX_gexpr_data[val_indices]).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        X_csg2a_val = torch.FloatTensor(
            np.asarray(VX_csg2a_pert_data[VX_csg2a_row_idx[val_indices]], dtype=np.float32)
        ).to(device)
        X_t_csg2a_val = torch.FloatTensor(X_t_csg2a_val).to(device)

        wait, best_auc = 0, 0
        EN_model = Encoder(X_gexpr_data.shape[1], nz, device).to(device)
        CSG2AState_model = CSG2AFeatureEncoder(
            X_csg2a_pert_data.shape[1],
            hidden_dim=self.csg2a_state_hidden_dim,
            out_dim=self.csg2a_state_out_dim,
            dropout=self.csg2a_dropout,
        ).to(device)
        CSG2ACompound_model = CSG2AFeatureEncoder(
            X_csg2a_comp_data.shape[1],
            hidden_dim=self.csg2a_comp_hidden_dim,
            out_dim=self.csg2a_comp_out_dim,
            dropout=self.csg2a_dropout,
        ).to(device)
        Fusion_dropout = nn.Dropout(self.fusion_dropout).to(device)
        fused_dim = nz + self.csg2a_state_out_dim + self.csg2a_comp_out_dim
        GCN_model = GCN(
            X_drug_feat_data.shape[2],
            [256, 256, 256],
            h_dims=[d_dim, fused_dim + d_dim],
            use_dropout=self.use_gcn_dropout,
        ).to(device)
        ADV_model = ADV(nz).to(device)
        ADV_csg2a_model = ADV(self.csg2a_state_out_dim).to(device)

        optimizer = torch.optim.Adam(
            itertools.chain(
                EN_model.parameters(),
                CSG2AState_model.parameters(),
                CSG2ACompound_model.parameters(),
                GCN_model.parameters(),
            ),
            lr=lr,
            weight_decay=self.weight_decay,
        )
        optimizer_adv = torch.optim.Adam(
            ADV_model.parameters(), lr=lr_adv, weight_decay=self.adv_weight_decay
        )
        optimizer_adv_csg2a = torch.optim.Adam(
            ADV_csg2a_model.parameters(), lr=lr_adv, weight_decay=self.adv_weight_decay
        )
        loss = torch.nn.BCELoss()

        for epoch in range(self.max_epochs):
            da_weight = self._da_weight(epoch)
            target_iter = iter(target_loader)
            target_csg2a_iter = iter(target_csg2a_loader)
            train_targets_epoch = []
            train_preds_epoch = []

            for drug_feat, drug_adj, gexpr, csg2a_pert, csg2a_comp, y_true in train_loader:
                try:
                    t_gexpr = next(target_iter)[0]
                except StopIteration:
                    target_iter = iter(target_loader)
                    t_gexpr = next(target_iter)[0]
                try:
                    t_csg2a = next(target_csg2a_iter)[0]
                except StopIteration:
                    target_csg2a_iter = iter(target_csg2a_loader)
                    t_csg2a = next(target_csg2a_iter)[0]

                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                csg2a_pert = csg2a_pert.to(device)
                csg2a_comp = csg2a_comp.to(device)
                y_true = y_true.view(-1, 1).to(device)
                t_gexpr = t_gexpr.to(device)
                t_csg2a = t_csg2a.to(device)

                EN_model.train()
                CSG2AState_model.train()
                CSG2ACompound_model.train()
                Fusion_dropout.train()
                GCN_model.train()

                if da_weight > 0:
                    ADV_model.train()
                    set_requires_grad(ADV_model, True)
                    optimizer_adv.zero_grad()
                    with torch.no_grad():
                        F_gexpr, _, _ = EN_model(gexpr)
                        F_t_gexpr, _, _ = EN_model(t_gexpr)
                    F_g_t_gexpr = torch.cat((F_gexpr, F_t_gexpr))
                    z_true = torch.cat(
                        (
                            torch.zeros(F_gexpr.shape[0], device=device),
                            torch.ones(F_t_gexpr.shape[0], device=device),
                        )
                    ).view(-1, 1)
                    z_pred = ADV_model(F_g_t_gexpr)
                    if IsNaN(z_pred):
                        return -1
                    adv_loss = loss(z_pred, z_true)
                    (da_weight * adv_loss).backward()
                    optimizer_adv.step()

                    ADV_csg2a_model.train()
                    set_requires_grad(ADV_csg2a_model, True)
                    optimizer_adv_csg2a.zero_grad()
                    with torch.no_grad():
                        F_csg2a = CSG2AState_model(csg2a_pert)
                        F_t_csg2a = CSG2AState_model(t_csg2a)
                    F_s_t_csg2a = torch.cat((F_csg2a, F_t_csg2a))
                    z_true_csg2a = torch.cat(
                        (
                            torch.zeros(F_csg2a.shape[0], device=device),
                            torch.ones(F_t_csg2a.shape[0], device=device),
                        )
                    ).view(-1, 1)
                    z_pred_csg2a = ADV_csg2a_model(F_s_t_csg2a)
                    if IsNaN(z_pred_csg2a):
                        return -1
                    adv_csg2a_loss = loss(z_pred_csg2a, z_true_csg2a)
                    (da_weight * adv_csg2a_loss).backward()
                    optimizer_adv_csg2a.step()

                set_requires_grad(ADV_model, False)
                set_requires_grad(ADV_csg2a_model, False)
                optimizer.zero_grad()

                raw_latents, _, _ = EN_model(gexpr)
                t_latents, _, _ = EN_model(t_gexpr)
                csg2a_state_latents = CSG2AState_model(csg2a_pert)
                t_csg2a_state_latents = CSG2AState_model(t_csg2a)
                csg2a_comp_latents = CSG2ACompound_model(csg2a_comp)
                fused_latents = torch.cat(
                    (raw_latents, csg2a_state_latents, csg2a_comp_latents),
                    dim=1,
                )
                fused_latents = Fusion_dropout(fused_latents)
                y_pred = GCN_model(drug_feat, drug_adj, fused_latents)
                if IsNaN(y_pred):
                    return -1

                F_g_t_latents = torch.cat((raw_latents, t_latents))
                z_true_flip = torch.cat(
                    (
                        torch.ones(raw_latents.shape[0], device=device),
                        torch.zeros(t_latents.shape[0], device=device),
                    )
                ).view(-1, 1)
                z_pred_flip = ADV_model(F_g_t_latents)
                if IsNaN(z_pred_flip):
                    return -1

                F_s_t_csg2a = torch.cat((csg2a_state_latents, t_csg2a_state_latents))
                z_true_csg2a_flip = torch.cat(
                    (
                        torch.ones(csg2a_state_latents.shape[0], device=device),
                        torch.zeros(t_csg2a_state_latents.shape[0], device=device),
                    )
                ).view(-1, 1)
                z_pred_csg2a_flip = ADV_csg2a_model(F_s_t_csg2a)
                if IsNaN(z_pred_csg2a_flip):
                    return -1

                adv_loss_flip = loss(z_pred_flip, z_true_flip)
                adv_csg2a_loss_flip = loss(z_pred_csg2a_flip, z_true_csg2a_flip)
                cdr_loss = loss(y_pred, y_true)
                total_loss = (
                    cdr_loss
                    + da_weight * lam * adv_loss_flip
                    + da_weight * self.lam_csg2a_adv * adv_csg2a_loss_flip
                )
                total_loss.backward()
                optimizer.step()
                set_requires_grad(ADV_model, True)
                set_requires_grad(ADV_csg2a_model, True)

                train_targets_epoch.append(y_true.detach().cpu())
                train_preds_epoch.append(y_pred.detach().cpu())

            y_val, pred_val = self._predict_loader(
                val_loader,
                EN_model,
                CSG2AState_model,
                CSG2ACompound_model,
                Fusion_dropout,
                GCN_model,
            )
            with torch.no_grad():
                EN_model.eval()
                CSG2AState_model.eval()
                ADV_model.eval()
                ADV_csg2a_model.eval()
                _, F_gexpr_val_mu, _ = EN_model(X_gexpr_val_raw)
                _, F_t_gexpr_val_mu, _ = EN_model(X_t_gexpr_val)
                F_gexpr_val = F_gexpr_val_mu
                F_t_gexpr_val = F_t_gexpr_val_mu
                F_g_t_gexpr_val = torch.cat((F_gexpr_val, F_t_gexpr_val))
                z_true_val = torch.cat(
                    (
                        torch.zeros(F_gexpr_val.shape[0], device=device),
                        torch.ones(F_t_gexpr_val.shape[0], device=device),
                    )
                ).view(-1, 1)
                z_pred_val = ADV_model(F_g_t_gexpr_val)
                raw_domain_acc_val = binary_accuracy_from_probs(z_true_val, z_pred_val)

                F_csg2a_val = CSG2AState_model(X_csg2a_val)
                F_t_csg2a_val = CSG2AState_model(X_t_csg2a_val)
                F_s_t_csg2a_val = torch.cat((F_csg2a_val, F_t_csg2a_val))
                z_true_csg2a_val = torch.cat(
                    (
                        torch.zeros(F_csg2a_val.shape[0], device=device),
                        torch.ones(F_t_csg2a_val.shape[0], device=device),
                    )
                ).view(-1, 1)
                z_pred_csg2a_val = ADV_csg2a_model(F_s_t_csg2a_val)
                csg2a_domain_acc_val = binary_accuracy_from_probs(
                    z_true_csg2a_val, z_pred_csg2a_val
                )

            auc_train = roc_auc_score(
                torch.cat(train_targets_epoch).numpy(),
                torch.cat(train_preds_epoch).numpy(),
            )
            auc_val = roc_auc_score(y_val, pred_val)

            print(
                "epoch %d | da_w: %.3f | raw domain acc: %.4f | "
                "csg2a-state domain acc: %.4f | GDSC train AUROC: %.4f | "
                "GDSC val AUROC: %.4f"
                % (
                    epoch,
                    da_weight,
                    raw_domain_acc_val,
                    csg2a_domain_acc_val,
                    auc_train,
                    auc_val,
                )
            )

            if auc_val >= best_auc:
                wait = 0
                best_auc = auc_val
                torch.save(
                    {
                        "EN_model": EN_model.state_dict(),
                        "CSG2AState_model": CSG2AState_model.state_dict(),
                        "CSG2ACompound_model": CSG2ACompound_model.state_dict(),
                        "GCN_model": GCN_model.state_dict(),
                        "ADV_model": ADV_model.state_dict(),
                        "ADV_csg2a_model": ADV_csg2a_model.state_dict(),
                        "params": params,
                        "csg2a_state_hidden_dim": self.csg2a_state_hidden_dim,
                        "csg2a_state_out_dim": self.csg2a_state_out_dim,
                        "csg2a_comp_hidden_dim": self.csg2a_comp_hidden_dim,
                        "csg2a_comp_out_dim": self.csg2a_comp_out_dim,
                        "csg2a_dropout": self.csg2a_dropout,
                        "fusion_dropout": self.fusion_dropout,
                        "use_gcn_dropout": self.use_gcn_dropout,
                        "weight_decay": self.weight_decay,
                        "adv_weight_decay": self.adv_weight_decay,
                        "lam_csg2a_adv": self.lam_csg2a_adv,
                        "da_start_epoch": self.da_start_epoch,
                        "da_warmup_epochs": self.da_warmup_epochs,
                    },
                    weight_path,
                )
            else:
                wait += 1
                if wait >= self.patience:
                    break

        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        EN_model.load_state_dict(checkpoint["EN_model"])
        CSG2AState_model.load_state_dict(checkpoint["CSG2AState_model"])
        CSG2ACompound_model.load_state_dict(checkpoint["CSG2ACompound_model"])
        GCN_model.load_state_dict(checkpoint["GCN_model"])
        y_test, pred_test = self._predict_loader(
            test_loader,
            EN_model,
            CSG2AState_model,
            CSG2ACompound_model,
            Fusion_dropout,
            GCN_model,
        )
        test_auc = roc_auc_score(y_test, pred_test)
        print("External AUROC after checkpoint selection: %.4f" % test_auc)
        return test_auc
