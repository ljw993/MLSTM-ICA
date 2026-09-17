import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.filterwarnings("ignore")

ONLY_PLOT = False
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


class MogrifierLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, mogrify_steps=3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mogrify_steps = mogrify_steps

        self.q_layers = nn.ModuleDict()
        self.r_layers = nn.ModuleDict()

        for i in range(1, mogrify_steps + 1):
            if i % 2 == 1:
                self.q_layers[str(i)] = nn.Linear(
                    hidden_dim,
                    input_dim,
                    bias=True
                )
            else:
                self.r_layers[str(i)] = nn.Linear(
                    input_dim,
                    hidden_dim,
                    bias=True
                )

        self.lstm_cell = nn.LSTMCell(
            input_size=input_dim,
            hidden_size=hidden_dim
        )

    def forward(self, x_t, h_prev, c_prev):
        x_m = x_t
        h_m = h_prev

        for i in range(1, self.mogrify_steps + 1):
            if i % 2 == 1:
                gate_x = 2.0 * torch.sigmoid(
                    self.q_layers[str(i)](h_m)
                )
                x_m = gate_x * x_m
            else:
                gate_h = 2.0 * torch.sigmoid(
                    self.r_layers[str(i)](x_m)
                )
                h_m = gate_h * h_m

        h_new, c_new = self.lstm_cell(
            x_m,
            (h_m, c_prev)
        )

        return h_new, c_new


class MogrifierLSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=2,
        mogrify_steps=3,
        dropout=0.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mogrify_steps = mogrify_steps
        self.dropout_rate = dropout

        self.cells = nn.ModuleList()

        for layer_idx in range(num_layers):
            current_input_dim = (
                input_dim
                if layer_idx == 0
                else hidden_dim
            )

            self.cells.append(
                MogrifierLSTMCell(
                    input_dim=current_input_dim,
                    hidden_dim=hidden_dim,
                    mogrify_steps=mogrify_steps
                )
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.shape
        device = x.device

        if hidden is None:
            h_states = [
                torch.zeros(
                    batch_size,
                    self.hidden_dim,
                    device=device
                )
                for _ in range(self.num_layers)
            ]

            c_states = [
                torch.zeros(
                    batch_size,
                    self.hidden_dim,
                    device=device
                )
                for _ in range(self.num_layers)
            ]
        else:
            h_init, c_init = hidden

            h_states = [
                h_init[i]
                for i in range(self.num_layers)
            ]

            c_states = [
                c_init[i]
                for i in range(self.num_layers)
            ]

        outputs = []

        for t in range(seq_len):
            layer_input = x[:, t, :]

            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(
                    layer_input,
                    h_states[layer_idx],
                    c_states[layer_idx]
                )

                h_states[layer_idx] = h_new
                c_states[layer_idx] = c_new
                layer_input = h_new

                if (
                    self.dropout_rate > 0
                    and layer_idx < self.num_layers - 1
                ):
                    layer_input = self.dropout(layer_input)

            outputs.append(layer_input.unsqueeze(1))

        output = torch.cat(outputs, dim=1)
        h_n = torch.stack(h_states, dim=0)
        c_n = torch.stack(c_states, dim=0)

        return output, (h_n, c_n)


class GRN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        dropout=0.1
    ):
        super().__init__()

        hidden_dim = (
            hidden_dim
            if hidden_dim is not None
            else input_dim
        )

        self.fc_a = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc_c = nn.Linear(
            input_dim,
            hidden_dim
        )

        self.fc_out = nn.Linear(
            hidden_dim,
            input_dim
        )

        self.gate_fc = nn.Linear(
            input_dim,
            input_dim
        )

        self.value_fc = nn.Linear(
            input_dim,
            input_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(
            input_dim
        )

        self.elu = nn.ELU()

    def forward(self, a, c=None):
        eta2 = self.fc_a(a)

        if c is not None:
            eta2 = eta2 + self.fc_c(c)

        eta2 = self.elu(eta2)
        eta1 = self.fc_out(eta2)
        eta1 = self.dropout(eta1)

        gate = torch.sigmoid(
            self.gate_fc(eta1)
        )

        value = self.value_fc(
            eta1
        )

        glu_output = gate * value

        output = self.layer_norm(
            a + glu_output
        )

        return output


class ConvolutionalAttentionModule(nn.Module):
    def __init__(
        self,
        hidden_dim,
        kernel_size=3
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.score = nn.Linear(
            hidden_dim,
            1
        )

        self.sigmoid_gate = nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def forward(self, x):
        conv_out = self.conv(
            x.transpose(1, 2)
        ).transpose(1, 2)

        score = self.score(
            conv_out
        )

        alpha = torch.softmax(
            score,
            dim=1
        )

        local_gate = torch.sigmoid(
            self.sigmoid_gate(
                conv_out
            )
        )

        local_enhanced = (
            x * local_gate
        )

        seq_len = x.size(1)

        enhanced_sequence = (
            local_enhanced
            * alpha
            * seq_len
        )

        global_feature = torch.sum(
            alpha * x,
            dim=1
        )

        return (
            enhanced_sequence,
            global_feature,
            alpha
        )


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.out_proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        query,
        key,
        value
    ):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores * self.scale

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attended = torch.matmul(
            attention_weights,
            V
        )

        output = self.out_proj(
            attended
        )

        return output, attention_weights


class WeightedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.proj = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.layer_norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(self, feature_a, feature_b):
        combined = torch.cat(
            [
                feature_a,
                feature_b
            ],
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate(combined)
        )

        candidate = torch.tanh(
            self.proj(combined)
        )

        fused = (
            gate * candidate
            +
            (1.0 - gate) * feature_b
        )

        fused = self.layer_norm(fused)

        return fused


class MLSTM_ICA(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        hidden_dim,
        target_dim,
        pred_steps=1,
        num_layers=2,
        mogrify_steps=3,
        dropout=0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.pred_steps = pred_steps

        self.encoder = MogrifierLSTM(
            input_dim=encoder_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mogrify_steps=mogrify_steps,
            dropout=dropout
        )

        self.decoder = MogrifierLSTM(
            input_dim=decoder_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mogrify_steps=mogrify_steps,
            dropout=dropout
        )

        self.cam = ConvolutionalAttentionModule(
            hidden_dim=hidden_dim,
            kernel_size=3
        )

        self.grn = GRN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )

        self.cross_attention = CrossAttention(
            hidden_dim
        )

        self.fusion = WeightedFusion(
            hidden_dim
        )

        self.output_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                target_dim * pred_steps
            )
        )

    def forward(
        self,
        x_enc,
        x_dec,
        return_attention=False
    ):
        enc_out, (
            h_enc,
            c_enc
        ) = self.encoder(
            x_enc
        )

        dec_out, (
            h_dec,
            c_dec
        ) = self.decoder(
            x_dec,
            hidden=(
                h_enc,
                c_enc
            )
        )

        cam_sequence, \
        cam_global, \
        cam_weights = self.cam(
            enc_out
        )

        encoder_context = enc_out.mean(
            dim=1,
            keepdim=True
        )

        encoder_context = encoder_context.expand(
            -1,
            dec_out.size(1),
            -1
        )

        grn_output = self.grn(
            a=dec_out,
            c=encoder_context
        )

        cross_output, \
        cross_weights = self.cross_attention(
            query=grn_output,
            key=cam_sequence,
            value=cam_sequence
        )

        fused_sequence = self.fusion(
            cross_output,
            grn_output
        )

        final_feature = fused_sequence[
            :,
            -1,
            :
        ]

        output = self.output_head(
            final_feature
        )

        output = output.view(
            -1,
            self.pred_steps,
            self.target_dim
        )

        if return_attention:
            return (
                output,
                cam_weights,
                cross_weights
            )

        return output


class LearnableWeightedHuberLoss(nn.Module):
    def __init__(
        self,
        target_dim,
        delta=0.1
    ):
        super().__init__()

        self.target_dim = target_dim
        self.delta = delta

        self.weight_logits = nn.Parameter(
            torch.zeros(target_dim)
        )

    def forward(
        self,
        prediction,
        target
    ):
        error = (
            prediction - target
        ).abs()

        quadratic = torch.clamp(
            error,
            max=self.delta
        )

        linear = (
            error - quadratic
        )

        huber = (
            0.5 * quadratic.pow(2)
            +
            self.delta * linear
        )

        weights = torch.softmax(
            self.weight_logits,
            dim=0
        )

        weights = (
            weights
            * self.target_dim
        )

        weighted_huber = (
            huber
            * weights.view(
                1,
                1,
                -1
            )
        )

        return weighted_huber.mean()

    def get_weights(self):
        with torch.no_grad():
            weights = torch.softmax(
                self.weight_logits,
                dim=0
            )

            weights = (
                weights
                * self.target_dim
            )

        return weights.cpu().numpy()


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        X_enc,
        X_dec,
        y
    ):
        self.X_enc = X_enc
        self.X_dec = X_dec
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_enc[idx],
            self.X_dec[idx],
            self.y[idx]
        )


class SeparatedInputProcessor:
    def __init__(
        self,
        encoder_features,
        decoder_features,
        target_cols,
        window_size=15,
        pred_steps=1
    ):
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        self.target_cols = target_cols
        self.window_size = window_size
        self.pred_steps = pred_steps

        self.encoder_scaler = MinMaxScaler()
        self.decoder_scaler = MinMaxScaler()

        self.target_scalers = {
            col: MinMaxScaler()
            for col in target_cols
        }

        self.fitted = False

    def check_data(self, data):
        required = list(
            dict.fromkeys(
                self.encoder_features
                +
                self.decoder_features
                +
                self.target_cols
            )
        )

        missing = [
            col
            for col in required
            if col not in data.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        values = data[
            required
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        if values.isnull().any().any():
            bad_cols = values.columns[
                values.isnull().any()
            ].tolist()

            raise ValueError(
                f"NaN or non-numeric values found in: {bad_cols}"
            )

        if not np.isfinite(
            values.values
        ).all():
            raise ValueError(
                "Inf or -Inf values found."
            )

    def fit_scalers(
        self,
        train_data
    ):
        enc = train_data[
            self.encoder_features
        ].values

        dec = train_data[
            self.decoder_features
        ].values

        target = train_data[
            self.target_cols
        ].values

        self.encoder_scaler.fit(enc)
        self.decoder_scaler.fit(dec)

        for i, col in enumerate(
            self.target_cols
        ):
            self.target_scalers[
                col
            ].fit(
                target[:, i].reshape(
                    -1,
                    1
                )
            )

        self.fitted = True

    def transform_segment(
        self,
        raw_data
    ):
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        encoder_data = raw_data[
            self.encoder_features
        ].values

        decoder_data = raw_data[
            self.decoder_features
        ].values

        target_data = raw_data[
            self.target_cols
        ].values

        scaled_encoder = (
            self.encoder_scaler
            .transform(
                encoder_data
            )
        )

        scaled_decoder = (
            self.decoder_scaler
            .transform(
                decoder_data
            )
        )

        scaled_targets = []

        for i, col in enumerate(
            self.target_cols
        ):
            transformed = (
                self.target_scalers[
                    col
                ].transform(
                    target_data[
                        :,
                        i
                    ].reshape(
                        -1,
                        1
                    )
                )
            )

            scaled_targets.append(
                transformed
            )

        scaled_target = np.hstack(
            scaled_targets
        )

        return self.make_windows(
            scaled_encoder,
            scaled_decoder,
            scaled_target
        )

    def make_windows(
        self,
        scaled_encoder,
        scaled_decoder,
        scaled_target
    ):
        X_enc = []
        X_dec = []
        y = []

        max_index = (
            len(scaled_encoder)
            -
            self.window_size
            -
            self.pred_steps
            +
            1
        )

        if max_index <= 0:
            raise ValueError(
                "Insufficient data length."
            )

        for i in range(max_index):
            X_enc.append(
                scaled_encoder[
                    i:
                    i + self.window_size
                ]
            )

            X_dec.append(
                scaled_decoder[
                    i:
                    i + self.window_size
                ]
            )

            y.append(
                scaled_target[
                    i + self.window_size:
                    i + self.window_size + self.pred_steps
                ]
            )

        X_enc = torch.tensor(
            np.array(X_enc),
            dtype=torch.float32
        )

        X_dec = torch.tensor(
            np.array(X_dec),
            dtype=torch.float32
        )

        y = torch.tensor(
            np.array(y),
            dtype=torch.float32
        )

        return X_enc, X_dec, y

    def inverse_transform_targets(
        self,
        scaled_data
    ):
        results = []

        for i, col in enumerate(
            self.target_cols
        ):
            restored = self.target_scalers[
                col
            ].inverse_transform(
                scaled_data[
                    :,
                    i
                ].reshape(
                    -1,
                    1
                )
            )

            results.append(
                restored
            )

        return np.hstack(results)


class ForecastingSystem:
    def __init__(
        self,
        model,
        processor,
        save_path,
        lr=1e-3,
        patience=10,
        huber_delta=0.1
    ):
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = model.to(
            self.device
        )

        self.processor = processor

        self.criterion = LearnableWeightedHuberLoss(
            target_dim=len(
                processor.target_cols
            ),
            delta=huber_delta
        ).to(
            self.device
        )

        parameters = (
            list(
                self.model.parameters()
            )
            +
            list(
                self.criterion.parameters()
            )
        )

        self.optimizer = optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=1e-5
        )

        self.save_path = save_path
        self.best_val_loss = np.inf
        self.patience = patience
        self.counter = 0
        self.train_loss_history = []
        self.val_loss_history = []

    def train_epoch(
        self,
        loader
    ):
        self.model.train()
        self.criterion.train()

        total_loss = 0.0

        for (
            x_enc,
            x_dec,
            targets
        ) in loader:
            x_enc = x_enc.to(
                self.device
            )

            x_dec = x_dec.to(
                self.device
            )

            targets = targets.to(
                self.device
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                x_enc,
                x_dec
            )

            loss = self.criterion(
                outputs,
                targets
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = (
            total_loss
            /
            len(loader)
        )

        self.train_loss_history.append(
            avg_loss
        )

        return avg_loss

    def validation_loss(
        self,
        loader
    ):
        self.model.eval()
        self.criterion.eval()

        total_loss = 0.0

        with torch.no_grad():
            for (
                x_enc,
                x_dec,
                targets
            ) in loader:
                x_enc = x_enc.to(
                    self.device
                )

                x_dec = x_dec.to(
                    self.device
                )

                targets = targets.to(
                    self.device
                )

                outputs = self.model(
                    x_enc,
                    x_dec
                )

                loss = self.criterion(
                    outputs,
                    targets
                )

                total_loss += loss.item()

        avg_loss = (
            total_loss
            /
            len(loader)
        )

        self.val_loss_history.append(
            avg_loss
        )

        return avg_loss

    def evaluate(
        self,
        loader,
        return_predictions=False
    ):
        self.model.eval()

        true_list = []
        pred_list = []

        with torch.no_grad():
            for (
                x_enc,
                x_dec,
                targets
            ) in loader:
                x_enc = x_enc.to(
                    self.device
                )

                x_dec = x_dec.to(
                    self.device
                )

                preds = self.model(
                    x_enc,
                    x_dec
                )

                true_list.append(
                    targets.numpy()
                )

                pred_list.append(
                    preds.cpu().numpy()
                )

        y_true = np.concatenate(
            true_list,
            axis=0
        )

        y_pred = np.concatenate(
            pred_list,
            axis=0
        )

        if y_true.shape[1] != 1:
            raise ValueError(
                "Current evaluation supports pred_steps=1."
            )

        y_true = (
            y_true[:, 0, :]
        )

        y_pred = (
            y_pred[:, 0, :]
        )

        y_true_unscaled = (
            self.processor
            .inverse_transform_targets(
                y_true
            )
        )

        y_pred_unscaled = (
            self.processor
            .inverse_transform_targets(
                y_pred
            )
        )

        metrics = {}

        for i, col in enumerate(
            self.processor.target_cols
        ):
            true_col = (
                y_true_unscaled[
                    :,
                    i
                ]
            )

            pred_col = (
                y_pred_unscaled[
                    :,
                    i
                ]
            )

            r2 = r2_score(
                true_col,
                pred_col
            )

            mae = mean_absolute_error(
                true_col,
                pred_col
            )

            rmse = np.sqrt(
                mean_squared_error(
                    true_col,
                    pred_col
                )
            )

            eps = 1e-8

            mape = np.mean(
                np.abs(
                    true_col - pred_col
                )
                /
                np.maximum(
                    np.abs(true_col),
                    eps
                )
            ) * 100

            metrics[col] = {
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE(%)": mape
            }

        if return_predictions:
            return (
                metrics,
                y_true_unscaled,
                y_pred_unscaled
            )

        return metrics

    def early_stopping(
        self,
        val_loss
    ):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0

            torch.save(
                {
                    "model_state_dict":
                        self.model.state_dict(),

                    "loss_state_dict":
                        self.criterion.state_dict()
                },
                self.save_path
            )

            return False

        self.counter += 1

        return (
            self.counter
            >=
            self.patience
        )

    def load_best_model(
        self
    ):
        checkpoint = torch.load(
            self.save_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.criterion.load_state_dict(
            checkpoint[
                "loss_state_dict"
            ]
        )


def plot_predictions(
    true_values,
    pred_values,
    target_names,
    model_name="MLSTM-ICA"
):
    mpl.rcParams.update(
        {
            "font.family":
                "Times New Roman",

            "axes.titlesize":
                16,

            "axes.labelsize":
                16,

            "xtick.labelsize":
                16,

            "ytick.labelsize":
                16,

            "legend.fontsize":
                14,

            "font.size":
                16,

            "mathtext.default":
                "regular"
        }
    )

    for i, col in enumerate(
        target_names
    ):
        plt.figure(
            figsize=(10, 6),
            dpi=300
        )

        true_col = (
            true_values[
                :,
                i
            ]
        )

        pred_col = (
            pred_values[
                :,
                i
            ]
        )

        plt.plot(
            true_col,
            label="Actual",
            linewidth=1.2,
            alpha=0.85
        )

        plt.plot(
            pred_col,
            label="Predicted",
            linewidth=1.2
        )

        plt.xlabel(
            "Sample NO."
        )

        if col in [
            "Tor",
            "Tor1"
        ]:
            ylabel = (
                "Torque (kN·m)"
            )

        elif col in [
            "Th",
            "Th1"
        ]:
            ylabel = (
                "Thrust (kN)"
            )

        else:
            ylabel = col

        plt.ylabel(
            ylabel
        )

        plt.title(
            f"Prediction with {model_name}"
        )

        y_min = min(
            true_col.min(),
            pred_col.min()
        )

        y_max = max(
            true_col.max(),
            pred_col.max()
        )

        y_range = (
            y_max - y_min
        )

        if y_range == 0:
            y_range = 1.0

        margin = (
            0.05 * y_range
        )

        plt.ylim(
            y_min - margin,
            y_max + margin
        )

        plt.legend(
            loc="upper right",
            frameon=False
        )

        plt.tight_layout()

        filename = (
            f"Prediction_"
            f"{model_name}_"
            f"{col}.png"
        )

        plt.savefig(
            filename,
            bbox_inches="tight",
            dpi=600
        )

        plt.close()

        print(
            f"Plot saved: {filename}"
        )


def plot_loss_curve(
    train_loss,
    val_loss
):
    plt.figure(
        figsize=(8, 5),
        dpi=300
    )

    plt.plot(
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Weighted Huber Loss"
    )

    plt.legend(
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        "MLSTM_ICA_Loss.png",
        dpi=600
    )

    plt.close()


if __name__ == "__main__":

    config = {
        "hidden_dim": 128,
        "num_layers": 2,
        "batch_size": 32,
        "mogrify_steps": 3,
        "window_size": 15,
        "pred_steps": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "huber_delta": 0.1,
        "max_epochs": 50,
        "patience": 10
    }

    csv_path = "your data.csv"

    model_name = "MLSTM-ICA"

    save_path = (
        "best_model_MLSTM_ICA.pth"
    )

    if not os.path.exists(
        csv_path
    ):
        raise FileNotFoundError(
            f"File not found: {csv_path}"
        )

    raw_data = pd.read_csv(
        csv_path
    )

    processor = SeparatedInputProcessor(
        encoder_features=[
            "Tor1",
            "Th1"
        ],
        decoder_features=[
            "F",
            "PR",
            "RPM"
        ],
        target_cols=[
            "Tor1",
            "Th1"
        ],
        window_size=
            config["window_size"],
        pred_steps=
            config["pred_steps"]
    )

    processor.check_data(
        raw_data
    )

    total_len = len(
        raw_data
    )

    train_pool_end = int(
        total_len * 0.80
    )

    train_pool = raw_data.iloc[
        :train_pool_end
    ].reset_index(
        drop=True
    )

    test_data = raw_data.iloc[
        train_pool_end:
    ].reset_index(
        drop=True
    )

    train_end = int(
        len(train_pool)
        * 0.90
    )

    train_data = train_pool.iloc[
        :train_end
    ].reset_index(
        drop=True
    )

    val_data = train_pool.iloc[
        train_end:
    ].reset_index(
        drop=True
    )

    processor.fit_scalers(
        train_data
    )

    X_enc_train, \
    X_dec_train, \
    y_train = processor.transform_segment(
        train_data
    )

    X_enc_val, \
    X_dec_val, \
    y_val = processor.transform_segment(
        val_data
    )

    X_enc_test, \
    X_dec_test, \
    y_test = processor.transform_segment(
        test_data
    )

    train_dataset = TimeSeriesDataset(
        X_enc_train,
        X_dec_train,
        y_train
    )

    val_dataset = TimeSeriesDataset(
        X_enc_val,
        X_dec_val,
        y_val
    )

    test_dataset = TimeSeriesDataset(
        X_enc_test,
        X_dec_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=
            config["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=
            config["batch_size"],
        shuffle=False
    )

    model = MLSTM_ICA(
        encoder_dim=len(
            processor.encoder_features
        ),
        decoder_dim=len(
            processor.decoder_features
        ),
        hidden_dim=
            config["hidden_dim"],
        target_dim=len(
            processor.target_cols
        ),
        pred_steps=
            config["pred_steps"],
        num_layers=
            config["num_layers"],
        mogrify_steps=
            config["mogrify_steps"],
        dropout=
            config["dropout"]
    )

    system = ForecastingSystem(
        model=model,
        processor=processor,
        save_path=save_path,
        lr=
            config["learning_rate"],
        patience=
            config["patience"],
        huber_delta=
            config["huber_delta"]
    )

    print(
        f"Device: {system.device}"
    )

    if not ONLY_PLOT:

        for epoch in range(
            config["max_epochs"]
        ):
            train_loss = (
                system.train_epoch(
                    train_loader
                )
            )

            val_loss = (
                system.validation_loss(
                    val_loader
                )
            )

            weights = (
                system.criterion
                .get_weights()
            )

            print(
                f"Epoch "
                f"{epoch + 1:03d}/"
                f"{config['max_epochs']} "
                f"| Train Loss="
                f"{train_loss:.6f} "
                f"| Val Loss="
                f"{val_loss:.6f} "
                f"| Weights="
                f"{weights}"
            )

            if system.early_stopping(
                val_loss
            ):
                print(
                    f"Early stopping at "
                    f"epoch {epoch + 1}"
                )
                break

        plot_loss_curve(
            system.train_loss_history,
            system.val_loss_history
        )

    if not os.path.exists(
        save_path
    ):
        raise FileNotFoundError(
            f"Model not found: {save_path}"
        )

    system.load_best_model()

    final_metrics, \
    y_true, \
    y_pred = system.evaluate(
        test_loader,
        return_predictions=True
    )

    plot_predictions(
        true_values=y_true,
        pred_values=y_pred,
        target_names=
            processor.target_cols,
        model_name=model_name
    )

    print(
        "\nFinal Test Results"
    )

    for col, result in (
        final_metrics.items()
    ):
        print(
            f"{col}: "
            f"R2={result['R2']:.4f}, "
            f"MAE={result['MAE']:.4f}, "
            f"RMSE={result['RMSE']:.4f}, "
            f"MAPE={result['MAPE(%)']:.2f}%"
        )

    print(
        "Weighted Huber weights:"
    )

    learned_weights = (
        system.criterion
        .get_weights()
    )

    for col, weight in zip(
        processor.target_cols,
        learned_weights
    ):
        print(
            f"{col}: {weight:.4f}"
        )
