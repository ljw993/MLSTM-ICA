import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import random
import os

# ------------------- Global Settings -------------------
warnings.filterwarnings('ignore')

# [Control Switch]
# True: Skip training, load .pth directly for plotting
# False: Retrain model
ONLY_PLOT = True


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ------------------- Core Modules (MLSTM-ICA Components) -------------------

class GRN(nn.Module):
    """Gated Residual Network"""

    def __init__(self, input_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.W2 = nn.Linear(input_dim, hidden_dim)
        self.W3 = nn.Linear(input_dim, hidden_dim)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))
        self.W1 = nn.Linear(hidden_dim, input_dim)
        self.b1 = nn.Parameter(torch.zeros(input_dim))
        self.glu = nn.Linear(input_dim, input_dim * 2)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, a, c=None):
        val = self.W2(a) + self.b2
        if c is not None:
            val = val + self.W3(c)
        eta2 = torch.nn.functional.elu(val)
        eta1 = self.W1(eta2) + self.b1
        gates = self.glu(eta1)
        gate, skip = gates.chunk(2, dim=-1)
        gated_output = torch.sigmoid(gate) * skip
        return self.layer_norm(a + self.dropout(gated_output))


class ConvAttention(nn.Module):
    """Convolutional Attention (Intra-variable correlation)"""

    def __init__(self, hidden_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.attn = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden]
        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        attn_weights = torch.softmax(self.attn(conv_out), dim=-1)
        weighted = torch.sum(attn_weights * x, dim=1)
        return weighted


class CrossAttention(nn.Module):
    """Cross Attention (Inter-stage correlation)"""

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32).to(query.device))

        attn_weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.out_linear(attended)


class WeightedFusion(nn.Module):
    """Adaptive Fusion Layer"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, dec_out, enc_out):
        combined = torch.cat([dec_out, enc_out], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        fused = torch.tanh(self.fuse(combined))
        return gate * fused + (1 - gate) * enc_out


class SafeMogrifierLSTM(nn.Module):
    """LSTM improved with Mogrifier interactions"""

    def __init__(self, input_dim, hidden_dim, num_layers=1, mogrify_steps=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mogrify_steps = mogrify_steps
        self.mogrifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, hidden_dim + input_dim)
            ) for _ in range(mogrify_steps)
        ])
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        else:
            h, c = hidden

        h_lstm = h.clone()
        c_lstm = c.clone()
        x_original = x.clone()

        # Mogrification steps
        for i in range(self.mogrify_steps):
            h_last = h_lstm[-1].unsqueeze(1)
            h_expanded = h_last.expand(-1, seq_len, -1)
            combined = torch.cat([x, h_expanded], dim=-1)
            gate = self.mogrifier[i](combined)
            x_gate, h_gate = torch.split(gate, [x.size(-1), self.hidden_dim], dim=-1)
            x = x_original * torch.sigmoid(x_gate)
            new_h_last = h_last * torch.sigmoid(h_gate[:, -1:, :])
            if self.num_layers > 1:
                h_lstm = torch.cat([h_lstm[:-1], new_h_last.squeeze(1).unsqueeze(0)], dim=0)
            else:
                h_lstm = new_h_last.squeeze(1).unsqueeze(0)

        out, (h, c) = self.lstm(x, (h_lstm.contiguous(), c_lstm.contiguous()))
        return out, (h, c)


# ------------------- Main Model: MLSTM-ICA -------------------

class MLSTM_ICA(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, hidden_dim, target_dim, pred_steps,
                 num_layers=2, mogrify_steps=3, num_heads=4):
        super().__init__()
        self.pred_steps = pred_steps
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim

        # 1. Encoder & Decoder (Mogrifier Enhanced)
        self.encoder = SafeMogrifierLSTM(encoder_dim, hidden_dim, num_layers, mogrify_steps)
        self.decoder = SafeMogrifierLSTM(decoder_dim, hidden_dim, num_layers, mogrify_steps)

        # 2. Gated Residual Network (GRN)
        self.grn = GRN(hidden_dim, hidden_dim * 2)

        # 3. Attention Mechanisms
        self.conv_attn_encoder = ConvAttention(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads=num_heads)

        # 4. Fusion Layer
        self.weighted_fusion = WeightedFusion(hidden_dim)

        # 5. Output Head
        self.head = nn.Linear(hidden_dim, target_dim * pred_steps)

    def forward(self, x_enc, x_dec):
        batch_size = x_enc.size(0)

        # 1. Temporal Processing
        enc_out, (h_enc, _) = self.encoder(x_enc)
        dec_out, (h_dec, _) = self.decoder(x_dec)

        # 2. Decoder Enhancement Path (GRN)
        dec_mean = dec_out.mean(dim=1)
        enc_mean = enc_out.mean(dim=1)
        dec_feat = self.grn(a=dec_mean, c=enc_mean)

        # 3. Encoder Feature Extraction Path (Attentions)
        # 3a. Intra-series correlation (Conv Attn)
        conv_feat = self.conv_attn_encoder(enc_out)

        # 3b. Inter-series correlation (Cross Attn)
        last_h_dec = h_dec[-1].unsqueeze(1)
        cross_feat = self.cross_attn(query=last_h_dec, key=enc_out, value=enc_out).squeeze(1)

        # Combine Attentions
        attn_output = conv_feat + cross_feat

        # Dimension alignment
        if dec_feat.dim() > 2: dec_feat = dec_feat.view(batch_size, -1)
        if attn_output.dim() > 2: attn_output = attn_output.view(batch_size, -1)

        # 4. Final Adaptive Fusion
        final_feat = self.weighted_fusion(dec_feat, attn_output)

        output = self.head(final_feat)
        return output.view(-1, self.pred_steps, self.target_dim)


# ------------------- Data Processing -------------------

class SafeSeparatedInputProcessor:
    def __init__(self, encoder_features, decoder_features, target_cols, window_size=15, pred_steps=1):
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        self.target_cols = target_cols
        self.window_size = window_size
        self.pred_steps = pred_steps
        self.encoder_scaler = MinMaxScaler()
        self.decoder_scaler = MinMaxScaler()
        self.target_scalers = {col: MinMaxScaler() for col in target_cols}
        self._fitted = False

    def load_data(self, csv_path, fit_scalers=False):
        raw_data = pd.read_csv(csv_path)
        encoder_data = raw_data[self.encoder_features].values
        decoder_data = raw_data[self.decoder_features].values
        target_data = raw_data[self.target_cols].values

        if fit_scalers or not self._fitted:
            scaled_encoder = self.encoder_scaler.fit_transform(encoder_data)
            scaled_decoder = self.decoder_scaler.fit_transform(decoder_data)
            scaled_target = np.hstack([
                self.target_scalers[col].fit_transform(target_data[:, i].reshape(-1, 1))
                for i, col in enumerate(self.target_cols)
            ])
            self._fitted = True
        else:
            scaled_encoder = self.encoder_scaler.transform(encoder_data)
            scaled_decoder = self.decoder_scaler.transform(decoder_data)
            scaled_target = np.hstack([
                self.target_scalers[col].transform(target_data[:, i].reshape(-1, 1))
                for i, col in enumerate(self.target_cols)
            ])

        X_enc, X_dec, y = [], [], []
        max_index = len(scaled_encoder) - self.window_size - self.pred_steps + 1
        for i in range(max_index):
            X_enc.append(scaled_encoder[i:i + self.window_size])
            X_dec.append(scaled_decoder[i:i + self.window_size])
            y.append(scaled_target[i + self.window_size:i + self.window_size + self.pred_steps])

        return (
            torch.FloatTensor(np.array(X_enc)),
            torch.FloatTensor(np.array(X_dec)),
            torch.FloatTensor(np.array(y))
        )

    def inverse_transform_targets(self, scaled_data):
        results = []
        for i, col in enumerate(self.target_cols):
            results.append(
                self.target_scalers[col].inverse_transform(scaled_data[:, i].reshape(-1, 1))
            )
        return np.hstack(results)


class TimeSeriesDataset(Dataset):
    def __init__(self, X_enc, X_dec, y):
        self.X_enc = X_enc
        self.X_dec = X_dec
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_enc[idx], self.X_dec[idx], self.y[idx]


# ------------------- Training System -------------------

class ForecastingSystem:
    def __init__(self, model, processor, save_path, lr=1e-3, patience=7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.processor = processor
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

        # Fixed to Huber Loss as per MLSTM-ICA config
        self.criterion = nn.HuberLoss(delta=1.0)

        self.best_mape = np.inf
        self.patience = patience
        self.counter = 0
        self.save_path = save_path
        self.train_loss_history = []

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for x_enc, x_dec, targets in train_loader:
            x_enc = x_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x_enc, x_dec)
            loss = self.criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        self.train_loss_history.append(avg_loss)
        return avg_loss

    def evaluate(self, loader, return_predictions=False):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x_enc, x_dec, targets in loader:
                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                preds = self.model(x_enc, x_dec)
                y_true.append(targets.cpu().numpy())
                y_pred.append(preds.cpu().numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_true_unscaled = self.processor.inverse_transform_targets(y_true[:, 0, :])
        y_pred_unscaled = self.processor.inverse_transform_targets(y_pred[:, 0, :])
        metrics = {}
        for i, col in enumerate(self.processor.target_cols):
            r2 = r2_score(y_true_unscaled[:, i], y_pred_unscaled[:, i])
            mape = np.mean(np.abs((y_true_unscaled[:, i] - y_pred_unscaled[:, i]) /
                                  (y_true_unscaled[:, i] + 1e-8))) * 100
            mae = mean_absolute_error(y_true_unscaled[:, i], y_pred_unscaled[:, i])
            rmse = np.sqrt(mean_squared_error(y_true_unscaled[:, i], y_pred_unscaled[:, i]))
            metrics[col] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE(%)': mape}
        if return_predictions:
            return metrics, y_true_unscaled, y_pred_unscaled
        return metrics

    def early_stopping(self, val_metrics):
        current_mape = np.mean([m['MAPE(%)'] for m in val_metrics.values()])
        if current_mape < self.best_mape:
            print(
                f"    >> Better model found (MAPE: {self.best_mape:.2f}% -> {current_mape:.2f}%), saving to {self.save_path}")
            self.best_mape = current_mape
            self.counter = 0
            torch.save(self.model.state_dict(), self.save_path)
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# ------------------- Plotting -------------------

def plot_predictions(true_values, pred_values, target_names=None, title_text=""):
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
        'font.size': 18,
        'mathtext.default': 'regular'
    })

    if target_names is None:
        target_names = [f"Target_{i}" for i in range(true_values.shape[1])]

    for i, col in enumerate(target_names):
        plt.figure(figsize=(10, 6), dpi=600)

        y_min_data = min(true_values[:, i].min(), pred_values[:, i].min())
        y_max_data = max(true_values[:, i].max(), pred_values[:, i].max())
        y_range = y_max_data - y_min_data

        plt.plot(true_values[:, i], label='Actual', linewidth=1.5, alpha=0.8)
        plt.plot(pred_values[:, i], label='Predicted', linewidth=1.5)

        plt.xlabel('Sample NO.', fontsize=20)

        if col == 'Tor1' or col == 'Tor':
            tick_interval = 500
            ylabel_text = 'Torque (kN·m)'

            target_max_y = 2500

            y_ticks_start = np.floor(y_min_data / tick_interval) * tick_interval

            y_ticks = np.arange(y_ticks_start, target_max_y + 1, tick_interval)

            plt.ylim(y_min_data - 0.05 * y_range, target_max_y)
            plt.yticks(y_ticks)

        elif col == 'Th1' or col == 'Th':
            tick_interval = 2500
            ylabel_text = 'Thrust (kN)'

            target_min_y = 2500

            y_max_adj = y_max_data + tick_interval

            y_ticks = np.arange(target_min_y, y_max_adj + tick_interval, tick_interval)

            plt.ylim(target_min_y, y_max_adj)
            plt.yticks(y_ticks)

        else:
            tick_interval = (y_range / 5) if y_range != 0 else 1
            ylabel_text = col

            y_max_adj = y_max_data + tick_interval
            y_ticks_start = np.floor(y_min_data / tick_interval) * tick_interval
            y_ticks_end = y_max_adj + tick_interval
            y_ticks = np.arange(y_ticks_start, y_ticks_end, tick_interval)

            plt.ylim(y_min_data - 0.05 * y_range, y_max_adj)
            plt.yticks(y_ticks)

        plt.ylabel(ylabel_text, fontsize=20)

        plt.title(f'Prediction with MLSTM-ICA', fontsize=22, pad=10)

        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=14)

        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.tight_layout()

        safe_suffix = title_text.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '')
        file_name = f"Prediction_{safe_suffix}_{col}.png"
        plt.savefig(file_name, bbox_inches='tight', dpi=600)
        print(f"    -> Plot saved to {file_name}")
        plt.close()


# ------------------- Main Execution -------------------

if __name__ == "__main__":

    config = {
        'hidden_dim': 64,
        'num_layers': 1,
        'batch_size': 32,
        'mogrify_steps': 4,
        'num_heads': 2,
        'lr': 0.001,
        'patience': 50,
        'max_epochs': 50
    }

    # Data Path
    csv_path = 'your data.csv'

    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
    else:
        print(f"Loading data from {csv_path}...")
        processor = SafeSeparatedInputProcessor(
            encoder_features=['Tor1', 'Th1'],
            decoder_features=['PR', 'F', 'RPM'],
            target_cols=['Tor1', 'Th1'],
            window_size=15,
            pred_steps=1
        )
        X_enc, X_dec, y = processor.load_data(csv_path, fit_scalers=True)

        split_idx = int(0.8 * len(X_enc))
        train_set = TimeSeriesDataset(X_enc[:split_idx], X_dec[:split_idx], y[:split_idx])
        val_set = TimeSeriesDataset(X_enc[split_idx:], X_dec[split_idx:], y[split_idx:])

        train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

        # Initialize MLSTM-ICA Model
        print(f"\n{'=' * 20} Initializing MLSTM-ICA {'=' * 20}")
        model_name = "MLSTM-ICA"
        save_path = f"best_model_{model_name}.pth"

        # Instantiate specific MLSTM-ICA Class
        model = MLSTM_ICA(
            encoder_dim=len(processor.encoder_features),
            decoder_dim=len(processor.decoder_features),
            hidden_dim=config['hidden_dim'],
            target_dim=len(processor.target_cols),
            pred_steps=1,
            num_layers=config['num_layers'],
            mogrify_steps=config['mogrify_steps'],
            num_heads=config['num_heads']
        )

        system = ForecastingSystem(
            model, processor,
            save_path=save_path,
            lr=config['lr'],
            patience=config['patience']
        )

        if not ONLY_PLOT:
            print("Starting Training...")
            #  - Skipped: No external query needed for internal architecture
            for epoch in range(config['max_epochs']):
                train_loss = system.train_epoch(train_loader)
                val_metrics = system.evaluate(val_loader)
                mape_avg = np.mean([m['MAPE(%)'] for m in val_metrics.values()])

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}: Train Loss (Huber)={train_loss:.4f}, Val Avg MAPE={mape_avg:.2f}%")

                if system.early_stopping(val_metrics):
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break
        else:
            print("  [Mode: ONLY_PLOT] Skipping training...")

        # Final Evaluation
        if os.path.exists(save_path):
            print(f"  Loading model from: {save_path}")
            model.load_state_dict(torch.load(save_path))

            final_metrics, y_true, y_pred = system.evaluate(val_loader, return_predictions=True)
            plot_predictions(y_true, y_pred, processor.target_cols, title_text=model_name)

            print(f"\n{'=' * 20} Final Results ({model_name}) {'=' * 20}")
            for col, res in final_metrics.items():
                print(f"{col}: R2={res['R2']:.4f}, MAE={res['MAE']:.4f}, MAPE={res['MAPE(%)']:.2f}%")
        else:
            print(f"  Error: Model file {save_path} not found.")
